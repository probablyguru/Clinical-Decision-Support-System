from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (299, 299)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorModel, self).__init__()
        self.base_model = models.inception_v3(weights='DEFAULT', aux_logits=True)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        score = output[:, class_idx]
        score.backward()
        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().detach().cpu().numpy()


print("Loading models...")
tumor_model = BrainTumorModel(num_classes=4)
tumor_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
tumor_model = tumor_model.to(DEVICE)
tumor_model.eval()

gradcam = GradCAM(tumor_model, tumor_model.base_model.Mixed_7c)

try:
    embedder = SentenceTransformer('allenai/aspire-sentence-embedder')
except:
    try:
        embedder = SentenceTransformer('pritamdeka/PubMedBERT-base-embeddings')
    except:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ All models loaded!")

app = Flask(__name__)
CORS(app)


def predict_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = tumor_model(img_tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, 2)
    return img, img_tensor, probs.cpu().numpy(), top_idxs.cpu().numpy(), top_probs.cpu().numpy()


def generate_assessment(ai_pred, ai_conf, all_probs, top_idxs, doctor_report, similarity):
    assessment = {
        'agreement': float(similarity),
        'ai_prediction': ai_pred,
        'doctor_prediction': None,
        'warning': None,
        'recommendation': None,
        'warning_flag': 'NONE',
        'agreement_level': '🟢 EXCELLENT'
    }
    recs = {
        'GLIOMA': "High-grade suspicion. Immediate neurosurgical consult and contrast MRI required.",
        'MENINGIOMA': "Potential benign growth. Schedule follow-up imaging in 3-6 months.",
        'PITUITARY': "Endocrine panel required. Check prolactin and growth hormone levels.",
        'NOTUMOR': "No malignancy detected. Continue routine clinical monitoring."
    }
    doctor_report_lower = doctor_report.lower()
    for cls in CLASSES:
        if cls in doctor_report_lower:
            assessment['doctor_prediction'] = cls.upper()
            break

    if ai_conf < 0.60:
        assessment['warning_flag'] = 'LOW_CONFIDENCE'
        assessment['warning'] = "⚠️ LOW AI CONFIDENCE: Below 60%"

    if len(all_probs) >= 2:
        gap = all_probs[int(top_idxs[0])] - all_probs[int(top_idxs[1])]
        if gap < 0.10:
            assessment['warning_flag'] = 'AMBIGUOUS'
            assessment['warning'] = "⚠️ AMBIGUOUS: Top 2 predictions very close"

    if assessment['doctor_prediction'] and assessment['doctor_prediction'] != ai_pred:
        doctor_prob = all_probs[CLASSES.index(assessment['doctor_prediction'].lower())]
        if doctor_prob < 0.20:
            assessment['warning_flag'] = 'CRITICAL_CONFLICT'
            assessment['warning'] = f"❌ CRITICAL: AI {ai_pred} vs Doctor {assessment['doctor_prediction']}"
            assessment['agreement_level'] = "🔴 CRITICAL"
            return assessment

    if similarity > 0.85:
        assessment['agreement_level'] = "🟢 EXCELLENT"
    elif similarity > 0.70:
        assessment['agreement_level'] = "🟡 GOOD"
    elif similarity > 0.50:
        assessment['agreement_level'] = "🟠 MODERATE"
        assessment['warning_flag'] = 'MODERATE_CONFLICT'
    else:
        assessment['agreement_level'] = "🔴 LOW"
        assessment['warning_flag'] = 'MODERATE_CONFLICT'

    return assessment


def generate_pdf(img, all_probs, top_idxs, top_probs, doctor_report, assessment, img_tensor, patient_name):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1f77b4'),
                                 alignment=TA_CENTER, fontName='Helvetica-Bold')
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12,
                                   textColor=colors.HexColor('#2ca02c'), fontName='Helvetica-Bold')
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10)

    story.append(Paragraph("Brain Tumor CDSS Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    info_data = [['Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")], ['Patient:', patient_name]]
    info_table = Table(info_data, colWidths=[1.5 * inch, 4.5 * inch])
    info_table.setStyle(
        TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.grey), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=2.5 * inch, height=2.5 * inch)

    cam = gradcam.generate(img_tensor, int(top_idxs[0]))
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img.resize(IMG_SIZE))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    overlay_pil = Image.fromarray(overlay)
    overlay_buffer = io.BytesIO()
    overlay_pil.save(overlay_buffer, format='PNG')
    overlay_buffer.seek(0)
    rl_overlay = RLImage(overlay_buffer, width=2.5 * inch, height=2.5 * inch)

    img_data = [[Paragraph("<b>Original</b>", normal_style), Paragraph("<b>Overlay</b>", normal_style)],
                [rl_img, rl_overlay]]
    img_table = Table(img_data, colWidths=[3 * inch, 3 * inch])
    img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(img_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("AI PREDICTION", heading_style))
    ai_data = [['Primary:', CLASSES[int(top_idxs[0])].upper()], ['Confidence:', f'{top_probs[0] * 100:.2f}%']]
    ai_table = Table(ai_data, colWidths=[2 * inch, 4 * inch])
    ai_table.setStyle(
        TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.lightblue), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(ai_table)
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("COMPARISON", heading_style))
    comp_data = [['Similarity:', f'{assessment["agreement"]:.2%}'], ['Warning:', assessment['warning_flag']]]
    comp_table = Table(comp_data, colWidths=[2 * inch, 4 * inch])
    comp_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(comp_table)

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


@app.route('/')
def home():
    return jsonify({"status": "API running", "version": "1.0.0"})


@app.route('/health')
def health():
    return jsonify({"status": "healthy", "device": str(DEVICE)})


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        image = request.files['image']
        doctor_report = request.form['doctor_report']
        patient_name = request.form.get('patient_name', 'Unknown')

        img, img_tensor, all_probs, top_idxs, top_probs = predict_image(image)

        similarity = cosine_similarity(
            [embedder.encode(f"Patient has {CLASSES[int(top_idxs[0])]} tumor")],
            [embedder.encode(doctor_report)]
        )[0][0]

        assessment = generate_assessment(
            CLASSES[int(top_idxs[0])],
            float(top_probs[0]),
            all_probs,
            top_idxs,
            doctor_report,
            float(similarity)
        )

        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "ai_prediction": CLASSES[int(top_idxs[0])].upper(),
            "confidence": round(float(top_probs[0]), 4),
            "confidence_percent": round(float(top_probs[0]) * 100, 2),
            "secondary_prediction": CLASSES[int(top_idxs[1])].upper(),
            "secondary_confidence": round(float(top_probs[1]) * 100, 2),
            "similarity_score": round(float(similarity), 4),
            "agreement_level": assessment['agreement_level'],
            "warning_flag": assessment['warning_flag'],
            "warning_message": assessment['warning'],
            "recommendation": assessment['recommendation']
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/analyze-with-pdf', methods=['POST', 'OPTIONS'])
def analyze_pdf():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        image = request.files['image']
        doctor_report = request.form['doctor_report']
        patient_name = request.form.get('patient_name', 'Unknown')

        img, img_tensor, all_probs, top_idxs, top_probs = predict_image(image)

        similarity = cosine_similarity(
            [embedder.encode(f"Patient has {CLASSES[int(top_idxs[0])]} tumor")],
            [embedder.encode(doctor_report)]
        )[0][0]

        assessment = generate_assessment(
            CLASSES[int(top_idxs[0])],
            float(top_probs[0]),
            all_probs,
            top_idxs,
            doctor_report,
            float(similarity)
        )

        pdf_buffer = generate_pdf(img, all_probs, top_idxs, top_probs, doctor_report, assessment, img_tensor,
                                  patient_name)

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'CDSS_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)