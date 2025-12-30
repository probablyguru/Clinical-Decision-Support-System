import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
import json

st.set_page_config(
    page_title="Brain Tumor CDSS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
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


@st.cache_resource
def load_tumor_model():
    try:
        model = BrainTumorModel(num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_resource
def load_gradcam(_model):
    return GradCAM(_model, _model.base_model.Mixed_7c)


@st.cache_resource
def load_clinical_embedder():
    try:
        return SentenceTransformer('allenai/aspire-sentence-embedder')
    except:
        try:
            return SentenceTransformer('pritamdeka/PubMedBERT-base-embeddings')
        except:
            return SentenceTransformer('all-MiniLM-L6-v2')


def predict_image(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, 2)
    return img, img_tensor, probs.cpu().numpy(), top_idxs.cpu().numpy(), top_probs.cpu().numpy()


def compare_reports(ai_report, doctor_report, embedder):
    embeddings = embedder.encode([ai_report, doctor_report])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity


def generate_smart_assessment(ai_pred, ai_conf, all_probs, top_idxs, doctor_report, similarity):
    assessment = {
        'agreement': similarity,
        'ai_prediction': ai_pred,
        'doctor_prediction': None,
        'warning': None,
        'recommendation': None,
        'confidence_warning': False,
        'warning_flag': 'NONE'
    }

    doctor_report_lower = doctor_report.lower()
    for cls in CLASSES:
        if cls in doctor_report_lower:
            assessment['doctor_prediction'] = cls.upper()
            break

    if ai_conf < 0.60:
        assessment['confidence_warning'] = True
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
        assessment['warning_flag'] = 'NONE'
    elif similarity > 0.70:
        assessment['agreement_level'] = "🟡 GOOD"
    elif similarity > 0.50:
        assessment['agreement_level'] = "🟠 MODERATE"
        assessment['warning_flag'] = 'MODERATE_CONFLICT'
    else:
        assessment['agreement_level'] = "🔴 LOW"
        assessment['warning_flag'] = 'MODERATE_CONFLICT'

    return assessment


def create_json_report(img_filename, all_probs, top_idxs, top_probs, assessment, doctor_report):
    all_probs_converted = [float(p) for p in all_probs]
    top_probs_converted = [float(p) for p in top_probs]

    return {
        "report_metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_filename": img_filename,
            "system": "Brain Tumor CDSS v1.0"
        },
        "ai_prediction": {
            "primary_prediction": {
                "tumor_type": CLASSES[int(top_idxs[0])].upper(),
                "confidence": round(top_probs_converted[0], 4),
                "confidence_percent": round(top_probs_converted[0] * 100, 2)
            },
            "secondary_prediction": {
                "tumor_type": CLASSES[int(top_idxs[1])].upper(),
                "confidence": round(top_probs_converted[1], 4),
                "confidence_percent": round(top_probs_converted[1] * 100, 2)
            },
            "all_predictions": {
                cls.upper(): round(prob * 100, 2)
                for cls, prob in zip(CLASSES, all_probs_converted)
            }
        },
        "clinical_assessment": {
            "doctor_prediction": assessment['doctor_prediction'],
            "similarity_score": round(float(assessment['agreement']), 4),
            "agreement_level": assessment['agreement_level'],
        },
        "warning_flags": {
            "warning_flag": assessment['warning_flag'],
            "has_warning": assessment['warning'] is not None,
            "warning_message": assessment['warning'],
        },
        "recommendations": {
            "recommendation": assessment['recommendation']
        }
    }


def generate_pdf_report(img, all_probs, top_idxs, top_probs, filename, doctor_report, assessment, img_tensor, model,
                        gradcam):
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

    info_data = [['Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")], ['File:', filename]]
    info_table = Table(info_data, colWidths=[1.5 * inch, 4.5 * inch])
    info_table.setStyle(TableStyle(
        [('BACKGROUND', (0, 0), (0, -1), colors.grey), ('GRID', (0, 0), (-1, -1), 1, colors.black),
         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold')]))
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("MRI IMAGES & ANALYSIS", heading_style))

    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=2.8 * inch, height=2.8 * inch)

    cam = gradcam.generate(img_tensor, int(top_idxs[0]))
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img.resize(IMG_SIZE))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    overlay_pil = Image.fromarray(overlay)
    overlay_buffer = io.BytesIO()
    overlay_pil.save(overlay_buffer, format='PNG')
    overlay_buffer.seek(0)
    rl_overlay = RLImage(overlay_buffer, width=2.8 * inch, height=2.8 * inch)

    img_data = [[Paragraph("<b>Original</b>", normal_style), Paragraph("<b>Overlay</b>", normal_style)],
                [rl_img, rl_overlay]]
    img_table = Table(img_data, colWidths=[3.2 * inch, 3.2 * inch])
    img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(img_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("AI PREDICTION", heading_style))
    ai_data = [['Primary:', CLASSES[int(top_idxs[0])].upper()], ['Confidence:', f'{top_probs[0] * 100:.2f}%']]
    ai_table = Table(ai_data, colWidths=[2 * inch, 4 * inch])
    ai_table.setStyle(TableStyle(
        [('BACKGROUND', (0, 0), (0, -1), colors.lightblue), ('GRID', (0, 0), (-1, -1), 1, colors.black),
         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold')]))
    story.append(ai_table)
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("DOCTOR ASSESSMENT", heading_style))
    story.append(Paragraph(doctor_report[:500], normal_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("COMPARISON", heading_style))
    comp_data = [['Similarity:', f'{assessment["agreement"]:.2%}'], ['Agreement:', assessment['agreement_level']],
                 ['Warning:', assessment['warning_flag']]]
    comp_table = Table(comp_data, colWidths=[2 * inch, 4 * inch])
    comp_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(comp_table)

    if assessment['warning']:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("⚠️ WARNING", heading_style))
        story.append(Paragraph(assessment['warning'], normal_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


st.title("🧠 Brain Tumor MRI - CDSS")
st.markdown("**Clinical Decision Support System**")

st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**CDSS System**
- InceptionV3 Model
- ClinicalBERT Comparison
- Grad-CAM Visualization

**Tumor Types:**
- 🔴 Glioma
- 🟡 Meningioma
- 🟢 No Tumor
- 🟣 Pituitary
""")

st.sidebar.write(f"**Device:** {DEVICE}")

tumor_model = load_tumor_model()
gradcam = load_gradcam(tumor_model)
embedder = load_clinical_embedder()

st.subheader("📤 Upload MRI Image")
uploaded_file = st.file_uploader("Choose MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔄 Analyzing..."):
        img, img_tensor, all_probs, top_idxs, top_probs = predict_image("temp_image.jpg", tumor_model)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📸 Image")
        st.image(img, width=300)
    with col2:
        st.subheader("✅ Prediction")
        st.metric(CLASSES[int(top_idxs[0])].upper(), f"{top_probs[0] * 100:.2f}%")
    with col3:
        st.subheader("🔥 Grad-CAM")
        cam = gradcam.generate(img_tensor, int(top_idxs[0]))
        cam = cv2.resize(cam, IMG_SIZE)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        st.image(heatmap, width=300, channels="BGR")

    st.markdown("---")
    st.subheader("👨‍⚕️ Doctor's Report")
    doctor_report = st.text_area("Enter doctor assessment:", height=150)

    if doctor_report:
        with st.spinner("Comparing..."):
            similarity = compare_reports(f"Patient has {CLASSES[int(top_idxs[0])]} tumor", doctor_report, embedder)

        assessment = generate_smart_assessment(CLASSES[int(top_idxs[0])], top_probs[0], all_probs, top_idxs,
                                               doctor_report, similarity)

        st.subheader("📊 Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Similarity", f"{assessment['agreement']:.2%}")
        with col2:
            st.metric("Agreement", assessment['agreement_level'])
        with col3:
            st.metric("Warning", assessment['warning_flag'])

        if assessment['warning']:
            st.warning(assessment['warning'])

        st.info(f"**Recommendation:** {assessment['recommendation']}")

        st.markdown("---")
        st.subheader("📋 JSON Report")
        json_report = create_json_report(Path(uploaded_file.name).name, all_probs, top_idxs, top_probs, assessment,
                                         doctor_report)

        with st.expander("View JSON"):
            st.json(json_report)

        st.download_button("📥 Download JSON", json.dumps(json_report, indent=2),
                           f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json",
                           use_container_width=True)

        st.markdown("---")
        if st.button("🔄 Generate PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf = generate_pdf_report(img, all_probs, top_idxs, top_probs, Path(uploaded_file.name).name,
                                          doctor_report, assessment, img_tensor, tumor_model, gradcam)
                st.download_button("📥 Download PDF", pdf, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   "application/pdf", use_container_width=True)
                st.success("✅ PDF Ready!")

        Path("temp_image.jpg").unlink()