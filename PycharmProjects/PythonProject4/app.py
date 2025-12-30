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

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Brain Tumor CDSS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SETUP
# ============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (299, 299)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pth")

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DICT = {label: idx for idx, label in enumerate(CLASSES)}

# Image transformation
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================
# DEFINE MODEL
# ============================================

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


# ============================================
# GRAD-CAM CLASS
# ============================================

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


# ============================================
# LOAD MODELS (CACHED)
# ============================================

@st.cache_resource
def load_tumor_model():
    try:
        model = BrainTumorModel(num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at {MODEL_PATH}. Please update the path.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


@st.cache_resource
def load_gradcam(_model):
    return GradCAM(_model, _model.base_model.Mixed_7c)


@st.cache_resource
def load_clinical_embedder():
    """Load ClinicalBERT (best for clinical text)"""
    try:
        model = SentenceTransformer('allenai/aspire-sentence-embedder')
        return model
    except:
        try:
            model = SentenceTransformer('pritamdeka/PubMedBERT-base-embeddings')
            return model
        except:
            return SentenceTransformer('all-MiniLM-L6-v2')


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_image(img_path, model):
    """Predict tumor type from MRI image"""

    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, 2)

    top_probs = top_probs.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()

    all_probs = probs.cpu().numpy()

    return img, img_tensor, all_probs, top_idxs, top_probs


# ============================================
# REPORT COMPARISON
# ============================================

def compare_reports(ai_report, doctor_report, embedder):
    """Compare AI report with doctor report"""

    embeddings = embedder.encode([ai_report, doctor_report])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return similarity


def generate_smart_assessment(ai_pred, ai_conf, all_probs, top_idxs, doctor_report, similarity, doctor_pred=None):
    """Generate smart clinical assessment based on comparison"""

    assessment = {
        'agreement': similarity,
        'ai_prediction': ai_pred,
        'doctor_prediction': doctor_pred,
        'warning': None,
        'recommendation': None,
        'confidence_warning': False,
        'prediction_conflict': False,
        'warning_flag': 'NONE'
    }

    # Extract doctor prediction from their report
    doctor_report_lower = doctor_report.lower()
    doctor_found_pred = None
    for cls in CLASSES:
        if cls in doctor_report_lower:
            assessment['doctor_prediction'] = cls.upper()
            doctor_found_pred = cls.upper()
            break

    # ============================================
    # CHECK 1: AI Confidence Issues
    # ============================================

    if ai_conf < 0.60:
        assessment['confidence_warning'] = True
        assessment['warning_flag'] = 'LOW_CONFIDENCE'
        assessment['warning'] = "⚠️ LOW AI CONFIDENCE: AI prediction confidence is below 60%. "

    # Check if top 2 predictions are very close (ambiguous)
    if len(all_probs) >= 2:
        top1_prob = all_probs[int(top_idxs[0])]
        top2_prob = all_probs[int(top_idxs[1])]
        confidence_gap = top1_prob - top2_prob

        if confidence_gap < 0.10:
            assessment['confidence_warning'] = True
            assessment['warning_flag'] = 'AMBIGUOUS'
            assessment['warning'] = "⚠️ AMBIGUOUS PREDICTION: Top 2 predictions are very close. AI is uncertain. "
            assessment[
                'recommendation'] = "Strong recommendation for clinical review. AI cannot confidently distinguish between options."

    # ============================================
    # CHECK 2: AI vs Doctor Disagreement
    # ============================================

    if doctor_found_pred and doctor_found_pred != ai_pred:
        # Check if doctor's prediction is secondary prediction
        doctor_prob = all_probs[CLASSES.index(doctor_found_pred.lower())]

        if doctor_prob < 0.20:  # Doctor predicted something AI thinks is very unlikely
            assessment['prediction_conflict'] = True
            assessment['warning_flag'] = 'CRITICAL_CONFLICT'
            assessment['warning'] = (
                f"❌ CRITICAL CONFLICT: AI predicts {ai_pred} ({ai_conf * 100:.2f}%) "
                f"but doctor suggests {doctor_found_pred} ({doctor_prob * 100:.2f}%). "
                f"Large confidence gap. This requires immediate specialist review."
            )
            assessment[
                'recommendation'] = "DO NOT proceed based on either assessment alone. Immediate consultation with senior radiologist required."
            assessment['agreement_level'] = "🔴 CRITICAL CONFLICT"
            return assessment

    # ============================================
    # CHECK 3: Semantic Similarity (if both broadly agree)
    # ============================================

    if similarity > 0.85 and not assessment['confidence_warning']:
        assessment['agreement_level'] = "🟢 EXCELLENT"
        assessment['warning_flag'] = 'NONE'
        assessment['recommendation'] = "Strong agreement between AI and clinical assessment. Findings are consistent."
    elif similarity > 0.70:
        assessment['agreement_level'] = "🟡 GOOD"
        assessment['warning_flag'] = 'NONE'
        assessment['recommendation'] = "Good agreement with minor differences. Review variations carefully."
    elif similarity > 0.50:
        assessment['agreement_level'] = "🟠 MODERATE"
        assessment['warning_flag'] = 'MODERATE_CONFLICT'
        if not assessment['warning']:
            assessment[
                'warning'] = "⚠️ MODERATE DISCREPANCY: AI and clinical assessments show moderate differences. Recommend additional review."
        assessment[
            'recommendation'] = "Investigate reasons for the difference. Consider additional imaging or consultation."
    else:
        assessment['agreement_level'] = "🔴 LOW"
        assessment['warning_flag'] = 'MODERATE_CONFLICT'
        if not assessment['warning']:
            assessment[
                'warning'] = "❌ SIGNIFICANT DISCREPANCY: Major disagreement between AI prediction and clinical assessment."
        assessment['recommendation'] = "Consult with radiologist and clinical team immediately."

    return assessment


def create_json_report(img_filename, all_probs, top_idxs, top_probs, assessment, doctor_report):
    """Create JSON report with all AI findings and warning flags"""

    # Convert numpy types to Python native types
    all_probs_converted = [float(p) for p in all_probs]
    top_probs_converted = [float(p) for p in top_probs]

    report_json = {
        "report_metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_filename": img_filename,
            "system": "Brain Tumor CDSS v1.0",
            "model": "InceptionV3 + ClinicalBERT"
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
            "doctor_report_excerpt": doctor_report[:300] + "..." if len(doctor_report) > 300 else doctor_report
        },
        "warning_flags": {
            "warning_flag": assessment['warning_flag'],
            "has_warning": assessment['warning'] is not None,
            "warning_message": assessment['warning'],
            "has_confidence_warning": bool(assessment.get('confidence_warning', False)),
            "has_prediction_conflict": bool(assessment.get('prediction_conflict', False))
        },
        "recommendations": {
            "recommendation": assessment['recommendation']
        }
    }

    return report_json


# ============================================
# PDF GENERATION WITH COMPARISON
# ============================================

def generate_pdf_report(img, all_probs, top_idxs, top_probs, filename, doctor_report, assessment, img_tensor, model,
                        gradcam):
    """Generate comprehensive PDF report with AI-Doctor comparison"""

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_LEFT
    )

    # Title
    story.append(Paragraph("🧠 Brain Tumor MRI - Clinical Decision Support Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Report Info
    info_data = [
        ['Report Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Image File:', filename],
    ]
    info_table = Table(info_data, colWidths=[1.8 * inch, 4.2 * inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    # ============================================
    # IMAGE SECTION - Original + Grad-CAM Overlay Side by Side
    # ============================================

    story.append(Paragraph("MRI IMAGES & ANALYSIS", heading_style))

    # Convert original image to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=2.8 * inch, height=2.8 * inch)

    # Generate Grad-CAM overlay
    cam = gradcam.generate(img_tensor, int(top_idxs[0]))
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Create overlay (MRI + Grad-CAM blend)
    img_np = np.array(img.resize(IMG_SIZE))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Convert overlay to bytes
    overlay_pil = Image.fromarray(overlay)
    overlay_buffer = io.BytesIO()
    overlay_pil.save(overlay_buffer, format='PNG')
    overlay_buffer.seek(0)
    rl_overlay = RLImage(overlay_buffer, width=2.8 * inch, height=2.8 * inch)

    # Create side-by-side table
    img_data = [
        [Paragraph("<b>Original MRI</b>", normal_style), Paragraph("<b>Grad-CAM Overlay</b>", normal_style)],
        [rl_img, rl_overlay]
    ]
    img_table = Table(img_data, colWidths=[3.2 * inch, 3.2 * inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F4F8')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.2 * inch))

    # ============================================
    # PREDICTIONS
    # ============================================

    story.append(Paragraph("AI MODEL PREDICTION", heading_style))
    ai_data = [
        ['Primary:', CLASSES[int(top_idxs[0])].upper()],
        ['Confidence:', f'{top_probs[0] * 100:.2f}%'],
        ['Secondary:', CLASSES[int(top_idxs[1])].upper() + f' ({top_probs[1] * 100:.2f}%)'],
    ]
    ai_table = Table(ai_data, colWidths=[1.8 * inch, 4.2 * inch])
    ai_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ADD8E6')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(ai_table)
    story.append(Spacer(1, 0.15 * inch))

    # ============================================
    # DOCTOR ASSESSMENT
    # ============================================

    story.append(Paragraph("CLINICAL ASSESSMENT (Doctor)", heading_style))
    story.append(Paragraph(doctor_report, normal_style))
    story.append(Spacer(1, 0.15 * inch))

    # ============================================
    # COMPARISON
    # ============================================

    story.append(Paragraph("AI-CLINICAL COMPARISON", heading_style))
    comp_data = [
        ['Similarity Score:', f'{assessment["agreement"]:.2%}'],
        ['Agreement Level:', assessment['agreement_level']],
        ['Doctor Predicted:', assessment['doctor_prediction'] or 'Not found'],
    ]
    comp_table = Table(comp_data, colWidths=[1.8 * inch, 4.2 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#FFE4B5')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 0.15 * inch))

    # ============================================
    # WARNINGS & RECOMMENDATIONS
    # ============================================

    if assessment['warning']:
        story.append(Paragraph("⚠️ WARNINGS", heading_style))
        story.append(Paragraph(assessment['warning'], normal_style))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("RECOMMENDATION", heading_style))
    story.append(Paragraph(assessment['recommendation'], normal_style))
    story.append(Spacer(1, 0.2 * inch))

    # ============================================
    # DISCLAIMER
    # ============================================

    story.append(Paragraph("IMPORTANT DISCLAIMER", heading_style))
    disclaimer = (
        "This is a Clinical Decision Support System (CDSS) for educational purposes. "
        "It is NOT a diagnostic tool and should NOT replace professional medical judgment. "
        "All findings must be verified by qualified radiologists and clinicians."
    )
    story.append(Paragraph(disclaimer, normal_style))

    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


# ============================================
# MAIN APP
# ============================================

st.title("🧠 Brain Tumor MRI - Clinical Decision Support System (CDSS)")
st.markdown("**AI-Assisted Analysis with Doctor Comparison & Clinical Validation**")
st.markdown("---")

# Sidebar
st.sidebar.title("ℹ️ About CDSS")
st.sidebar.info(
    """
    **Clinical Decision Support System**

    This CDSS uses:
    - InceptionV3 (Tumor Detection)
    - ClinicalBERT (Medical Text)
    - Grad-CAM (Explainability)

    **Tumor Types:**
    - 🔴 Glioma
    - 🟡 Meningioma
    - 🟢 No Tumor
    - 🟣 Pituitary

    ⚠️ **FOR CLINICAL SUPPORT ONLY**
    """
)

st.sidebar.write(f"**Device:** {DEVICE}")

# Load models
tumor_model = load_tumor_model()
gradcam = load_gradcam(tumor_model)

st.sidebar.info("Loading clinical embedder...")
embedder = load_clinical_embedder()
st.sidebar.success("✅ ClinicalBERT loaded!")

# Upload image
st.subheader("📤 Step 1: Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict
    with st.spinner("🔄 Analyzing image..."):
        img, img_tensor, all_probs, top_idxs, top_probs = predict_image("temp_image.jpg", tumor_model)

    # Display AI results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📸 Original Image")
        st.image(img, width=350)

    with col2:
        st.subheader("✅ AI Prediction")
        st.success(f"**PRIMARY**")
        st.metric(CLASSES[int(top_idxs[0])].upper(), f"{top_probs[0] * 100:.2f}%")

        st.warning(f"**SECONDARY**")
        st.metric(CLASSES[int(top_idxs[1])].upper(), f"{top_probs[1] * 100:.2f}%")

    with col3:
        st.subheader("🔥 Grad-CAM")
        cam = gradcam.generate(img_tensor, int(top_idxs[0]))
        cam = cv2.resize(cam, IMG_SIZE)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        st.image(heatmap, width=350, channels="BGR")

    st.markdown("---")

    # Grad-CAM Overlay
    st.subheader("🎯 Grad-CAM Overlay")
    cam = gradcam.generate(img_tensor, int(top_idxs[0]))
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img.resize(IMG_SIZE))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    st.image(overlay, use_column_width=True)

    st.markdown("---")

    # Probability chart
    st.subheader("📈 All Predictions")
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_chart = ['#2ecc71' if i == int(top_idxs[0]) else '#3498db' for i in range(len(CLASSES))]
    bars = ax.barh(CLASSES, all_probs, color=colors_chart)
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.bar_label(bars, fmt='%.3f', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Doctor Input Section
    st.subheader("👨‍⚕️ Step 2: Doctor Clinical Assessment")
    st.info("Enter the doctor's clinical findings and diagnosis")

    doctor_report = st.text_area(
        "Doctor's Report:",
        placeholder="Paste the doctor's clinical assessment, findings, and diagnosis here...",
        height=150
    )

    if doctor_report:
        st.markdown("---")

        # Compare reports
        with st.spinner("🔄 Comparing AI prediction with clinical assessment..."):
            similarity = compare_reports(
                f"Patient has {CLASSES[int(top_idxs[0])]} tumor with {top_probs[0] * 100:.2f}% confidence",
                doctor_report,
                embedder
            )

        assessment = generate_smart_assessment(
            CLASSES[int(top_idxs[0])],
            top_probs[0],
            all_probs,
            top_idxs,
            doctor_report,
            similarity
        )

        # Display Comparison Results
        st.subheader("📊 AI-Clinical Comparison Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Similarity Score", f"{assessment['agreement']:.2%}")
        with col2:
            st.metric("Agreement Level", assessment['agreement_level'])
        with col3:
            st.metric("Doctor Predicted", assessment['doctor_prediction'] or "N/A")

        # Show warnings if any
        if assessment['warning']:
            st.warning(assessment['warning'], icon="⚠️")

        # Show recommendation
        st.info(f"**Recommendation:** {assessment['recommendation']}", icon="💡")

        st.markdown("---")

        # Generate JSON Report
        st.subheader("📋 AI Report (JSON)")

        json_report = create_json_report(
            Path(uploaded_file.name).name,
            all_probs,
            top_idxs,
            top_probs,
            assessment,
            doctor_report
        )

        # Display JSON in expander
        with st.expander("📄 Click to view full JSON report"):
            st.json(json_report)

        # Download JSON button
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(json_report, indent=2),
                file_name=f"AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("---")

        # Generate PDF Report
        st.subheader("📄 Generate Comprehensive Report")

        if st.button("🔄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_report = generate_pdf_report(
                    img, all_probs, top_idxs, top_probs,
                    Path(uploaded_file.name).name,
                    doctor_report,
                    assessment,
                    img_tensor,
                    tumor_model,
                    gradcam
                )

                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_report,
                    file_name=f"CDSS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ Report ready for download!")

        Path("temp_image.jpg").unlink()

else:
    st.info("👆 **Upload an MRI image to begin CDSS analysis**")