"""
app.py  -  CLAM MIL Colon Cancer Detection - Streamlit Demo
============================================================
Run:
    streamlit run app.py
"""

import os
import glob
import tempfile
import time
from datetime import datetime
import cv2

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from pipeline import run_full_inference, get_true_label

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title  = "ColonAI - Cancer Detection",
    page_icon   = ":microscope:",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ============================================================================
# GLOBAL STYLE
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 18px 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    color: #a0aec0 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e2e8f0 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #0f3460;
    border-radius: 12px;
    padding: 10px;
}

.result-card {
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    margin: 8px 0;
}
.tumor-card {
    background: linear-gradient(135deg, #4a0000 0%, #7b0000 100%);
    border: 1px solid #ff4444;
    box-shadow: 0 0 30px rgba(255,68,68,0.25);
}
.normal-card {
    background: linear-gradient(135deg, #003300 0%, #005500 100%);
    border: 1px solid #44ff44;
    box-shadow: 0 0 30px rgba(68,255,68,0.20);
}
.result-card h1 { margin: 0 0 6px 0; font-size: 2.4rem; }
.result-card p  { margin: 0; font-size: 0.95rem; color: #ccc; }

.section-heading {
    font-size: 1.15rem;
    font-weight: 600;
    color: #90cdf4;
    margin: 8px 0 4px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #2d3748;
}
.info-box {
    background: #1a202c;
    border-left: 4px solid #4299e1;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.9rem;
    color: #cbd5e0;
    line-height: 1.6;
}
.warn-box {
    background: #2d2000;
    border-left: 4px solid #f6ad55;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.9rem;
    color: #fbd38d;
    line-height: 1.6;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    color: #e2e8f0;
    border: 1px solid #4299e1;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1a4a80 0%, #1a2a50 100%);
    border-color: #63b3ed;
    box-shadow: 0 0 16px rgba(66,153,225,0.35);
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
MODEL_DIR   = os.path.join("models", "five_fold")
MODEL_PATHS = sorted(glob.glob(os.path.join(MODEL_DIR, "best_fold_*.pt")))
THRESHOLD   = 0.276
DISPLAY_SIZE = 768

# ============================================================================
# HELPERS
# ============================================================================

def render_heatmap(grid: np.ndarray, title: str, figsize=(6, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(grid, cmap="jet", interpolation="bilinear",
                   origin="lower", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Attention", color="#a0aec0", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#a0aec0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#a0aec0", fontsize=8)
    cbar.outline.set_edgecolor("#2d3748")
    ax.set_title(title, color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    plt.tight_layout(pad=0.5)
    return fig


def resize_for_display(image: np.ndarray, target_size: int = DISPLAY_SIZE) -> np.ndarray:
    """
    Resize image to a uniform square (target_size x target_size) while preserving aspect ratio.
    Applies padding to prevent layout shifts across UI tabs/columns.
    """
    h, w = image.shape[:2]
    scale = target_size / float(max(h, w))
    
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    # Pad with black background for consistency
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def confidence_bar(prob: float, threshold: float = 0.276):
    pct      = int(prob * 100)
    color    = "#e53e3e" if prob >= threshold else "#38a169"
    bg_color = "#1a202c"
    st.markdown(f"""
    <div style="margin: 6px 0 14px 0;">
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#a0aec0; font-size:0.8rem;">Normal (0%)</span>
        <span style="color:#a0aec0; font-size:0.8rem; font-weight:600;">
            Tumour probability: {pct}%
        </span>
        <span style="color:#a0aec0; font-size:0.8rem;">Tumour (100%)</span>
      </div>
      <div style="background:{bg_color}; border-radius:8px; height:14px; position:relative; border:1px solid #2d3748;">
        <div style="width:{pct}%; background:{color}; border-radius:8px;
                    height:100%; transition:width 0.4s ease;"></div>
        <div style="position:absolute; left:{int(threshold*100)}%;
                    top:0; bottom:0; width:2px; background:#f6ad55;"></div>
      </div>
      <div style="text-align:right; margin-top:3px;">
        <span style="color:#f6ad55; font-size:0.75rem;">
            &#9651; Decision threshold ({int(threshold*100)}%)
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


def get_interpretation(grid_8x8: np.ndarray) -> str:
    idx = np.unravel_index(np.argmax(grid_8x8), grid_8x8.shape)
    y, x = idx
    v_pos = "upper" if y > 3 else "lower"
    h_pos = "left"  if x < 4 else "right"
    return (f"High attention observed in the {v_pos}-{h_pos} region, "
            "indicating possible tumour features in this area.")


def generate_report(result: dict, filename: str) -> bytes:
    """Generate a medical-style PDF report and return as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'],
        fontSize=22, alignment=1, spaceAfter=20
    )
    header_style = ParagraphStyle(
        'HeaderStyle', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor("#0f3460"), spaceBefore=15
    )

    elements = []

    # Title
    elements.append(Paragraph("ColonAI - Clinical Detection Report", title_style))
    elements.append(Paragraph(f"Slide: {os.path.basename(filename)}", styles["Normal"]))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # Diagnostic Summary
    elements.append(Paragraph("Diagnostic Summary", header_style))
    status_color = "#ff4444" if result["prediction"] == 1 else "#44bb44"
    elements.append(Paragraph(
        f"<b>Result:</b> <font color='{status_color}'>{result['label'].upper()}</font>",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"<b>Tumour Probability:</b> {result['probability']:.4f} ({result['probability']*100:.1f}%)",
        styles["Normal"]
    ))
    elements.append(Paragraph(f"<b>Confidence Score:</b> {result['confidence']:.4f}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Decision Threshold:</b> {THRESHOLD}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Patch Count:</b> {result['n_patches']:,}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # AI Interpretation
    elements.append(Paragraph("AI Interpretation", header_style))
    if result.get("heatmap_grid") is not None:
        interp = get_interpretation(result["heatmap_grid"])
    else:
        interp = "No spatial coordinates available for attention mapping."
    elements.append(Paragraph(interp, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Attention Heatmaps
    if result.get("heatmap_grid") is not None:
        elements.append(Paragraph("Attention Visualization", header_style))
        elements.append(Paragraph(
            "Heatmaps show focal regions where the model concentrated its prediction effort. "
            "Note: The detailed spatial heatmap should be viewed in the application UI.",
            styles["Italic"]
        ))

        def grid_to_img(grid, title):
            fig = render_heatmap(grid, title, figsize=(5, 4))
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, facecolor='#ffffff')
            plt.close(fig)
            buf.seek(0)
            return Image(buf, width=220, height=176)

        img_grid  = grid_to_img(result["heatmap_grid"],  "Coarse Attention Map")
        # For simplicity in PDF, maybe only include the grid view, or include spatial if PIL is used
        t = Table([[img_grid]], colWidths=[240])
        elements.append(t)
        elements.append(Spacer(1, 12))

    # Methodology
    elements.append(Paragraph("Methodology", header_style))
    elements.append(Paragraph("Model: CLAM-SB (Attention-based MIL)", styles["Normal"]))
    elements.append(Paragraph("Inference: 5-Fold Ensemble Averaging", styles["Normal"]))
    elements.append(Paragraph(f"Decision Threshold: {THRESHOLD} (tuned to maximise F1)", styles["Normal"]))
    elements.append(Paragraph("Training: 46 slides, TCGA-COAD/READ, 5-fold stratified CV", styles["Normal"]))
    elements.append(Spacer(1, 40))

    elements.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an experimental AI system for "
        "research and demonstration purposes only. It is NOT a substitute for expert "
        "pathological diagnosis by a certified medical professional.",
        styles["Italic"]
    ))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## ColonAI")
    st.markdown("**CLAM - Attention MIL**")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool analyses pre-extracted features from
    **colon histopathology whole-slide images (WSIs)**
    and classifies them as **Tumour** or **Normal**.

    Built with:
    - CLAM (attention-based MIL)
    - TCGA-COAD dataset
    - 5-fold cross-validation
    """)
    st.markdown("---")
    st.markdown("### Model Status")
    if MODEL_PATHS:
        st.success(f"OK - {len(MODEL_PATHS)} fold models loaded")
    else:
        st.error("No model checkpoints found")
    st.markdown("---")
    st.markdown("### Decision Threshold")
    st.info(f"Threshold = **{THRESHOLD}**\n\n(tuned to maximise F1)")
    st.markdown("---")
    st.caption("Buddham Rajbhandari - Kaviya Darshini - Dakshini")


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div style="text-align:center; padding: 10px 0 6px 0;">
    <h1 style="font-size:2.4rem; font-weight:700; margin-bottom:4px;">
        ColonAI - Cancer Detection
    </h1>
    <p style="color:#90cdf4; font-size:1.1rem; margin:0; font-weight:500;">
        AI system for detecting colon cancer and highlighting suspicious regions
    </p>
    <p style="color:#a0aec0; font-size:0.9rem; margin-top:4px;">
        Attention-based Multiple Instance Learning on Histopathology WSIs
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ============================================================================
# FILE UPLOAD
# ============================================================================
st.markdown('<p class="section-heading">Analysis Input</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
Upload a <code>.pt</code> file containing pre-extracted features for one WSI.<br>
Expected format: <code>{'features': [N, 2048], 'coords': [N, 2]}</code>
</div>
""", unsafe_allow_html=True)
st.markdown(" ")

uploaded = st.file_uploader(
    "Drop a .pt feature bag here",
    type=["pt"],
    label_visibility="collapsed",
)


# ============================================================================
# INFERENCE
# ============================================================================
if uploaded is not None:

    if not MODEL_PATHS:
        st.error("No trained model checkpoints found. Train the model first.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Running ensemble inference across all 5 fold models..."):
        try:
            start_time = time.time()
            result = run_full_inference(tmp_path, MODEL_PATHS)
            end_time = time.time()
            inference_time = end_time - start_time
        except Exception as e:
            st.error(f"Inference failed: {e}")
            os.remove(tmp_path)
            st.stop()

    os.remove(tmp_path)

    prob       = result["probability"]
    pred       = result["prediction"]
    label      = result["label"]
    confidence = result["confidence"]
    n_patches  = result["n_patches"]
    has_coords = result["has_coords"]

    true_label = get_true_label(uploaded.name)

    st.markdown("---")

    # =========================================================================
    # PREDICTION RESULT
    # =========================================================================
    st.markdown('<p class="section-heading">Slide-Level Prediction</p>', unsafe_allow_html=True)

    col_res, col_detail = st.columns([1, 1.6], gap="large")

    with col_res:
        if pred == 1:
            st.markdown("""
            <div class="result-card tumor-card">
                <h1>TUMOUR</h1>
                <p>Malignant tissue detected</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card normal-card">
                <h1>NORMAL</h1>
                <p>No malignancy detected</p>
            </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown(f"**Uploaded Slide:** `{uploaded.name}`")
        st.markdown("**Tumour Probability**")
        confidence_bar(prob, THRESHOLD)

        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Probability %", f"{prob*100:.1f}%")
        with d2: st.metric("Confidence",    f"{confidence:.3f}")
        with d3: st.metric("Patches",       f"{n_patches:,}")
        with d4: st.metric("Processing Time", f"{inference_time:.2f} seconds")

        if true_label is not None:
            correct = (pred == true_label)
            icon = "✅" if correct else "❌"
            label_name = "TUMOUR" if true_label == 1 else "NORMAL"
            st.markdown(f"**Ground Truth:** `{label_name}` ({icon})")

    st.markdown("---")

    # =========================================================================
    # ATTENTION HEATMAPS
    # =========================================================================
    if has_coords and result.get("heatmap_grid") is not None:
        st.markdown('<p class="section-heading">Spatial Attention Heatmaps</p>',
                    unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Coarse Attention Map (Grid View)", "Spatial Attention Map"])
        
        with tab1:
            st.markdown("""
            <div class="info-box" style="margin-bottom: 12px;">
            <strong>Coarse grid view</strong> provides a quick attention summary. 
            Hotspots indicate patch regions the model weighted most heavily.
            </div>
            """, unsafe_allow_html=True)
            fig_grid = render_heatmap(result["heatmap_grid"], "Coarse Attention Map (Grid View)", figsize=(6, 5))
            buf = io.BytesIO()
            fig_grid.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig_grid)
            
            buf.seek(0)
            from PIL import Image
            img_grid = np.array(Image.open(buf).convert("RGB"))
            
            img_grid_display = resize_for_display(img_grid, target_size=DISPLAY_SIZE)
            st.image(img_grid_display, use_column_width=True)
            
            interp = get_interpretation(result["heatmap_grid"])
            st.markdown(f"""
            <div class="info-box" style="margin-top:12px;">
            <strong>AI Interpretation:</strong> {interp}
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.info("Heatmap generated from patch-level attention scores without original tissue image")
            
            # User Controls
            ctrl1, ctrl2 = st.columns(2)
            with ctrl1:
                threshold_val = st.slider("Attention Threshold", 0.0, 1.0, 0.6, step=0.05, 
                                          help="Suppress low attention regions below this threshold.")
            with ctrl2:
                alpha_val = st.slider("Heatmap Intensity", 0.0, 1.0, 0.6, step=0.05,
                                      help="Adjust the alpha blending intensity of the heatmap colors.")
                                      
            st.caption("🔴 **Red** = High attention (potential cancer foci) | 🔵 **Blue** = Low attention (normal tissue)")
            
            from heatmap_utils import create_spatial_heatmap
            with st.spinner("Rendering spatial heatmap..."):
                spatial_heatmap = create_spatial_heatmap(
                    result["coords"], result["attention"], 
                    patch_size=256, threshold=threshold_val, alpha=alpha_val
                )
                
            spatial_display = resize_for_display(spatial_heatmap, target_size=DISPLAY_SIZE)
            st.image(spatial_display, caption="Detailed Spatial Overlay", use_column_width=True)

        st.markdown("---")
    else:
        st.info("No patch coordinates found in this feature file - heatmaps unavailable.")
        st.markdown("---")

    # =========================================================================
    # REPORT DOWNLOAD
    # =========================================================================
    st.markdown('<p class="section-heading">Diagnostic Report</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Download a structured PDF report containing the diagnostic summary,
    AI interpretation, attention heatmaps (if available), and methodology details.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(" ")

    with st.spinner("Generating PDF report..."):
        try:
            pdf_bytes = generate_report(result, uploaded.name)
            report_name = f"ColonAI_Report_{os.path.splitext(uploaded.name)[0]}.pdf"
            st.download_button(
                label     = "Download Diagnostic Report (PDF)",
                data      = pdf_bytes,
                file_name = report_name,
                mime      = "application/pdf",
                use_container_width = True,
            )
            st.success("Report ready - click the button above to download.")
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    st.markdown("---")


# ============================================================================
# MODEL LIMITATIONS
# ============================================================================
st.markdown('<p class="section-heading">Model Limitations and Known Failure Cases</p>',
            unsafe_allow_html=True)
st.markdown("""
<div class="warn-box">
<strong>What can go wrong?</strong><br><br>
<strong>False Negatives (missed tumours)</strong> - The model predicts <em>Normal</em>
for a slide that is actually tumour. In our 5-fold evaluation,
<strong>4 out of 23 tumour slides</strong> were missed (sensitivity = 81.8%).<br><br>
<strong>Under-confident predictions</strong> - Because training used only 46 balanced
slides, probabilities are systematically lower than 0.5 even for true tumours.
We compensate with a tuned threshold of <strong>0.276</strong>.<br><br>
<strong>Attention vs Pathologist annotation</strong> - High-attention regions are
statistically correlated with the slide label, but are <em>not</em> guaranteed to
correspond to tumour cells. Always confirm with expert review.
</div>
""", unsafe_allow_html=True)

st.markdown(" ")
with st.expander("Known False Negative Example from Cross-Validation"):
    col_fn1, col_fn2 = st.columns([1, 2])
    with col_fn1:
        st.markdown("""
        | Field | Value |
        |---|---|
        | **Slide** | TCGA-3L-AA1B-01Z |
        | **True label** | Tumour |
        | **Predicted** | Normal |
        | **Probability** | 0.196 |
        | **Threshold** | 0.276 |
        | **Patches** | 26,812 |
        """)
    with col_fn2:
        st.markdown("""
        <div class="warn-box">
        This tumour slide scored only <strong>0.196</strong> - well below
        the 0.276 threshold - and was therefore classified as Normal.
        The attention heatmap shows diffuse low-level activation with no
        strong focal hotspot. This is a <em>challenging case</em> even for expert pathologists.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# HOW IT WORKS
# ============================================================================
with st.expander("How the Pipeline Works"):
    st.markdown("""
    ```
    WSI (.svs)
       -> Tissue segmentation + patch extraction (256x256 px at 20x)
       -> ResNet-50 feature extraction  ->  [N, 2048] per patch
       -> Saved as .pt file  { features, coords }
          ---- your upload starts here ----
       -> AttentionNet: Linear(2048->512) -> Tanh -> Dropout -> Linear(512->1)
       -> Softmax over ALL N patches  ->  attention distribution [N]
       -> Weighted sum  ->  slide representation [2048]
       -> Classifier: Linear(2048->1) + Sigmoid  ->  probability
       -> 5x fold ensemble (average probabilities + attention)
       -> Threshold @ 0.276  ->  Tumour / Normal
       -> Grid reconstruction (8x8, 16x16)  ->  heatmaps
    ```
    **Training:** 5-fold stratified cross-validation - 46 slides (23 tumour + 23 normal)
    - Adam (LR=5e-5) - BCE loss - Early stopping on validation loss
    """)

st.markdown("---")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 8px 0 20px 0;">
    ColonAI - CLAM-based MIL for Colon Cancer Detection -
    Buddham Rajbhandari &nbsp;-&nbsp; Kaviya Darshini &nbsp;-&nbsp; Dakshini
    <br>
    <span style="font-size:0.72rem;">
    For research and demonstration purposes only.
    Not a substitute for expert pathological diagnosis.
    </span>
</div>
""", unsafe_allow_html=True)