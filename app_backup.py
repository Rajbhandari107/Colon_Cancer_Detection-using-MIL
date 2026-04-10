"""
app.py  —  CLAM MIL Colon Cancer Detection · Streamlit Demo
============================================================
Run:
    streamlit run app.py
"""

import os
import glob
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import io

# ── reportlab for PDF ───────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── project imports ─────────────────────────────────────────────────────────
from pipeline import run_full_inference, get_true_label

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "ColonAI — Cancer Detection",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── metric cards ── */
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

/* ── upload box ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #0f3460;
    border-radius: 12px;
    padding: 10px;
}

/* ── result banner common ── */
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

/* ── section headings ── */
.section-heading {
    font-size: 1.15rem;
    font-weight: 600;
    color: #90cdf4;
    margin: 8px 0 4px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #2d3748;
}

/* ── info boxes ── */
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
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
MODEL_DIR   = os.path.join("models", "five_fold")
MODEL_PATHS = sorted(glob.glob(os.path.join(MODEL_DIR, "best_fold_*.pt")))
THRESHOLD   = 0.276          # tuned via F1 on training folds

# ════════════════════════════════════════════════════════════════════════════
# HELPER — heatmap rendering
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# HELPER — render heatmap to matplotlib figure
# ════════════════════════════════════════════════════════════════════════════
def render_heatmap(grid: np.ndarray, title: str, figsize=(6, 5)) -> plt.Figure:
    """Return a tight matplotlib figure for a 2-D attention grid."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(
        grid,
        cmap        = "jet",
        interpolation = "bilinear",
        origin      = "lower",
        vmin        = 0.0,
        vmax        = 1.0,
    )

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


def confidence_bar(prob: float, threshold: float = 0.276):
    """Render a simple HTML progress-bar showing tumour probability."""
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
        <!-- threshold marker -->
        <div style="position:absolute; left:{int(threshold*100)}%;
                    top:0; bottom:0; width:2px; background:#f6ad55;"></div>
      </div>
      <div style="text-align:right; margin-top:3px;">
        <span style="color:#f6ad55; font-size:0.75rem;">
            ▲ Decision threshold ({int(threshold*100)}%)
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPER — INTERPRETATION & PDF REPORT
# ════════════════════════════════════════════════════════════════════════════

def get_interpretation(grid_8x8: np.ndarray) -> str:
    """Determine the region of highest attention on an 8x8 grid."""
    idx = np.unravel_index(np.argmax(grid_8x8), grid_8x8.shape)
    y, x = idx # row (y), col (x)
    v_pos = "upper" if y > 3 else "lower"
    h_pos = "left"  if x < 4 else "right"
    return f"High attention observed in the {v_pos}-{h_pos} region, indicating possible tumour features in this area."

def generate_report(result: dict, filename: str) -> bytes:
    """Generate a medical-style PDF report and return as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=22, alignment=1, spaceAfter=20
    )
    header_style = ParagraphStyle(
        'HeaderStyle', parent=styles['Heading2'], fontSize=16, color=colors.HexColor("#0f3460"), spaceBefore=15
    )
    
    elements = []
    
    # Header
    elements.append(Paragraph("🔬 ColonAI — Clinical Detection Report", title_style))
    elements.append(Paragraph(f"Slide Identification: {os.path.basename(filename)}", styles["Normal"]))
    elements.append(Paragraph(f"Date Generated: {np.datetime64('now')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    
    # Section 1: Summary
    elements.append(Paragraph("📌 Diagnostic Summary", header_style))
    status_color = "#ff4444" if result["prediction"] == 1 else "#44ff44"
    elements.append(Paragraph(
        f"<b>Result:</b> <font color='{status_color}'>{result['label'].upper()}</font>", 
        styles["Normal"]
    ))
    elements.append(Paragraph(f"<b>Probability:</b> {result['probability']:.4f}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {result['confidence']:.4f}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Patch Count:</b> {result['n_patches']:,}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    
    # Section 2: Interpretation
    elements.append(Paragraph("🧠 AI Interpretation", header_style))
    interpretation = get_interpretation(result["heatmap_8x8"])
    elements.append(Paragraph(interpretation, styles["Normal"]))
    elements.append(Spacer(1, 12))
    
    # Section 3: Visualization (Heatmaps)
    elements.append(Paragraph("🌡️ Attention Visualization", header_style))
    elements.append(Paragraph("Heatmaps show focal regions where the model concentrated its prediction effort.", styles["Italic"]))
    
    # Save heatmaps to temporary buffers for embedding
    def grid_to_img(grid, title):
        fig = render_heatmap(grid, title, figsize=(4, 3))
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=100, facecolor='#ffffff')
        plt.close(fig)
        img_buffer.seek(0)
        return Image(img_buffer, width=180, height=135)

    img8 = grid_to_img(result["heatmap_8x8"], "Coarse (8x8)")
    img16 = grid_to_img(result["heatmap_16x16"], "Fine (16x16)")
    
    table_data = [[img8, img16]]
    t = Table(table_data)
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Section 4: Methodology
    elements.append(Paragraph("🏗️ Methodology Details", header_style))
    elements.append(Paragraph("Model Arch: CLAM-SB (Attention-based MIL)", styles["Normal"]))
    elements.append(Paragraph("Inference: 5-Fold Ensemble Voting", styles["Normal"]))
    elements.append(Paragraph(f"Decision Threshold: {THRESHOLD}", styles["Normal"]))
    
    # Footer Disclaimer
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("<b>Disclaimer:</b> For research and demonstration purposes only. Not for clinical diagnosis.", styles["Italic"]))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 ColonAI")
    st.markdown("**CLAM · Attention MIL**")
    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This tool analyses pre-extracted features from
    **colon histopathology whole-slide images (WSIs)**
    and classifies them as **Tumour** or **Normal**.

    Built with:
    - 🧠 **CLAM** (attention-based MIL)
    - 🏥 **TCGA-COAD** dataset
    - 🔁 **5-fold cross-validation**
    """)

    st.markdown("---")
    st.markdown("### Model Status")
    if MODEL_PATHS:
        st.success(f"✅  {len(MODEL_PATHS)} fold models loaded")
    else:
        st.error("❌  No model checkpoints found")

    st.markdown("---")
    st.markdown("### Decision Threshold")
    st.info(f"Threshold = **{THRESHOLD}**  \n(tuned to maximise F1)")

    st.markdown("---")
    st.caption("Buddham Rajbhandari · Kaviya Darshini · Dakshini")


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 10px 0 6px 0;">
    <h1 style="font-size:2.4rem; font-weight:700; margin-bottom:4px;">
        🔬 ColonAI — Cancer Detection
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


# ── File Upload ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-heading">📂 Analysis Input</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Upload a <code>.pt</code> file containing pre-extracted features for one WSI.
Expected format: <code>{'features': [N, 2048], 'coords': [N, 2]}</code>
</div>
""", unsafe_allow_html=True)
st.markdown(" ")

uploaded = st.file_uploader(
    "Drop a .pt feature bag here",
    type=["pt"],
    label_visibility="collapsed",
)


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE — runs when file is uploaded
# ════════════════════════════════════════════════════════════════════════════
if uploaded is not None:

    if not MODEL_PATHS:
        st.error("No trained model checkpoints found in models/five_fold/. "
                 "Please train the model first.")
        st.stop()

    # ── Save upload to a temp file (run_full_inference needs a path) ─────────
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ── Run ensemble inference ───────────────────────────────────────────────
    with st.spinner("Running ensemble inference across all 5 fold models…"):
        try:
            result = run_full_inference(tmp_path, MODEL_PATHS)
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

    # Try to decode true label from TCGA filename
    true_label = get_true_label(uploaded.name)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # PREDICTION RESULT
    # ════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-heading">🧬 Slide-Level Prediction</p>',
                unsafe_allow_html=True)

    col_res, col_detail = st.columns([1, 1.6], gap="large")

    with col_res:
        if pred == 1:
            st.markdown("""
            <div class="result-card tumor-card">
                <h1>🔴 TUMOUR</h1>
                <p>Malignant tissue detected</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card normal-card">
                <h1>🟢 NORMAL</h1>
                <p>No malignancy detected</p>
            </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown(f"**Uploaded Slide:** `{uploaded.name}`")
        st.markdown("**Tumour Probability**")
        confidence_bar(prob, THRESHOLD)

        d1, d2, d3 = st.columns(3)
        with d1: st.metric("Probability %", f"{prob*100:.1f}%")
        with d2: st.metric("Confidence",   f"{confidence:.3f}")
        with d3: st.metri    st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 20px 0;">
    ColonAI · Diagnostic Tool ·
    Buddham Rajbhandari &nbsp;·&nbsp; Kaviya Darshini &nbsp;·&nbsp; Dakshini
    <br>
    <span style="font-size:0.72rem;">
    For research purposes only. Not for clinical diagnosis.
    </span>
</div>
""", unsafe_allow_html=True)
�═══════════════════════════════════════════════════
st.markdown('<p class="section-heading">⚠️ Model Limitations & Known Failure Cases</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
<strong>What can go wrong?</strong><br><br>

🔸 <strong>False Negatives (missed tumours)</strong>  — The model predicts
<em>Normal</em> for a slide that is actually tumour.  This typically occurs
when the tumour focus is small, scattered, or poorly differentiated, making
it difficult for the attention mechanism to assign high scores to the correct
regions.  In our 5-fold evaluation, <strong>4 out of 23 tumour slides</strong>
were missed (sensitivity = 81.8 %).<br><br>

🔸 <strong>Under-confident predictions</strong>  — Because training used only
46 balanced slides, the model's raw sigmoid probabilities are systematically
lower than 0.5 even for true tumours.  We compensate with a tuned threshold
of <strong>0.276</strong>, but this means small distribution shifts (staining
differences, scanner changes) could push borderline cases across the boundary.<br><br>

🔸 <strong>Attention ≠ Pathologist annotation</strong>  — High-attention
regions are statistically correlated with the slide label, but are <em>not</em>
guaranteed to correspond to tumour cells as defined by a pathologist.
Always confirm results with expert review.
</div>
""", unsafe_allow_html=True)

# ── Show known False Negative from training results ──────────────────────────
st.markdown(" ")
with st.expander("📋  Known False Negative Example from Cross-Validation"):
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
        This tumour slide scored only <strong>0.196</strong> — well below
        the 0.276 threshold — and was therefore classified as Normal.
        The attention heatmap shows diffuse low-level activation with no
        strong focal hotspot, suggesting the tumour signal is spread thinly
        across the tissue rather than concentrated in a single region.
        This is a <em>challenging case</em> even for expert pathologists.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS  (collapsible)
# ════════════════════════════════════════════════════════════════════════════
with st.expander("🏗️  How the Pipeline Works"):
    st.markdown("""
    ```
    WSI (.svs)
       ↓  Tissue segmentation + patch extraction (256 × 256 px @ 20×)
       ↓  ResNet-50 feature extraction  →  [N, 2048] per patch
       ↓  Saved as .pt file  { features, coords }
          ──── your upload starts here ────
       ↓  AttentionNet: Linear(2048→512) → Tanh → Dropout → Linear(512→1)
       ↓  Softmax over ALL N patches  →  attention distribution [N]
       ↓  Weighted sum  →  slide representation [2048]
       ↓  Classifier: Linear(2048→1) + Sigmoid  →  probability
       ↓  5 × fold ensemble (average probabilities + attention)
       ↓  Threshold @ 0.276  →  Tumour / Normal
       ↓  Grid reconstruction (8×8, 16×16)  →  heatmaps
    ```
    **Training:** 5-fold stratified cross-validation · 46 slides (23 tumour + 23 normal)
    · Adam (LR=5e-5) · BCE loss · Early stopping on validation loss
    """)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 8px 0 20px 0;">
    ColonAI · CLAM-based MIL for Colon Cancer Detection ·
    Buddham Rajbhandari &nbsp;·&nbsp; Kaviya Darshini &nbsp;·&nbsp; Dakshini
    <br>
    <span style="font-size:0.72rem;">
    For research and demonstration purposes only.
    Not a substitute for expert pathological diagnosis.
    </span>
</div>
""", unsafe_allow_html=True)
