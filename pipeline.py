"""
pipeline.py
===========
Clean inference pipeline for the CLAM MIL colon-cancer detection project.

Usage
-----
    # Run with defaults (picks first .pt in features/ and fold-0 model)
    python pipeline.py

    # Run with specific slide and enable all-fold ensemble heatmap
    python pipeline.py --slide features/TCGA-AA-3529-01Z-00-DX1.99453fef.pt \
                       --ensemble

What this script does
---------------------
  1. Load feature bag (.pt) from FEATURE_DIR
  2. Load trained model (single fold or ensemble of all folds)
  3. Run model → slide-level prediction + probability
  4. Extract FULL attention (all N patches, no top-K cutoff)
  5. Build 8×8 and 16×16 attention grids from patch coordinates
  6. Save heatmaps (heatmap_8x8.png, heatmap_16x16.png)
  7. Print prediction, true label, confidence
  8. (Optional) ROC curve and confusion matrix if ensemble CSV exists
"""

import os
import glob
import argparse

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Local imports ─────────────────────────────────────────────────────────────
from clam_model import CLAM_SB
from heatmap_utils import (
    get_full_attention,
    create_attention_grid,
    plot_heatmap,
    plot_roc_curve,
    plot_confusion_matrix,
    ensemble_attention,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_DIR        = "features"
MODEL_DIR          = os.path.join("models", "five_fold")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_fold_0.pt")
ENSEMBLE_CSV       = os.path.join(MODEL_DIR, "ensemble_predictions.csv")

OUTPUT_DIR         = "outputs"      # all generated images go here
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD FEATURE BAG
# ══════════════════════════════════════════════════════════════════════════════

def load_features(path: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Load a pre-extracted feature bag from a .pt file.

    Supports two formats:
        dict  → keys 'features' [N, D] and optionally 'coords' [N, 2]
        tensor → raw [N, D] tensor (legacy format, no coords)

    Returns
    -------
    features : torch.Tensor  [N, D]
    coords   : torch.Tensor  [N, 2]  or  None
    """
    data = torch.load(path, map_location="cpu")

    if isinstance(data, dict):
        features = data.get("features", data.get("feats"))
        coords   = data.get("coords")
    elif torch.is_tensor(data):
        features = data
        coords   = None
        print("  [WARN] No coordinates found — heatmaps will be skipped.")
    else:
        raise ValueError(f"Unsupported .pt format in {path}: {type(data)}")

    if features is None:
        raise ValueError(f"Could not find 'features' key in {path}")

    return features.float(), coords


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> CLAM_SB:
    """
    Instantiate CLAM_SB and load weights from a checkpoint.
    Handles DataParallel 'module.' prefixes and wrapped state-dicts.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = CLAM_SB()
    state = torch.load(model_path, map_location="cpu")

    # Unwrap if the checkpoint contains a wrapper dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif hasattr(state, "state_dict"):
        state = state.state_dict()

    # Remove DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RUN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model: CLAM_SB, features: torch.Tensor) -> tuple[float, float, int]:
    """
    Forward pass through the model.

    Returns
    -------
    logit : raw scalar output from the classifier
    prob  : sigmoid probability of Tumor (class 1)
    pred  : binary prediction  (1 = Tumor, 0 = Normal)
    """
    features = features.to(device)
    with torch.no_grad():
        logits, _ = model(features)  # top-K attention not used here
        prob  = torch.sigmoid(logits).item()
        pred  = int(prob >= 0.5)
    return logits.item(), prob, pred


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — DECODE TRUE LABEL FROM TCGA FILENAME
# ══════════════════════════════════════════════════════════════════════════════

def get_true_label(filename: str) -> int | None:
    """
    TCGA naming convention:
        -01Z- / -01A-  →  primary tumour   → label 1
        -11A- / -11B-  →  normal adjacent  → label 0
    Returns None if the code cannot be parsed.
    """
    stem  = os.path.basename(filename).replace(".pt", "")
    parts = stem.split("-")
    if len(parts) >= 4:
        code = parts[3]
        if code.startswith("01"):
            return 1
        if code.startswith("11"):
            return 0
    return None


# ══════════════════════════════════════════════════════════════════════════════
# REUSABLE INFERENCE FUNCTION  (Streamlit / notebook friendly)
# ══════════════════════════════════════════════════════════════════════════════

def run_full_inference(pt_path: str, model_paths: list[str]) -> dict:
    """
    Run the complete CLAM MIL inference pipeline on a single slide.

    This function is designed to be called from a UI (e.g. Streamlit) or
    any external script.  It does NOT write files or print to stdout — it
    simply returns a plain Python dict with all results.

    Parameters
    ----------
    pt_path     : str
        Absolute or relative path to a .pt feature bag containing:
            { 'features': Tensor[N, 2048],  'coords': Tensor[N, 2] }

    model_paths : list[str]
        Ordered list of trained checkpoint paths (one per fold).
        ALL models are used for ensemble prediction and attention averaging.
        At least one path must be valid.

    Returns
    -------
    dict with keys:
        "probability"   : float        — avg sigmoid probability of Tumor
        "prediction"    : int          — 0 (Normal) or 1 (Tumor)
        "label"         : str          — "Tumor" or "Normal"
        "confidence"    : float        — distance from threshold (0.276)
        "n_patches"     : int          — total number of patches in this slide
        "attention"     : np.ndarray   — full attention scores, shape [N]
        "heatmap_8x8"   : np.ndarray   — normalised grid, shape [8, 8]
        "heatmap_16x16" : np.ndarray   — normalised grid, shape [16, 16]
        "has_coords"    : bool         — False if coords were absent (no heatmaps)

    Raises
    ------
    FileNotFoundError  if pt_path does not exist or no valid checkpoint is found
    ValueError         if the .pt file has an unsupported format
    """

    # ── 0. Validate inputs ────────────────────────────────────────────────────
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Feature file not found: {pt_path}")

    valid_paths = [p for p in model_paths if os.path.exists(p)]
    if not valid_paths:
        raise FileNotFoundError(
            f"No valid model checkpoints found in: {model_paths}"
        )

    # Force CPU so the function works on any machine without a GPU requirement
    infer_device = torch.device("cpu")

    # ── 1. Load features and coords ───────────────────────────────────────────
    features, coords = load_features(pt_path)   # [N, D], [N, 2] or None
    features = features.to(infer_device)
    N = features.shape[0]

    # ── 2. Ensemble prediction + attention ────────────────────────────────────
    #
    #  For every fold model:
    #    a. Load model weights
    #    b. Forward pass  → sigmoid probability
    #    c. Extract FULL attention [N, 1]  (no top-K filtering)
    #
    #  Then average probabilities and attention vectors across all fold models.

    all_probs    = []   # one float per fold
    all_attentions = [] # one [N, 1] tensor per fold

    with torch.no_grad():
        for ckpt_path in valid_paths:
            # --- load model onto CPU ---
            model = load_model(ckpt_path)
            model = model.to(infer_device)
            model.eval()

            # --- slide-level probability ---
            logits, _ = model(features)            # top-K branch (not used for heatmap)
            prob_fold  = torch.sigmoid(logits).item()
            all_probs.append(prob_fold)

            # --- full attention (all N patches, no top-K) ---
            A_fold = get_full_attention(model, features)   # [N, 1]
            all_attentions.append(A_fold)

    # Average probability across folds
    avg_prob   = float(np.mean(all_probs))

    # Average attention across folds  →  [N, 1]  →  flatten to [N]
    avg_attention = torch.stack(all_attentions, dim=0).mean(dim=0)  # [N, 1]
    avg_attention = avg_attention.squeeze(1).numpy()                 # [N]

    # ── 3. Classification with tuned threshold ────────────────────────────────
    THRESHOLD  = 0.276                           # tuned via F1 on training folds
    prediction = int(avg_prob >= THRESHOLD)
    label_str  = "Tumor" if prediction == 1 else "Normal"
    confidence = abs(avg_prob - THRESHOLD)       # how far from the decision boundary

    # ── 4. Grid reconstruction ────────────────────────────────────────────────
    #
    #  Only possible when coords are available in the .pt file.
    #  coords [N, 2] gives the (x, y) pixel position of every patch on the WSI.

    has_coords      = coords is not None
    heatmap_grid    = None
    heatmap_spatial = None

    if has_coords:
        coords_np = coords.numpy() if torch.is_tensor(coords) else np.array(coords)
        x = coords_np[:, 0]
        y = coords_np[:, 1]
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1
        aspect = y_range / x_range
        # Set max side of the coarse grid to 16
        if aspect >= 1:
            grid_h, grid_w = 16, max(1, int(16 / aspect))
        else:
            grid_h, grid_w = max(1, int(16 * aspect)), 16

        heatmap_grid = create_attention_grid(coords, avg_attention, grid_size=(grid_h, grid_w))
        
        from heatmap_utils import create_spatial_heatmap
        heatmap_spatial = create_spatial_heatmap(coords, avg_attention, patch_size=256)

    # ── 5. Return results dict ────────────────────────────────────────────────
    return {
        "probability":     avg_prob,
        "prediction":      prediction,
        "label":           label_str,
        "confidence":      confidence,
        "n_patches":       N,
        "attention":       avg_attention,           # np.ndarray [N]
        "heatmap_grid":    heatmap_grid,            # np.ndarray [H_g, W_g] or None
        "heatmap_spatial": heatmap_spatial,         # np.ndarray [H, W, 3] or None
        "has_coords":      has_coords,
        "coords":          coords_np if has_coords else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    print("\n" + "="*60)
    print("  CLAM MIL Inference Pipeline")
    print("="*60)

    # ── Pick slide file ───────────────────────────────────────────────────────
    if args.slide:
        slide_path = args.slide
    else:
        pt_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {FEATURE_DIR}")
        slide_path = pt_files[0]

    print(f"\n  Slide  : {slide_path}")

    # ── Load features and coordinates ────────────────────────────────────────
    features, coords = load_features(slide_path)
    N, D = features.shape
    print(f"  Patches: {N:,}   Feature dim: {D}")

    # ── Load model (single fold or ensemble) ──────────────────────────────────
    all_fold_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "best_fold_*.pt")))

    if args.ensemble and len(all_fold_paths) > 1:
        print(f"\n  Mode   : ensemble of {len(all_fold_paths)} fold models")
        # For prediction: use fold-0 model (representative)
        model = load_model(all_fold_paths[0])
    else:
        model_path = args.model if args.model else DEFAULT_MODEL_PATH
        print(f"\n  Model  : {model_path}")
        model = load_model(model_path)

    # ── Inference ─────────────────────────────────────────────────────────────
    logit, prob, pred = run_inference(model, features)
    true_label = get_true_label(slide_path)

    print(f"\n{'─'*60}")
    print(f"  Logit      : {logit:+.4f}")
    print(f"  Probability: {prob:.4f}  ({'Tumor' if prob >= 0.5 else 'Normal'})")
    print(f"  Prediction : {'Tumor (1)' if pred == 1 else 'Normal (0)'}")
    if true_label is not None:
        correct = "✓ CORRECT" if pred == true_label else "✗ WRONG"
        print(f"  True Label : {'Tumor (1)' if true_label == 1 else 'Normal (0)'}  {correct}")
    print(f"{'─'*60}")

    # ── Attention extraction ──────────────────────────────────────────────────
    if coords is None:
        print("\n  [SKIP] No coordinates — heatmaps cannot be generated.")
        return

    features_dev = features.to(device)

    if args.ensemble and len(all_fold_paths) > 1:
        print(f"\n  Extracting ensemble attention from {len(all_fold_paths)} models…")
        A = ensemble_attention(CLAM_SB, all_fold_paths, features_dev, device=device)
    else:
        print("\n  Extracting full attention from single model…")
        A = get_full_attention(model, features_dev)

    A = A.cpu()   # [N, 1]
    print(f"  Attention shape: {tuple(A.shape)}  (all {N:,} patches covered)")

    # ── Grid reconstruction + heatmaps ────────────────────────────────────────
    print("\n  Generating dual attention heatmaps…")
    
    # Calculate grid size dynamically
    coords_np = coords.numpy() if torch.is_tensor(coords) else np.array(coords)
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range == 0: x_range = 1
    if y_range == 0: y_range = 1
    aspect = y_range / x_range
    if aspect >= 1:
        grid_h, grid_w = 16, max(1, int(16 / aspect))
    else:
        grid_h, grid_w = max(1, int(16 * aspect)), 16

    grid = create_attention_grid(coords, A, grid_size=(grid_h, grid_w))
    out_path_grid = os.path.join(OUTPUT_DIR, f"heatmap_grid.png")
    slide_name = os.path.basename(slide_path).replace(".pt", "")
    label_str  = "Tumor" if pred == 1 else "Normal"

    plot_heatmap(
        grid,
        title=f"Coarse Attention Grid ({grid_h}x{grid_w}) | {label_str} (p={prob:.2f})",
        output_path=out_path_grid,
    )

    from heatmap_utils import create_spatial_heatmap
    import cv2
    spatial_heatmap = create_spatial_heatmap(coords, A, patch_size=256)
    out_path_spatial = os.path.join(OUTPUT_DIR, f"heatmap_spatial.png")
    
    # spatial_heatmap is RGB uint8 image, let's use cv2 or plt to save
    spatial_heatmap_bgr = cv2.cvtColor(spatial_heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path_spatial, spatial_heatmap_bgr)
    print(f"  Saved: {out_path_spatial}")

    # ── Optional: ROC curve + confusion matrix from ensemble CSV ─────────────
    if os.path.exists(ENSEMBLE_CSV):
        print(f"\n  Loading ensemble predictions from {ENSEMBLE_CSV}…")
        df = pd.read_csv(ENSEMBLE_CSV)

        # Tune threshold to maximise F1 for the plot
        from train_clam_5fold import tune_threshold
        threshold = tune_threshold(df["label"].values, df["prob"].values)
        preds_all = (df["prob"].values >= threshold).astype(int)

        roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
        cm_path  = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

        plot_roc_curve(
            df["label"].values,
            df["prob"].values,
            output_path=roc_path,
            threshold=threshold,
        )

        plot_confusion_matrix(
            df["label"].values.astype(int),
            preds_all,
            output_path=cm_path,
        )

        auc = roc_auc_score(df["label"].values, df["prob"].values)
        print(f"\n  Ensemble AUC (all slides): {auc:.4f}")
        print(f"  Optimal threshold        : {threshold:.4f}")

    # ── Save single-slide metrics report ──────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "inference_report.txt")
    with open(report_path, "w") as fh:
        fh.write("=== CLAM Inference Report ===\n\n")
        fh.write(f"Slide       : {slide_path}\n")
        fh.write(f"Patches     : {N}\n")
        fh.write(f"Feature dim : {D}\n")
        fh.write(f"Logit       : {logit:.4f}\n")
        fh.write(f"Probability : {prob:.4f}\n")
        fh.write(f"Prediction  : {pred} ({'Tumor' if pred == 1 else 'Normal'})\n")
        if true_label is not None:
            fh.write(f"True Label  : {true_label}\n")
            fh.write(f"Correct     : {pred == true_label}\n")
    print(f"\n  Report saved: {report_path}")

    print("\n  ✅  Pipeline complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CLAM MIL Inference Pipeline")
    p.add_argument(
        "--slide",  type=str, default=None,
        help="Path to a .pt feature file. Defaults to first file in features/.",
    )
    p.add_argument(
        "--model",  type=str, default=None,
        help="Path to a model checkpoint. Defaults to models/five_fold/best_fold_0.pt.",
    )
    p.add_argument(
        "--ensemble", action="store_true",
        help="Average attention from all available fold models for heatmap.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
