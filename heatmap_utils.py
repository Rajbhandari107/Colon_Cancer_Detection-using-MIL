"""
heatmap_utils.py
================
Utility module for:
  1. Extracting FULL (non-top-K) attention scores from a CLAM model
  2. Building a 2-D attention grid from patch coordinates
  3. Plotting and saving heatmaps
  4. ROC curve + Confusion matrix visualisation
  5. Multi-fold attention ensemble averaging

Usage example
-------------
    from heatmap_utils import (
        get_full_attention, create_attention_grid, plot_heatmap,
        plot_roc_curve, plot_confusion_matrix, ensemble_attention,
    )
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc as sklearn_auc, confusion_matrix
import cv2

# ──────────────────────────────────────────────────────────────────────────────
# 1.  FULL ATTENTION EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def get_full_attention(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """
    Extract normalised attention scores for **all** N patches — no Top-K filtering.

    The standard CLAM forward pass keeps only the top-K most attended patches,
    which makes heatmaps sparse and misleading.  This function runs the attention
    MLP on every patch and applies a global softmax so we get a proper probability
    distribution over the full bag.

    Parameters
    ----------
    model    : trained CLAM_SB instance (in eval mode)
    features : torch.Tensor of shape [N, D]

    Returns
    -------
    A_full : torch.Tensor of shape [N, 1]
        Softmax-normalised attention weight for every patch.
    """
    model.eval()
    with torch.no_grad():
        # Raw (unnormalised) logit per patch  →  [N, 1]
        raw_A = model.attention_net.attention(features)

        # Numerical stability: subtract max before softmax
        raw_A = raw_A - raw_A.max()

        # Softmax over all N patches  →  [N, 1]
        A_full = F.softmax(raw_A, dim=0)

    return A_full   # shape [N, 1]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GRID RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def create_attention_grid(
    coords:    torch.Tensor | np.ndarray,
    attention: torch.Tensor | np.ndarray,
    grid_size: int | tuple = 8,
) -> np.ndarray:
    """
    Aggregate patch attention scores onto a regular 2-D grid.

    Each patch is assigned to the grid cell that contains its (x, y) coordinate.
    Cells with multiple patches store the **mean** attention; empty cells are 0.
    The returned grid is min-max normalised to [0, 1].

    Parameters
    ----------
    coords    : [N, 2] — (x, y) pixel coordinates of each patch centre
    attention : [N] or [N, 1] — pre-extracted attention scores (full, not top-K)
    grid_size : int or tuple — number of rows/columns (8 → 8×8, (16, 8) → 16×8)

    Returns
    -------
    grid_norm : np.ndarray of shape [grid_rows, grid_cols], values in [0, 1]
    """
    # ── Convert to numpy ──────────────────────────────────────────────────────
    coords_np = coords.numpy() if torch.is_tensor(coords) else np.array(coords)
    attn_np   = attention.numpy() if torch.is_tensor(attention) else np.array(attention)
    attn_np   = attn_np.flatten()   # ensure 1-D

    x = coords_np[:, 0].astype(float)
    y = coords_np[:, 1].astype(float)

    # ── Normalise coordinates to [0, 1) ───────────────────────────────────────
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Avoid division by zero for degenerate slides
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_norm = (x - x_min) / x_range   # ∈ [0, 1]
    y_norm = (y - y_min) / y_range   # ∈ [0, 1]

    # ── Map normalised coords to grid indices ──────────────────────────────────
    if isinstance(grid_size, int):
        grid_rows, grid_cols = grid_size, grid_size
    else:
        grid_rows, grid_cols = grid_size

    # Clamp so that coord == 1.0 ends up in the last cell, not out of bounds
    col_idx = np.clip((x_norm * grid_cols).astype(int), 0, grid_cols - 1)   # x → column
    row_idx = np.clip((y_norm * grid_rows).astype(int), 0, grid_rows - 1)   # y → row

    # ── Accumulate attention and count patches per cell ────────────────────────
    grid   = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    counts = np.zeros((grid_rows, grid_cols), dtype=np.float64)

    for r, c, a in zip(row_idx, col_idx, attn_np):
        grid[r, c]   += a
        counts[r, c] += 1

    # ── Mean attention per cell (avoid divide-by-zero for empty cells) ─────────
    with np.errstate(invalid="ignore"):
        grid_mean = np.where(counts > 0, grid / counts, 0.0)

    # ── Min-max normalise the final grid to [0, 1] ────────────────────────────
    g_min, g_max = grid_mean.min(), grid_mean.max()
    if g_max > g_min:
        grid_norm = (grid_mean - g_min) / (g_max - g_min)
    else:
        # All cells identical (e.g. all zero) — return zeros
        grid_norm = np.zeros_like(grid_mean)

    return grid_norm   # shape [grid_rows, grid_cols]


# ──────────────────────────────────────────────────────────────────────────────
# 2.5. SPATIAL HEATMAP OVERLAY
# ──────────────────────────────────────────────────────────────────────────────

def create_spatial_heatmap(
    coords:    torch.Tensor | np.ndarray,
    attention: torch.Tensor | np.ndarray,
    patch_size: int = 256,
    threshold:  float = 0.0,
    alpha:      float = 0.6,
) -> np.ndarray:
    """
    Generate a high-resolution spatial heatmap mapped to coordinate patches.
    Blends the spatial attention mapping onto a black canvas for maximum contrast.

    Returns
    -------
    overlay_rgb : np.ndarray [H, W, 3] uint8 image
    """
    coords_np = coords.numpy() if torch.is_tensor(coords) else np.array(coords)
    attn_np   = attention.numpy() if torch.is_tensor(attention) else np.array(attention)
    attn_np   = attn_np.flatten()

    x = coords_np[:, 0]
    y = coords_np[:, 1]
    
    # Detect if they are indices vs absolute pixels
    if np.max(x) < 5000 and np.max(y) < 5000:
        raw_px_x = x * patch_size
        raw_px_y = y * patch_size
    else:
        raw_px_x = x
        raw_px_y = y

    # Shift minimum coordinates to 0,0 locally
    raw_px_x = raw_px_x - np.min(raw_px_x)
    raw_px_y = raw_px_y - np.min(raw_px_y)

    # Protect against massive memory allocation / OpenCV limits
    MAX_DIM = 800
    orig_height = np.max(raw_px_y) + patch_size
    orig_width  = np.max(raw_px_x) + patch_size
    
    scale = MAX_DIM / float(max(orig_height, orig_width))
        
    pixel_x = (raw_px_x * scale).astype(int)
    pixel_y = (raw_px_y * scale).astype(int)
    scaled_patch_size = max(1, int(patch_size * scale))

    height = int(np.max(pixel_y) + scaled_patch_size)
    width  = int(np.max(pixel_x) + scaled_patch_size)

    # Fill regions
    heatmap = np.zeros((height, width), dtype=np.float32)
    for px, py, a in zip(pixel_x, pixel_y, attn_np):
        heatmap[py:py+scaled_patch_size, px:px+scaled_patch_size] = float(a)
        
    # Normalize to [0,1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
        
    # Optional thresholding: Suppress low-attention regions
    if threshold > 0.0:
        heatmap[heatmap < threshold] = 0.0
        
    # Gaussian smoothing - correctly scaled to blur the discrete patches
    kernel_size = max(3, int(scaled_patch_size * 2.5) | 1) # ensure odd
    if kernel_size > 101:
        kernel_size = 101
        
    heatmap_blurred = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    
    # Colormap expects uint8 [0,255]
    heatmap_uint8 = (heatmap_blurred * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend exactly as requested
    if alpha < 1.0:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        overlay = cv2.addWeighted(image, 1.0 - alpha, heatmap_rgb, alpha, 0)
    else:
        overlay = heatmap_rgb
        
    # Prevent extreme aspect ratios that get hyper-stretched by Streamlit (which looks 'so zoomed')
    final_h, final_w = overlay.shape[:2]
    if final_h > final_w * 2:
        # Too tall and skinny -> pad width
        pad = (final_h - final_w) // 2
        overlay = cv2.copyMakeBorder(overlay, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif final_w > final_h * 2:
        # Too short and wide -> pad height
        pad = (final_w - final_h) // 2
        overlay = cv2.copyMakeBorder(overlay, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HEATMAP VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    grid:       np.ndarray,
    title:      str = "Attention Heatmap",
    output_path: str | None = None,
    cmap:       str = "jet",
    figsize:    tuple = (8, 6),
) -> None:
    """
    Render a 2-D attention grid as a colour heatmap.

    Parameters
    ----------
    grid        : [G, G] numpy array with values in [0, 1]
    title       : figure title string
    output_path : if given, save figure here (e.g. 'heatmap_8x8.png')
    cmap        : matplotlib colourmap name (default 'jet')
    figsize     : (width, height) in inches
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        grid,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",        # (0,0) at bottom-left — matches WSI convention
        vmin=0.0,
        vmax=1.0,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Attention", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Grid column  (→ x)", fontsize=10)
    ax.set_ylabel("Grid row  (→ y)", fontsize=10)

    # Tick positions at cell centres
    g = grid.shape[0]
    ax.set_xticks(range(g))
    ax.set_yticks(range(g))
    ax.tick_params(labelsize=7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROC CURVE
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    labels:      np.ndarray,
    probs:       np.ndarray,
    output_path: str = "roc_curve.png",
    threshold:   float | None = None,
) -> None:
    """
    Plot and save an ROC curve with AUC annotation.

    Parameters
    ----------
    labels      : ground-truth binary labels [0 / 1]
    probs       : predicted probability of class 1  (real values in [0, 1])
    output_path : path to save the PNG
    threshold   : if given, mark the operating point on the curve
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = sklearn_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"ROC curve  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")

    # Mark the chosen operating threshold
    if threshold is not None:
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(fpr[idx], tpr[idx], s=80, zorder=5, color="#F44336",
                   label=f"Threshold = {threshold:.3f}")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels:      np.ndarray,
    preds:       np.ndarray,
    class_names: list[str] = ["Normal", "Tumor"],
    output_path: str = "confusion_matrix.png",
) -> None:
    """
    Plot and save a labelled confusion matrix.

    Parameters
    ----------
    labels      : ground-truth binary labels
    preds       : binary predictions (after thresholding)
    class_names : list of class name strings  [negative, positive]
    output_path : path to save the PNG
    """
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate each cell with its count and percentage
    total = cm.sum()
    thresh = cm.max() / 2.0   # threshold for label colour contrast
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct   = 100 * count / total
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=color, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MULTI-FOLD ATTENTION ENSEMBLE
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_attention(
    model_class,
    model_paths: list[str],
    features:    torch.Tensor,
    device:      torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Average full attention scores from multiple fold models.

    This gives more stable and reliable heatmaps compared to using a single fold.

    Parameters
    ----------
    model_class  : the CLAM model class (must accept no constructor args for defaults)
    model_paths  : list of .pt checkpoint paths (one per fold)
    features     : [N, D] feature tensor for the slide
    device       : torch device string or object

    Returns
    -------
    A_avg : torch.Tensor of shape [N, 1]
        Element-wise mean of all per-fold softmax attention scores.
    """
    if not model_paths:
        raise ValueError("model_paths must not be empty")

    device    = torch.device(device) if isinstance(device, str) else device
    features  = features.float().to(device)
    A_list    = []

    for path in model_paths:
        if not os.path.exists(path):
            print(f"  [WARN] checkpoint not found, skipping: {path}")
            continue

        model = model_class().to(device)
        state = torch.load(path, map_location=device)

        # Handle wrapped state-dicts (DataParallel, etc.)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif hasattr(state, "state_dict"):
            state = state.state_dict()

        # Strip 'module.' prefix from DataParallel checkpoints
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean, strict=False)

        A_list.append(get_full_attention(model, features))   # [N, 1]

    if not A_list:
        raise RuntimeError("No valid checkpoints found in model_paths")

    # Stack along a new leading dim and average  →  [N, 1]
    A_avg = torch.stack(A_list, dim=0).mean(dim=0)
    return A_avg
