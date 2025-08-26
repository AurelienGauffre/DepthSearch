#!/usr/bin/env python3
"""
Plotting utilities (no globals). Pass the mask you want to overlay.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Return a truncated version of a colormap"""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


# Create custom truncated viridis (cut at 80% - remove top 20%)
viridis_custom = truncate_colormap(plt.cm.viridis, 0.0, 0.8)


def setup_matplotlib_theme():
    mpl.rcParams.update({
        "figure.figsize": (10, 7),
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
    # Light grey background everywhere
    "figure.facecolor": "#ebebeb",
    "axes.facecolor": "#ebebeb",
    "savefig.facecolor": "#ebebeb",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _legend_with_bg(ax, loc="upper right"):
    leg = ax.legend(loc=loc, frameon=True)
    if leg is not None:
        leg.get_frame().set_facecolor((1, 1, 1, 0.7))
        leg.get_frame().set_edgecolor("none")
    return leg


def _get_bathy_color_scale(depth_map: np.ndarray, max_alt: float = 0.0):
    """Calculate a reasonable color scale for bathymetry.
    """
    finite_depths = depth_map[np.isfinite(depth_map)]
    if len(finite_depths) == 0:
        return -200, max_alt
    
    vmin = float(np.min(finite_depths))  # Maximum depth found
    vmax = min(float(np.max(finite_depths)), max_alt)  # Limited to max_alt
    
    return vmin, vmax


def plot_full_map(depth_map: np.ndarray, mask: np.ndarray, out_path: str):
    setup_matplotlib_theme()
    fig, ax = plt.subplots()
    vmin, vmax = _get_bathy_color_scale(depth_map)
    im = ax.imshow(depth_map, cmap=viridis_custom, origin="upper", interpolation="nearest", 
                   vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Depth (m)")
    ax.set_title("Full bathymetry with mask")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect('equal')

    if mask is not None:
        land = ~mask
        if land.any():
            overlay = np.zeros((depth_map.shape[0], depth_map.shape[1], 4), dtype=float)
            overlay[land] = [1.0, 0.0, 0.0, 0.35]
            ax.imshow(overlay, origin="lower", interpolation="nearest")

    fig.savefig(out_path)
    plt.close(fig)


def plot_map_with_points(depth_map, mask, true_traj, est, subtitle):
    setup_matplotlib_theme()
    fig, ax = plt.subplots()
    vmin, vmax = _get_bathy_color_scale(depth_map)
    im = ax.imshow(depth_map, origin="upper", cmap=viridis_custom, interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.9).set_label("Depth (m)")

    # True start: orange plus
    ax.scatter([true_traj.xs_int[0]], [true_traj.ys_int[0]],
               marker="+", s=360, linewidths=3.0, color="#ff9500", alpha=0.95,
               label="True start")
    if est is not None:
        # Predicted start: red cross
        ax.scatter([est.x0], [est.y0],
                   marker="x", s=360, linewidths=3.0, color="#ff1a1a", alpha=0.95,
                   label="Predicted start")

    if mask is not None:
        land = ~mask
        if land.any():
            overlay = np.zeros((depth_map.shape[0], depth_map.shape[1], 4), dtype=float)
            overlay[land] = [1.0, 0.0, 0.0, 0.25]
            ax.imshow(overlay, origin="upper", interpolation="nearest")

    ax.set_title("True (+) & Predicted (×) starts\n" + subtitle)
    ax.set_aspect('equal')
    _legend_with_bg(ax)
    return fig


def plot_profile_match(z_meas, est, z_true, using_bias, ylims, xlims):
    setup_matplotlib_theme()
    # 1-based x-axis: measurement number starting at 1
    t = np.arange(1, len(z_meas) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Light grey background is set globally by theme

    # Colors
    col_orange = "#ff9500"
    col_red = "#ff1a1a"

    # Lines
    l_meas, = ax.plot(t, z_meas, color=col_orange, label="Measured")
    l_est,  = ax.plot(t, est.h_est, color=col_red, linestyle="--", label=("Predicted (+b)" if using_bias else "Predicted"))
    l_true = None
    if z_true is not None:
        l_true, = ax.plot(t, z_true, color=col_orange, linestyle=":", label="True profile")

    # Cross markers on top
    ax.scatter(t, z_meas, marker="x", s=28, color=col_orange, zorder=(l_meas.get_zorder() + 1))
    ax.scatter(t, est.h_est, marker="x", s=28, color=col_red, zorder=(l_est.get_zorder() + 1))
    if l_true is not None:
        ax.scatter(t, z_true, marker="+", s=36, linewidths=1.2, color=col_orange, zorder=(l_true.get_zorder() + 1))
    ax.set_xlabel("# of measures")
    ax.set_ylabel("Depth (m)")
    if ylims:
        ax.set_ylim(*ylims)
    if xlims:
        ax.set_xlim(*xlims)
    ax.set_title(f"Profile match | MSE={est.mse:.3f}")
    _legend_with_bg(ax, loc="best")
    return fig


def plot_mse_map(depth_map: np.ndarray,
                 mask: np.ndarray,
                 x0_grid: np.ndarray,
                 y0_grid: np.ndarray,
                 mse_map: np.ndarray,
                 title: str = "MSE map"):
    """Visualize the MSE values computed on a sparse (x0_grid, y0_grid) as a
    continuous overlay using imshow with proper extents over the bathymetry.
    """
    setup_matplotlib_theme()
    H, W = depth_map.shape

    fig, ax = plt.subplots()

    # Background bathymetry for context (subtle grayscale)
    vmin, vmax = _get_bathy_color_scale(depth_map)
    ax.imshow(depth_map, origin="lower", cmap="gray", alpha=0.35, interpolation="nearest",
              vmin=vmin, vmax=vmax)

    # Robust color scaling from finite MSE values
    finite = mse_map[np.isfinite(mse_map)]
    if finite.size == 0:
        # Nothing to show; present background with a note
        ax.text(0.5, 0.5, "No MSE values (all NaN)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.set_title(title)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_aspect('equal')
        return fig

    if finite.size >= 20:
        vmin, vmax = np.nanpercentile(finite, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if vmin == vmax:
        vmax = vmin + 1e-6

    # Colormap with transparent NaNs so non-evaluated grid cells show background
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad((0, 0, 0, 0))

    # Map the (rows, cols) of mse_map onto pixel-space using extent
    # Assumes x0_grid and y0_grid are sorted ascending and represent pixel indices
    extent = [float(x0_grid[0]) - 0.5,
              float(x0_grid[-1]) + 0.5,
              float(y0_grid[0]) - 0.5,
              float(y0_grid[-1]) + 0.5]

    im = ax.imshow(mse_map, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", extent=extent, alpha=0.95)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("MSE")

    # Subtle land overlay on top
    if mask is not None:
        land = ~mask
        if np.any(land):
            overlay = np.zeros((H, W, 4), dtype=float)
            overlay[land] = [1.0, 0.0, 0.0, 0.15]
            ax.imshow(overlay, origin="lower", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect('equal')
    return fig


def plot_topk_starts_on_map(depth_map, mask, true_traj, topk, subtitle, k=5):
    """Plot classical depth map with:
    - true start (+)
    - top-K candidate start positions (small black circles)
    - best start as red ×
    - true trajectory path
    """
    setup_matplotlib_theme()
    fig, ax = plt.subplots()
    vmin, vmax = _get_bathy_color_scale(depth_map)
    im = ax.imshow(depth_map, origin="upper", cmap=viridis_custom, interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.9).set_label("Depth (m)")

    # True trajectory path (small black stars)
    if true_traj is not None:
        ax.scatter(true_traj.xs, true_traj.ys, marker="*", s=18, color="black", alpha=0.9, label="Trajectory")
    # True start: orange plus
    ax.scatter([true_traj.xs_int[0]], [true_traj.ys_int[0]], marker="+", s=360, linewidths=3.0,
           color="#ff9500", alpha=0.95, label="True start")

    # Land overlay
    if mask is not None:
        land = ~mask
        if np.any(land):
            overlay = np.zeros((depth_map.shape[0], depth_map.shape[1], 4), dtype=float)
            overlay[land] = [1.0, 0.0, 0.0, 0.25]
            ax.imshow(overlay, origin="upper", interpolation="nearest")

    # Top-K candidate starts
    if topk:
        black_label_added = False
        for i, cand in enumerate(topk[:k]):
            if i == 0:
                # Predicted start: red cross
                ax.scatter([cand.x0], [cand.y0], marker="x", s=360, linewidths=3.0, color="#ff1a1a",
                           alpha=0.95, label="Predicted start")
            else:
                label = None if black_label_added else "Top-K predicted starts"
                black_label_added = True
                ax.scatter([cand.x0], [cand.y0], s=36, facecolors='none', edgecolors='black', linewidths=1.5,
                           label=label)

    ax.set_title("Top-K candidate starts and trajectory\n" + subtitle)
    ax.set_aspect('equal')
    
    # Create legend with 20% smaller font size and smaller markers
    leg = ax.legend(loc="upper right", frameon=True, fontsize='small', prop={'size': 8.8}, markerscale=0.8)
    if leg is not None:
        leg.get_frame().set_facecolor((1, 1, 1, 0.7))
        leg.get_frame().set_edgecolor("none")
    
    return fig


def plot_accuracy_vs_param(sweep_param: str,
                           sweep_values,
                           accuracies,
                           ylabel: str = "Accuracy (≤ 2×grid stride)"):
    """Line plot of accuracy vs. a swept parameter.

    Returns the Matplotlib figure; caller is responsible for saving/closing.
    """
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sweep_values, accuracies, marker="o")
    ax.set_xlabel(sweep_param)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    return fig


def mse_to_probability_map(mse_map: np.ndarray,
                           valid_grid_mask: np.ndarray,
                           temperature: float = 1.0) -> np.ndarray:
    """Convert MSE values on the grid to a normalized probability map.

    Uses a softmax-like transform p ∝ exp(-(mse - m_min)/T) on finite/valid cells.
    Returns an array of same shape as mse_map, zeros elsewhere, sum≈1 over valid.
    """
    prob = np.zeros_like(mse_map, dtype=float)
    if mse_map.size == 0:
        return prob
    mask = np.isfinite(mse_map)
    if valid_grid_mask is not None:
        mask = mask & valid_grid_mask
    vals = mse_map[mask]
    if vals.size == 0:
        return prob
    m_min = float(np.nanmin(vals))
    T = max(float(temperature), 1e-6)
    logits = - (vals - m_min) / T
    logits -= float(np.max(logits))  # stabilize
    exps = np.exp(logits)
    Z = float(np.sum(exps))
    if Z <= 0.0 or not np.isfinite(Z):
        return prob
    prob_vals = (exps / Z)
    prob[mask] = prob_vals
    return prob


def plot_probability_map(depth_map: np.ndarray,
                         mask: np.ndarray,
                         prob_map: np.ndarray,
                         x0_grid: np.ndarray,
                         y0_grid: np.ndarray,
                         topk,
                         true_traj,
                         title: str = "Probability map (where we are)"):
    """Plot probability over start positions with requested styling, plus markers:
    - grayscale: black=0, white=max probability
    - top-5 predictions: small identical black circles at candidate end points
    - best (most probable) candidate end point: red cross
    - true current point and true trajectory so far
    """
    setup_matplotlib_theme()
    H, W = depth_map.shape
    fig, ax = plt.subplots()

    # Probability overlay (start-grid domain)
    finite = prob_map[np.isfinite(prob_map)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= 0.0 or not np.isfinite(vmax):
        vmax = 1.0
    extent = [float(x0_grid[0]) - 0.5,
              float(x0_grid[-1]) + 0.5,
              float(y0_grid[0]) - 0.5,
              float(y0_grid[-1]) + 0.5]
    # 'inferno' colormap: black low, bright yellow/white high
    im = ax.imshow(prob_map, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax,
                   interpolation="nearest", extent=extent, alpha=1.0)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Probability")

    # Subtle land overlay
    if mask is not None:
        land = ~mask
        if np.any(land):
            overlay = np.zeros((H, W, 4), dtype=float)
            overlay[land] = [1.0, 0.0, 0.0, 0.12]
            ax.imshow(overlay, origin="lower", interpolation="nearest")

    # True trajectory so far and current point
    if true_traj is not None:
        ax.plot(true_traj.xs, true_traj.ys, color="#1f9e5a", linewidth=2.0, alpha=0.8, label="True trajectory")
        ax.scatter([true_traj.xs[-1]], [true_traj.ys[-1]], marker="+", s=300, color="#1f9e5a", linewidths=2.5,
                   label="True current")

    # Top-k candidate end points
    if topk:
        # Small identical black circles for top-k
        for j, cand in enumerate(topk):
            ax.scatter([cand.xs[-1]], [cand.ys[-1]], s=36, facecolors='none', edgecolors='black', linewidths=1.5,
                       label=("Top-K predicted starts" if j == 0 else None))
        # Best candidate: red cross at end point
        best = topk[0]
        ax.scatter([best.xs[-1]], [best.ys[-1]], marker='x', s=200, linewidths=2.5, color='#ff1a1a', label='Best')

    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect('equal')
    _legend_with_bg(ax, loc="upper right")
    return fig



