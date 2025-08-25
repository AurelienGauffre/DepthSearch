#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Dict, Any
import numpy as np
import shutil
import matplotlib.pyplot as plt

from configs import CONFIG
from dataloading import load_bathy, BathyDataset
from plotting import (
    plot_map_with_points, plot_profile_match,
    setup_matplotlib_theme, plot_mse_map, plot_topk_starts_on_map,
    plot_accuracy_vs_param,
)
from trajectory import (
    line_samples, in_bounds,
    mask_along_line, make_straight_trajectory, compute_valid_grid_mask, pick_random_valid_scenario,
)
from solver import (
    profile_along_line,
    solverGridSearch,
)

if __name__ == "__main__":
    # This main function essentially runs 2 nested loops : s different scenarios with v sweep values
    out_dir = Path(CONFIG.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds: BathyDataset = load_bathy(CONFIG)
    depth_map = ds.depth
    valid_mask = ds.mask
    H, W = depth_map.shape
    print(f"[main] Depth map ready: H={H}, W={W} | res≈{ds.resolution_m:.1f} m/px")

    x0_grid = np.arange(CONFIG.margin_px, W - CONFIG.margin_px, CONFIG.grid_stride_px, dtype=np.int32)
    y0_grid = np.arange(CONFIG.margin_px, H - CONFIG.margin_px, CONFIG.grid_stride_px, dtype=np.int32)

    sweep_param = CONFIG.sweep_param
    sweep_values = list(CONFIG.sweep_values)
    s = CONFIG.s

    # Worst-case path for scenario generation
    step_worst = float(max(sweep_values)) if sweep_param == "true_step_px" else CONFIG.true_step_px
    n_worst    = int(max(sweep_values))  if sweep_param == "n_samples"    else CONFIG.n_samples

    rng = np.random.default_rng(CONFIG.seed + 999)
    scenarios = [pick_random_valid_scenario(depth_map, valid_mask, step_worst, n_worst, rng)
                 for _ in range(s)]

    success_threshold = 2.0 * CONFIG.grid_stride_px
    original_value = getattr(CONFIG, sweep_param)
    successes_per_value: Dict[Any, int] = {v: 0 for v in sweep_values}

    for si, (x0_true, y0_true, theta_true) in enumerate(scenarios):
        noise_seed_si = CONFIG.seed + 10000 + si

        # Fix profile axes from largest n
        ylims_profiles = None; xlims_profiles = None
        try:
            h_full, *_ = profile_along_line(depth_map, valid_mask, x0_true, y0_true, theta_true,
                                            step=step_worst, n=n_worst, bilinear=CONFIG.bilinear)
            pad = 0.05 * (np.max(h_full) - np.min(h_full) + 1e-6)
            ylims_profiles = (float(np.min(h_full) - pad), float(np.max(h_full) + pad))
            xlims_profiles = (1.0, float(n_worst))
        except Exception:
            pass

        # Recompute validity per sweep value so border candidates valid for smaller n/step aren't excluded.
        for v in sweep_values:
            setattr(CONFIG, sweep_param, v)
            n = int(CONFIG.n_samples)
            step_px = float(CONFIG.true_step_px)

            traj = make_straight_trajectory(depth_map, valid_mask,
                                            x0_true, y0_true, theta_true,
                                            step_px, n,
                                            noise_std=CONFIG.sonar_noise_std,
                                            noise_seed=noise_seed_si,
                                            bilinear=CONFIG.bilinear)

            valid_grid_mask = compute_valid_grid_mask(
                depth_map, valid_mask, theta_true, x0_grid, y0_grid,
                step=step_px, n=n, bilinear=CONFIG.bilinear
            )

            topk, mse_map = solverGridSearch(depth_map, valid_mask, traj.depths_noisy,
                                         step=step_px, n=n, theta=theta_true,
                                         x0_grid=x0_grid, y0_grid=y0_grid,
                                         fit_bias=CONFIG.fit_bias,
                                         valid_grid_mask=valid_grid_mask)
            best = topk[0]

            subtitle = f"res≈{ds.resolution_m:.0f} m/px | step={step_px:.2f} px | n={n} | θ={math.degrees(theta_true):.1f}°"

            fig_map = plot_map_with_points(depth_map, valid_mask, traj, best, subtitle)
            fig_map.savefig(out_dir / f"map_s{si:02d}_{sweep_param}_{v}.png"); plt.close(fig_map)

            fig_prof = plot_profile_match(traj.depths_noisy, best, traj.depths_true,
                                          using_bias=CONFIG.fit_bias,
                                          ylims=ylims_profiles, xlims=xlims_profiles)
            fig_prof.savefig(out_dir / f"profile_s{si:02d}_{sweep_param}_{v}.png"); plt.close(fig_prof)

            # fig_mse = plot_mse_map(depth_map, valid_mask, x0_grid, y0_grid, mse_map,
            #                        title=f"MSE map | {subtitle}")
            # fig_mse.savefig(out_dir / f"mse_map_s{si:02d}_{sweep_param}_{v}.png"); plt.close(fig_mse)

            fig_topk = plot_topk_starts_on_map(depth_map, valid_mask, traj, topk, subtitle, k=5)
            fig_topk.savefig(out_dir / f"topk_starts_s{si:02d}_{sweep_param}_{v}.png"); plt.close(fig_topk)

            err = math.hypot(best.x0 - x0_true, best.y0 - y0_true)
            err_m = err * float(ds.resolution_m)
            successes_per_value[v] += int(err < success_threshold)
            print(f"→ scenario {si}, {sweep_param}={v}: err={err:.2f}px ({err_m:.2f} m)")

    # Restore config
    setattr(CONFIG, sweep_param, original_value)

    # Summary
    accuracies = [successes_per_value[v] / float(s) for v in sweep_values]
    print("\n[summary] Accuracy per sweep value:")
    for v, acc in zip(sweep_values, accuracies):
        print(f"  {sweep_param}={v} → accuracy={acc:.3f}")

    # Plot accuracy vs param
    fig = plot_accuracy_vs_param(sweep_param, sweep_values, accuracies)
    fig.savefig(out_dir / f"accuracy_vs_{sweep_param}.png"); plt.close(fig)

    print(f"Done. Results in {out_dir.resolve()}")