#!/usr/bin/env python3
"""
Configuration file for depth-only localization demo.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Any

@dataclass
class SimConfig:
    # --- Data source ---
    data_source: str = "raster"   # "synthetic" or "raster"
    data_dir: str = "data"        # folder with *.grd if data_source="raster"
    
    # True trajectory (straight line)
    true_step_px: float = 15.0  # how many pixels the boat moves between sampling
    n_samples: int = 10
    sonar_noise_std: float = 1.2
    bilinear: bool = True

    #TODO check that according to Nyquist–Shannon theorem, a value of true_step_px under 0.5 
    # should not help to gain precision ? Maybe not true because of bilinear interpolation ?

    # Solver grid
    grid_stride_px: int = 5
    margin_px: int = 10
    topk: int = 5
    verbose: bool = True

    # Bias handling
    fit_bias: bool = False #TODO confirm that fit_bias=True decreases accuracy
    bias_ridge_lambda: float = 0.0

    # Output
    out_dir: str = "figs"
    save_png: bool = True

    # Synthetic map (used when data_source="synthetic")
    grid_height: int = 1000
    grid_width: int = 1000
    resolution_m: float = 10.0
    max_depth: float = 120.0
    seed: int = 42
    num_waves: int = 12
    min_cycles: float = 1.0
    max_cycles: float = 12.0
    amplitude_range: Tuple[float, float] = (4.0, 35.0)
    gaussian_smooth_iters: int = 1
    small_noise_std: float = 0.8

     # --- Optional cropping for raster mode ---
    crop_enable: bool = True
    crop_by: str = "px"           # "px" or "crs"
    crop_x1: float = 1000 #3000
    crop_x2: float = 4000
    crop_y1: float = 3500 #3500
    crop_y2: float = 5500 #5500


    # Batch / Sweep controls
    sweep_param: str = "n_samples"
    sweep_values: List[Any] = field(default_factory=lambda: [5,10,15,20,25])
    s: int = 10  # number of scenarios


# Default global CONFIG instance
CONFIG = SimConfig()
