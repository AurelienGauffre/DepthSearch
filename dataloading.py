#!/usr/bin/env python3
"""
Data loading module for depth-only localization demo.
Returns a BathyDataset
"""

import math
import numpy as np
from pathlib import Path
from typing import Optional, Any, Tuple
from dataclasses import dataclass

from configs import CONFIG, SimConfig

# Optional import for raster (non synthetic) map loading
try:
    import rasterio
    from rasterio.windows import Window, from_bounds
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
except Exception:
    rasterio = None


@dataclass
class BathyDataset:
    depth: np.ndarray            # (H, W) float32
    mask: np.ndarray             # (H, W) bool  (True = water/valid)
    transform: Any               # raster transform or None
    crs: Any                     # raster CRS or None
    resolution_m: float          # meters/pixel (approx or from CRS)
# ---- Raster loading ----
def _estimate_resolution_m(transform, crs, H: int, W: int, fallback: float) -> float:
    """Estimate resolution in meters per pixel.
    For projected CRS: uses transform.a directly
    For geographic CRS: averages x and y resolution in meters
    """
    try:
        px_w = abs(transform.a)  # deg/px in x
        px_h = abs(transform.e)  # deg/px in y (generally negative value)
        
        if crs and getattr(crs, "is_projected", False):
            # For projected CRS, return pixel width directly (assuming square pixels)
            return float(px_w)
        
        # For geographic CRS, convert both axes to meters and average
        _, lat0 = transform * (0, 0)
        _, lat1 = transform * (W, H)
        lat_mid = (lat0 + lat1) / 2.0
        
        # Convert degrees to meters
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_mid))
        m_per_deg_lat = 110540.0
        
        # Resolution in meters for each axis
        rx = px_w * m_per_deg_lon
        ry = px_h * m_per_deg_lat
        
        return float((rx + ry) * 0.5)
    except Exception:
        return fallback

def _apply_window_and_read(ds, win: Optional[Any]):
    if win is None:
        arr = ds.read(1).astype(np.float32)
        transform = ds.transform
    else:
        arr = ds.read(1, window=win).astype(np.float32)
        transform = ds.window_transform(win)
    return arr, transform

def _load_first_bag_as_array(data_dir: str):
    if rasterio is None:
        raise RuntimeError("rasterio is required for data_source='raster' but is not installed.")
    data_path = Path(data_dir)
    bag_files = sorted(data_path.glob("*.bag"))
    if not bag_files:
        raise FileNotFoundError(f"No .bag file found in {data_dir}")
    f = bag_files[0]
    print(f"[raster] Loading: {f.name}")
    
    with rasterio.open(f) as ds:
        # Load the full dataset first (no cropping yet)
        arr = ds.read(1).astype(np.float32)
        src_transform = ds.transform
        src_crs = ds.crs
        nodata = ds.nodata
        
        print(f"[raster] Taille originale: {ds.height} x {ds.width}")
        
        # Define target CRS as Lambert-93 (EPSG:2154) - official projection for France
        dst_crs = CRS.from_epsg(2154)
        print(f"[raster] Reprojection: {src_crs} -> {dst_crs}")
        
        # Calculate transform and dimensions for the reprojected data (full extent)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, ds.width, ds.height, 
            *ds.bounds  # Use full bounds of the dataset
        )
        
        # Create destination array for full reprojection with NaN as default
        dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        
        # Reproject the full bathymetry data with proper nodata handling
        reproject(
            source=arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=np.nan,
        )
        
        print(f"[raster] Size after full reprojection: {dst_height} x {dst_width}")
        
        # Now crop in the projected coordinate system if needed
        if CONFIG.crop_enable:
            x1, y1, x2, y2 = CONFIG.crop_x1, CONFIG.crop_y1, CONFIG.crop_x2, CONFIG.crop_y2
            xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
            ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
            
            if CONFIG.crop_by.lower() == "px":
                # Crop by pixel coordinates in the projected space
                xmin_i = int(max(0, min(dst_width, math.floor(xmin))))
                xmax_i = int(max(0, min(dst_width, math.ceil(xmax))))
                ymin_i = int(max(0, min(dst_height, math.floor(ymin))))
                ymax_i = int(max(0, min(dst_height, math.ceil(ymax))))
                
                w = max(0, xmax_i - xmin_i)
                h = max(0, ymax_i - ymin_i)
                if w == 0 or h == 0:
                    raise ValueError("Crop window in pixels is empty after clamping.")
                
                # Crop the reprojected array
                dst_arr = dst_arr[ymin_i:ymax_i, xmin_i:xmax_i]
                
                # Update the transform for the cropped area using rasterio API
                win = Window(xmin_i, ymin_i, w, h)
                dst_transform = rasterio.windows.transform(win, dst_transform)
                
                print(f"[raster] Crop in projected space (px): x=[{xmin_i},{xmax_i}), y=[{ymin_i},{ymax_i}) -> {w}×{h}")
                
            elif CONFIG.crop_by.lower() == "crs":
                # For CRS-based cropping, we need to convert the crop bounds to the projected CRS first
                # This is more complex and would require coordinate transformation
                print("[raster] WARNING: CRS-based crop after reprojection not yet implemented. Using pixels.")
                # For now, fall back to no cropping when using CRS bounds
            else:
                raise ValueError("CONFIG.crop_by must be 'px' or 'crs'.")

    # Clean NaN / NoData / extremes → mask
    # mask: True = water/valid
    mask = np.isfinite(dst_arr)
    if nodata is not None:
        mask &= (dst_arr != nodata)
    extreme = (dst_arr < -12000) | (dst_arr > 10000)
    mask &= ~extreme

    # Fill invalid with median for robust sampling
    med = float(np.nanmedian(dst_arr[mask])) if mask.any() else 0.0
    arr_clean = dst_arr.copy()
    arr_clean[~mask] = med

    return arr_clean.astype(np.float32), mask.astype(bool), dst_transform, dst_crs


def _load_first_bag_original(data_dir: str):
    """Charge le fichier .bag sans reprojection pour comparaison."""
    if rasterio is None:
        raise RuntimeError("rasterio is required for data_source='raster' but is not installed.")
    data_path = Path(data_dir)
    bag_files = sorted(data_path.glob("*.bag"))
    if not bag_files:
        raise FileNotFoundError(f"No .bag file found in {data_dir}")
    f = bag_files[0]
    print(f"[raster original] Loading: {f.name}")
    with rasterio.open(f) as ds:
        win = None
        if CONFIG.crop_enable:
            x1, y1, x2, y2 = CONFIG.crop_x1, CONFIG.crop_y1, CONFIG.crop_x2, CONFIG.crop_y2
            xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
            ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
            if CONFIG.crop_by.lower() == "px":
                xmin_i = int(max(0, min(ds.width, math.floor(xmin))))
                xmax_i = int(max(0, min(ds.width, math.ceil(xmax))))
                ymin_i = int(max(0, min(ds.height, math.floor(ymin))))
                ymax_i = int(max(0, min(ds.height, math.ceil(ymax))))
                w = max(0, xmax_i - xmin_i); h = max(0, ymax_i - ymin_i)
                if w == 0 or h == 0:
                    raise ValueError("Crop window in pixels is empty after clamping.")
                win = Window(col_off=xmin_i, row_off=ymin_i, width=w, height=h)
                print(f"[raster original] Crop (px): x=[{xmin_i},{xmax_i}), y=[{ymin_i},{ymax_i}) -> {w}×{h}")
            elif CONFIG.crop_by.lower() == "crs":
                win = from_bounds(left=xmin, bottom=ymin, right=xmax, top=ymax, transform=ds.transform)
                print(f"[raster original] Crop (CRS): [{xmin},{ymin},{xmax},{ymax}]")
            else:
                raise ValueError("CONFIG.crop_by must be 'px' or 'crs'.")
        arr, transform = _apply_window_and_read(ds, win)
        # Read CRS directly from the dataset
        src_crs = ds.crs
        nodata = ds.nodata

    # Clean NaN / NoData / extremes → mask (same as projected version)
    mask = np.ones_like(arr, dtype=bool)
    if np.isnan(arr).any():
        mask &= ~np.isnan(arr)
    if nodata is not None:
        mask &= (arr != nodata)
    extreme = (arr < -12000) | (arr > 10000)
    mask &= ~extreme

    # Fill invalid with median for robust sampling
    med = float(np.median(arr[mask])) if mask.any() else 0.0
    arr_clean = arr.copy()
    arr_clean[~mask] = med

    return arr_clean.astype(np.float32), mask.astype(bool), transform, src_crs


# ---- Public interface ----
def load_bathy(config: SimConfig = CONFIG) -> BathyDataset:
    """
    Build/load bathymetric dataset with its validity mask.
    """
    if config.data_source.lower() == "synthetic":
        depth_map = generate_synthetic_bathymetry(config)
        mask = np.ones_like(depth_map, dtype=bool)
        transform = None
        crs = None
        res_m = float(config.resolution_m)

    elif config.data_source.lower() == "raster":
        arr, mask, transform, crs = _load_first_bag_as_array(config.data_dir)
        H, W = arr.shape
        config.grid_height = int(H)
        config.grid_width = int(W)
        res_m = _estimate_resolution_m(transform, crs, H, W, fallback=config.resolution_m)

    else:
        raise ValueError("CONFIG.data_source must be 'synthetic' or 'raster'.")

    return BathyDataset(depth=arr if config.data_source=='raster' else depth_map,
                        mask=mask,
                        transform=transform,
                        crs=crs,
                        resolution_m=res_m)


def pixel_to_xy(ds: BathyDataset, col: float, row: float) -> Tuple[float, float]:
    """Convert a pixel (col, row) to coordinates (x, y) in the dataset's CRS."""
    if ds.transform is None:
        raise RuntimeError("Transform missing to convert to coordinates.")
    x, y = ds.transform * (col, row)
    return float(x), float(y)





# ----- Synthetic generation ----
def _smooth_around(z: np.ndarray) -> np.ndarray:
    return (z
            + np.roll(z, 1, axis=0) + np.roll(z, -1, axis=0)
            + np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1)) / 5.0

def generate_synthetic_bathymetry(config: SimConfig = CONFIG) -> np.ndarray:
    H, W = config.grid_height, config.grid_width
    rng = np.random.default_rng(config.seed)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    X = np.broadcast_to(x, (H, W))
    Y = np.broadcast_to(y, (H, W))
    depth = np.zeros((H, W), dtype=np.float32)

    for _ in range(config.num_waves):
        phi = rng.uniform(0.0, np.pi)
        cycles = rng.uniform(config.min_cycles, config.max_cycles)
        U = X * np.cos(phi) + Y * np.sin(phi)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(*config.amplitude_range)
        depth += (amp * np.sin(2.0 * np.pi * cycles * U + phase)).astype(np.float32)

    if config.small_noise_std > 0:
        depth += rng.normal(0.0, config.small_noise_std, size=(H, W)).astype(np.float32)

    for _ in range(max(0, config.gaussian_smooth_iters)):
        depth = _smooth_around(depth)

    depth -= depth.min()
    if depth.max() > 0:
        depth = depth / depth.max()

    return (depth * config.max_depth).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotting import viridis_custom
    



    #### Test of map loading and projection


    CONFIG.crop_enable = False # Optional, to plot full map
    
    # Load complete reprojected data and save
    ds_projected = load_bathy()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(ds_projected.depth, cmap=viridis_custom, aspect='equal', vmin=-200, vmax=0, origin="upper")
    plt.savefig("full_projected_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Full map saved: full_projected_map.png")