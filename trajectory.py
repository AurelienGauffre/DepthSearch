#!/usr/bin/env python3
"""
Trajectory utilities: line sampling, bounds checks, sampling methods,
mask checks, and trajectory generation with noise.
"""

import math
from typing import Optional, Tuple, Any
import rasterio
import numpy as np

from rasterio.warp import transform as rasterio_transform
from rasterio.crs import CRS


def line_samples(x0: float, y0: float, theta: float, step: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    ks = np.arange(n, dtype=float)
    dx, dy = math.cos(theta), math.sin(theta)
    xs = x0 + ks * step * dx
    ys = y0 + ks * step * dy
    return xs, ys


def in_bounds(xs: np.ndarray, ys: np.ndarray, H: int, W: int, margin: float = 0.0) -> bool:
    return (
        np.all(ys >= margin) and np.all(ys <= H - 1 - margin) and
        np.all(xs >= margin) and np.all(xs <= W - 1 - margin)
    )


def sample_nearest(depth_map: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    ys_int = np.rint(ys).astype(np.int32)
    xs_int = np.rint(xs).astype(np.int32)
    depths = depth_map[ys_int, xs_int]
    return depths, ys_int, xs_int


def sample_bilinear(depth_map: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, W = depth_map.shape
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = xs - x0; wy = ys - y0
    Ia = depth_map[y0, x0]; Ib = depth_map[y0, x1]
    Ic = depth_map[y1, x0]; Id = depth_map[y1, x1]
    wa = (1 - wx) * (1 - wy); wb = wx * (1 - wy)
    wc = (1 - wx) * wy;      wd = wx * wy
    return (Ia * wa + Ib * wb + Ic * wc + Id * wd)


def mask_along_line(valid_mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, bilinear: bool = True) -> bool:
    H, W = valid_mask.shape
    margin = 1.0 if bilinear else 0.0
    if not in_bounds(xs, ys, H, W, margin=margin):
        return False
    ys_int = np.rint(ys).astype(np.int32)
    xs_int = np.rint(xs).astype(np.int32)
    return bool(np.all(valid_mask[ys_int, xs_int]))


def pick_random_valid_scenario(depth_map: np.ndarray, valid_mask: np.ndarray,
                               step_px_worst: float, n_worst: int, rng: np.random.Generator):
    """Draw a random valid (x0, y0, theta) such that the straight line
    stays within bounds and over water for the given step and length.
    """
    H, W = depth_map.shape

    # Draw among water pixels, preferably inside a safe inner margin
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise RuntimeError(f"valid_mask has wrong shape: {mask.shape}")
    # Safe radius so the line of length (n-1)*step remains in-bounds for any theta
    reach = max(0.0, (float(n_worst) - 1.0) * float(step_px_worst) + 1.0)
    xi_min = int(np.ceil(reach)); yi_min = int(np.ceil(reach))
    xi_max = int(np.floor(W - 1 - reach)); yi_max = int(np.floor(H - 1 - reach))
    inner = np.zeros_like(mask, dtype=bool)
    if xi_max >= xi_min and yi_max >= yi_min:
        inner[yi_min:yi_max+1, xi_min:xi_max+1] = True
    inner_mask = mask & inner
    # If inner region is empty, fallback to full water mask (will rely on theta check)
    cand_mask = inner_mask if np.any(inner_mask) else mask
    vy, vx = np.where(cand_mask)
    if len(vx) == 0:
        raise RuntimeError("No valid water pixels found in mask.")

    for _ in range(20000):
        idx = rng.integers(0, len(vx))
        x0, y0 = float(vx[idx]), float(vy[idx])
        theta = rng.uniform(0.0, math.pi)

        xs, ys = line_samples(x0, y0, theta, step_px_worst, n_worst)
        if in_bounds(xs, ys, H, W, margin=1.0) and mask_along_line(mask, xs, ys, bilinear=True):
            return x0, y0, theta

    raise RuntimeError("No valid scenario found (after 20000 tries).")


class Trajectory:
    def __init__(self, xs, ys, ys_int, xs_int, depths_true, depths_noisy):
        self.xs = xs; self.ys = ys
        self.ys_int = ys_int; self.xs_int = xs_int
        self.depths_true = depths_true; self.depths_noisy = depths_noisy


def make_straight_trajectory(depth_map: np.ndarray,
                             valid_mask: np.ndarray,
                             x0: float, y0: float, theta: float, step: float, n: int,
                             noise_std: float = 1.0, noise_seed: Optional[int] = None,
                             bilinear: bool = True) -> Trajectory:
    """Sample a straight trajectory with n depth samples with noise."""
    xs, ys = line_samples(x0, y0, theta, step, n)
    H, W = depth_map.shape
    if not in_bounds(xs, ys, H, W, margin=1.0 if bilinear else 0.0):
        raise ValueError("Line out of bounds.")
    if valid_mask is not None and not mask_along_line(valid_mask, xs, ys, bilinear=bilinear):
        raise ValueError("Line crosses invalid (land) cells.")
    if bilinear:
        depths_true = sample_bilinear(depth_map, xs, ys)
        _, ys_int, xs_int = sample_nearest(depth_map, xs, ys)
    else:
        depths_true, ys_int, xs_int = sample_nearest(depth_map, xs, ys)
    if noise_std > 0:
        rng = np.random.default_rng(noise_seed if noise_seed is not None else 0)
        eps = rng.normal(0.0, noise_std, size=n) #TODO more realistic sonar noise modeling
        depths_noisy = (depths_true + eps)
    else:
        depths_noisy = depths_true
    return Trajectory(xs, ys,
                      ys_int, xs_int,
                      depths_true,
                      depths_noisy)


def compute_valid_grid_mask(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    theta: float,
    x0_grid: np.ndarray,
    y0_grid: np.ndarray,
    step: float,
    n: int,
    bilinear: bool = True,
):
    """Return a boolean mask of valid (x0,y0) grid starts for a given theta.

    A start is valid if the entire line of length n at step spacing stays in bounds
    (with margin depending on bilinear) and within the valid_mask (water-only),
    when applicable.
    """
    Hy, Hx = len(y0_grid), len(x0_grid)
    mask = np.zeros((Hy, Hx), dtype=bool)
    H, W = depth_map.shape
    for yi, y0 in enumerate(y0_grid):
        for xi, x0 in enumerate(x0_grid):
            xs, ys = line_samples(float(x0), float(y0), theta, step, n)
            if not in_bounds(xs, ys, H, W, margin=1.0 if bilinear else 0.0):
                continue
            if valid_mask is not None and not mask_along_line(
                valid_mask, xs, ys, bilinear=bilinear
            ):
                continue
            mask[yi, xi] = True
    return mask


def pixel_to_latlon(transform: Any, crs: Any, col: float, row: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to lat/lon (WGS84).
    """
    if transform is None or crs is None:
        raise RuntimeError("Transform or CRS is None")
    
    # Convert pixel to projected coordinates
    x, y = transform * (col, row)
    # Convert to lat/lon (WGS84)
    dst_crs = CRS.from_epsg(4326)
    lon, lat = rasterio_transform(crs, dst_crs, [x], [y])
    
    return float(lat[0]), float(lon[0])


def latlon_to_pixel(transform: Any, crs: Any, lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert lat/lon (WGS84) to pixel coordinates.
    """
    # Convert lat/lon to projected coordinates
    src_crs = CRS.from_epsg(4326)
    x, y = rasterio_transform(src_crs, crs, [lon], [lat])
    
    # Convert projected coordinates to pixels
    inv_transform = ~transform
    col, row = inv_transform * (x[0], y[0])
    
    return float(col), float(row)
