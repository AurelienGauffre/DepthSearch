#!/usr/bin/env python3
"""
Solver utilities: candidate struct, profile sampling along a line,
grid search with optional bias fitting, and helper masks.
"""

from typing import Optional, Tuple, List
import heapq
import numpy as np

from trajectory import line_samples, sample_bilinear, sample_nearest, in_bounds, mask_along_line


class CandidateResult:
    def __init__(self, theta, x0, y0, mse, b_est, xs, ys, ys_int, xs_int, h_est):
        self.theta = theta; self.x0 = x0; self.y0 = y0
        self.mse = mse; self.b_est = b_est
        self.xs = xs; self.ys = ys
        self.ys_int = ys_int; self.xs_int = xs_int
        self.h_est = h_est


def fit_bias_closed_form(z_meas: np.ndarray, h_line: np.ndarray, lam: float = 0.0) -> float:
    diff_sum = float(np.sum(z_meas - h_line))
    N = float(len(z_meas))
    return diff_sum / (N + lam)


def profile_along_line(depth_map: np.ndarray, valid_mask: np.ndarray,
                       x0: float, y0: float, theta: float,
                       step: float, n: int, bilinear: bool = True):
    xs, ys = line_samples(x0, y0, theta, step, n)
    H, W = depth_map.shape
    margin = 1.0 if bilinear else 0.0
    if not in_bounds(xs, ys, H, W, margin=margin):
        raise ValueError("Out of bounds")
    if valid_mask is not None and not mask_along_line(valid_mask, xs, ys, bilinear=bilinear):
        raise ValueError("Invalid candidate: crosses land")
    if bilinear:
        h = sample_bilinear(depth_map, xs, ys)
        _, ys_int, xs_int = sample_nearest(depth_map, xs, ys)
    else:
        h, ys_int, xs_int = sample_nearest(depth_map, xs, ys)
    return h, xs, ys, ys_int, xs_int


def solverGridSearch(depth_map: np.ndarray, valid_mask: np.ndarray,
                       z_meas: np.ndarray, step: float, n: int, theta: float,
                       x0_grid: np.ndarray, y0_grid: np.ndarray,
                       bilinear: bool = True, topk: int = 5,
                       fit_bias: bool = False, bias_ridge_lambda: float = 0.0,
                       valid_grid_mask: Optional[np.ndarray] = None):
    heap: List[Tuple[float, int, CandidateResult]] = []
    mse_map = np.full((len(y0_grid), len(x0_grid)), np.nan)
    counter = 0
    for yi, y0 in enumerate(y0_grid):
        for xi, x0 in enumerate(x0_grid):
            if valid_grid_mask is not None and not valid_grid_mask[yi, xi]:
                continue
            try:
                h, xs, ys, ys_int, xs_int = profile_along_line(
                    depth_map, valid_mask, float(x0), float(y0), float(theta), step, n, bilinear
                )
            except ValueError:
                continue
            if fit_bias:
                b = fit_bias_closed_form(z_meas, h, lam=bias_ridge_lambda)
                h_est = h + b
            else:
                b = 0.0; h_est = h
            mse = float(np.mean((z_meas - h_est) ** 2))
            mse_map[yi, xi] = mse
            cand = CandidateResult(theta, float(x0), float(y0), mse, b,
                                   xs, ys, ys_int, xs_int, h_est)
            if len(heap) < topk:
                heapq.heappush(heap, (-mse, counter, cand))
            else:
                if mse < -heap[0][0]:
                    heapq.heapreplace(heap, (-mse, counter, cand))
            counter += 1
    heap.sort(key=lambda t: -t[0])
    return [t[2] for t in heap], mse_map


    
