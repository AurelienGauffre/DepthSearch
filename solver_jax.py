# solver_jax.py
#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple, List
from functools import partial
import jax
import jax.numpy as jnp

Array = jnp.ndarray

@dataclass
class CandidateResultJax:
    theta: float
    x0: float
    y0: float
    mse: float
    b_est: float              # = 0.0 (no fit_bias)
    xs: list                  # Python lists for plotting compatibility
    ys: list
    ys_int: list
    xs_int: list
    h_est: list               # predicted profile (without bias)

# ---------- Vectorized primitives ----------
@partial(jax.jit, static_argnames=('n',))  # avoid classical ConcretizationTypeError
def _line_samples_batch(x0: Array, y0: Array, theta: float, step: float, n: int) -> Tuple[Array, Array]:
    """x0,y0: (C,) -> xs, ys: (C,n)"""
    ks = jnp.arange(n, dtype=jnp.float32)          # (n,)
    dx, dy = jnp.cos(theta), jnp.sin(theta)
    xs = x0[:, None] + ks[None, :] * step * dx
    ys = y0[:, None] + ks[None, :] * step * dy
    return xs, ys

def _sample_bilinear_batch(depth_map: Array, xs: Array, ys: Array) -> Array:
    """depth_map: (H,W); xs, ys: (C,n) -> depths: (C,n)"""
    H, W = depth_map.shape
    x0 = jnp.floor(xs).astype(jnp.int32)
    y0 = jnp.floor(ys).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1)
    y1 = jnp.minimum(y0 + 1, H - 1)

    wx = xs - x0.astype(xs.dtype)
    wy = ys - y0.astype(ys.dtype)

    Ia = depth_map[y0, x0]
    Ib = depth_map[y0, x1]
    Ic = depth_map[y1, x0]
    Id = depth_map[y1, x1]

    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

@partial(jax.jit, static_argnames=('n', 'k'))
def _mse_and_topk_indices(
    depth_map: Array,
    z_meas: Array,             # (n,)
    x0_flat: Array,            # (C,)
    y0_flat: Array,            # (C,)
    theta: float,
    step: float,
    n: int,
    valid_flat: Array,         # (C,) bool
    k: int,
):
    """
    Returns:
        - mse_flat_nan: (C,) with NaN on invalid entries (for the map)
        - topk_idx_sorted: (k,) indices of the top k candidates (sorted by increasing MSE)
    """
    xs, ys = _line_samples_batch(x0_flat, y0_flat, theta, step, n)   # (C,n)
    h = _sample_bilinear_batch(depth_map, xs, ys)                    # (C,n)
    diffs = h - z_meas[None, :]
    mse_flat = jnp.mean(diffs * diffs, axis=1)                       # (C,)

    # For the map: NaN on invalid entries
    mse_flat_nan = jnp.where(valid_flat, mse_flat, jnp.nan)

    # For selection: +inf on invalid entries -> excluded from top-k
    mse_for_sel = jnp.where(valid_flat, mse_flat, jnp.inf)

    # IMPORTANT: ne pas modifier k avec jnp.* ici (k est statique & hashable)
    # On suppose que k <= C (garanti côté host).
    part_idx = jnp.argpartition(mse_for_sel, k - 1)[:k]              # (k,)
    # final sort of these k indices
    part_vals = mse_for_sel[part_idx]
    order = jnp.argsort(part_vals)
    topk_idx_sorted = part_idx[order]                                # (k,)

    return mse_flat_nan, topk_idx_sorted

# ---------- Main Function ----------
def solverGridSearchJax(
    depth_map: Array,
    z_meas: Array,                # (n,)
    step: float,
    n: int,
    theta: float,
    x0_grid: Array,               # (Nx,)
    y0_grid: Array,               # (Ny,)
    valid_grid_mask: Optional[Array],  # (Ny,Nx) bool or None
    k: int
) -> Tuple[List[CandidateResultJax], Array]:
    """
    JAX grid-search + Top-K (no fit_bias, bilinear).
    Returns (topk_list, mse_map(Ny,Nx)).

    Notes:
    - All JAX tensors should be float32/bool (consistent).
    - If valid_grid_mask is None -> all valid.
    """
    Ny = int(y0_grid.shape[0])
    Nx = int(x0_grid.shape[0])

    # Candidate mesh
    X0, Y0 = jnp.meshgrid(x0_grid, y0_grid, indexing="xy")  # (Ny,Nx)
    x0_flat = X0.reshape(-1).astype(jnp.float32)            # (C,)
    y0_flat = Y0.reshape(-1).astype(jnp.float32)            # (C,)

    if valid_grid_mask is None:
        valid_flat = jnp.ones((Ny * Nx,), dtype=jnp.bool_)
    else:
        valid_flat = valid_grid_mask.reshape(-1).astype(jnp.bool_)

    # Clamp k côté host (indispensable pour éviter d'opérer sur k sous JAX)
    C = int(x0_flat.size)
    k_eff = int(min(int(k), C))

    # Compile + run: MSE pour la carte + indices top-k
    mse_flat_nan, topk_idx = _mse_and_topk_indices(
        depth_map=jnp.asarray(depth_map, dtype=jnp.float32),
        z_meas=jnp.asarray(z_meas, dtype=jnp.float32),
        x0_flat=x0_flat,
        y0_flat=y0_flat,
        theta=jnp.asarray(theta, dtype=jnp.float32),
        step=jnp.asarray(step, dtype=jnp.float32),
        n=int(n),
        valid_flat=valid_flat,
        k=int(k_eff),
    )

    # Map (Ny,Nx)
    mse_map = mse_flat_nan.reshape((Ny, Nx))

    # ---------- Retrieve details for the Top-K ----------
    xs_top, ys_top = _line_samples_batch(
        x0_flat[topk_idx], y0_flat[topk_idx], theta, step, n
    )  # (k,n)
    h_top = _sample_bilinear_batch(jnp.asarray(depth_map, dtype=jnp.float32), xs_top, ys_top)  # (k,n)
    mse_top = mse_flat_nan[topk_idx]                                                            # (k,)

    # Integer indices (for overlay / plotting)
    ys_int = jnp.clip(jnp.rint(ys_top), 0, depth_map.shape[0] - 1).astype(jnp.int32)           # (k,n)
    xs_int = jnp.clip(jnp.rint(xs_top), 0, depth_map.shape[1] - 1).astype(jnp.int32)           # (k,n)

    # Convert to Python for compatibility with your plot_* (attributes .x0 etc.)
    topk_list: List[CandidateResultJax] = []
    x0_sel = x0_flat[topk_idx]
    y0_sel = y0_flat[topk_idx]

    # bring to host (Python lists) pour construire les objets Python
    x0_list = x0_sel.tolist()
    y0_list = y0_sel.tolist()
    mse_list = mse_top.tolist()
    xs_list = xs_top.tolist()
    ys_list = ys_top.tolist()
    xs_int_list = xs_int.tolist()
    ys_int_list = ys_int.tolist()
    h_list = h_top.tolist()

    for i in range(len(x0_list)):
        topk_list.append(
            CandidateResultJax(
                theta=float(theta),
                x0=float(x0_list[i]),
                y0=float(y0_list[i]),
                mse=float(mse_list[i]),
                b_est=0.0,
                xs=list(xs_list[i]),
                ys=list(ys_list[i]),
                ys_int=list(ys_int_list[i]),
                xs_int=list(xs_int_list[i]),
                h_est=list(h_list[i]),
            )
        )

    return topk_list, mse_map
