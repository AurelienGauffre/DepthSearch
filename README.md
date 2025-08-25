# ⛵Depth-Only Localization
**Can you locate a boat precisely without GPS, using only a few measurements from an onboard depth sensor?**

![all](https://github.com/user-attachments/assets/9453d613-16bf-4021-bbeb-de5775d27d11)



This project explores that idea: given a real bathymetric map and a short sequence the depth sensors readings, we attempt to recover the boat’s position by matching depth profiles on high resolution depths maps. After demonstrating the feasibility of this approach with a realistic modeling of the different errors, the next objective is to develop efficient methods to achieve real-time localization.

## 🧪 First Step
This first phase focuses on feasibility and intuition:
- Is the solution identifiable using only a few depth values, given the current resolution of best available bathymetric maps.
- How robust is the approach under realistic noise and biases?
- How does accuracy evolve with trajectory length and sampling step?

## 🎲 Error Modeling
- Bathymetric map precision (grid resolution and accuracy)
- Depth sensor measurement errors (noise, bias, and limited resolution)
- Tide level offsets (tide maps spatio-temporal accuracy) 
- Boat trajectory (currently assume a straight line with a known $\theta$ angle)

## Quickstart (uv)
```powershell
uv sync # install deps
uv run python .\main.py
```
(Install uv if needed: https://docs.astral.sh/uv/)

## Project layout
- configs.py: all parameters of the simulation
- dataloading.py: load raster/synthetic bathymetry and mask
- trajectory.py: sampling utilities and noisy trajectory generation
- solver.py: profile building and grid search
- plotting.py: figures
- main.py: run simulation

## Notes
- Raster CRS is assumed WGS84 for simplicity when estimating resolution.
- `data/` should contain a *.grd file for raster mode, otherwise uses synthetic data,
you can download a high resolution bathymetric map of the Atlantic coast and more from SHOM Diffusion [here](https://diffusion.shom.fr/multiproduct/product/configure/id/180) with a free account.
