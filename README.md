# RadAI: Physics-Informed Neural Networks for Stellar Atmosphere Radiative Transfer
## O# O5083

### Summary

This code trains a physics-informed neural network (PINN) to learn the steady-state, one-dimensional radiative transfer equation. It is intended for astrophysical applications to stellar atmospheres, using realistic opacities, bulk densities, and distances. Right now, it solves a simple proof-of-concept problem to establish that PINNs are a viable step toward emulating 3D stellar atmosphere spectra.

### Usage Instructions

The code currently uses the physical parameters described in the `sim_data/*.fits` files corresponding to a Solar-like stellar atmosphere. These are loaded in `utils.py` and are currently hard-coded; support for flexible stellar atmosphere inputs will be added in the future.

To train the PINN and generate diagnostic plots, run the Python script `run_RadAI.py`. This will generate a default `runs/RadAI` directory containing the `I_vs_lambda.pdf` plot showing the spectrum at various distances throughout the atmosphere compared to an analytic solution, with comparison to a spectrum calculated by the Korg 1-D radiative transfer code written in Julia.
