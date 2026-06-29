#© 2026. Triad National Security, LLC. All rights reserved.
#This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

import torch
import numpy as np
import matplotlib, dill, glob, sys
import matplotlib.pyplot as plt
from astropy import units as unit
from astropy import constants
from astropy.io import fits
from scipy.interpolate import CubicSpline as cs
from scipy.interpolate import RectBivariateSpline
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson
from scipy.signal import savgol_filter
import sys
import os
from utils import load_korg_data
from datetime import datetime
from pathlib import Path

def to_numpy(x):
    """Convert tensor to numpy"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def nbins(r_min, r_max, tau_0, tau_factor=1):
    """Estimate number of bins for integration"""
    return int(max(100, tau_0 * tau_factor * 10))

class RadAI():

    def __init__(self, r_norm='log', r_max_factor=1,
                 test_name='RadAI', lr=1e-3, n_hidden=512, n_layers=6,
                 mu_min=0.05, mu_max=1.0, n_collocation=1024, n_boundary=100):

        self.r_norm = r_norm
        self.test_name = test_name
        self.r_max_factor = float(r_max_factor)
        self.lr = float(lr)
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.n_collocation = int(n_collocation)
        self.n_boundary = int(n_boundary)

        # Validate mu range
        assert 0 < self.mu_min < self.mu_max <= 1.0, "mu_min must be > 0 and mu_max must be <= 1.0 with mu_min < mu_max"

        print(self.r_norm, self.r_max_factor)
        print(f'mu range: [{self.mu_min}, {self.mu_max}]')

        # Create output directory
        output_dir = Path(f'runs/{self.test_name}')
        output_dir.mkdir(parents=True, exist_ok=True)

        # PyTorch settings

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(torch.float32)
        print(f"Using device: {self.device}")

        # Define constants
                
        self.h = 6.626e-27 # cm^2 g s^-1
        self.c = 2.9979e10 # cm/s
        self.k = 1.3807e-16 # cm^2 g s^-2 K^-1

        # Load Korg simulation data and relevant quantities

        #self.korg_wav, z, temp, korg_rho, korg_kappa_array = load_multi_wavelength_data()
        self.korg_wav, self.korg_flux, z, temp, korg_rho, korg_kappa_array, tau_5000 = load_korg_data()

        z_min = z.min()
        z_max = z.max()

         #-- Get R_MIN, R_MAX from Korg simulation z (scale height)
        self.R_MIN, self.R_MAX = torch.tensor(z.min(), dtype=torch.float32), torch.tensor(z.max(), dtype=torch.float32)

        if (self.r_norm == 'minmax'):
            self.R_0 = self.R_MAX - self.R_MIN
        elif self.r_norm == 'log': 
            self.R_0 = torch.tensor(1) # no normalization
 
        #-- Define r_hat range (normalized between 0 and 1 by definition), and I_hat_0

        self.r_hat_min = self.r_hat_func(self.R_MIN)
        self.r_hat_max = self.r_hat_func(self.R_MAX/self.r_max_factor)
        print(f'r_hat_min: {self.r_hat_min}, r_hat_max:{self.r_hat_max}, R_MIN: {self.r_from_rhat(self.r_hat_min)}, R_MAX: {self.r_from_rhat(self.r_hat_max)}')
       
        #-- Wavelength range in physical units (Angstroms)
        self.LAMBDA_MIN = self.korg_wav.min()  
        self.LAMBDA_MAX = self.korg_wav.max()  

        #-- Define normalization constants
        r_sample = torch.linspace(self.r_hat_min, self.r_hat_max, 1000)
        wav_sample = torch.linspace(self.LAMBDA_MIN, self.LAMBDA_MAX, 1000)
        self.wav = np.linspace(to_numpy(self.LAMBDA_MIN), to_numpy(self.LAMBDA_MAX), 70)
        self.wav_nn = np.linspace(to_numpy(self.LAMBDA_MIN), to_numpy(self.LAMBDA_MAX), 7001)

        self.KAPPA_MIN = torch.tensor(korg_kappa_array.min(), dtype=torch.float32)
        self.KAPPA_MAX = torch.tensor(korg_kappa_array.max(), dtype=torch.float32)
        self.korg_kappa_array = torch.tensor(korg_kappa_array, dtype=torch.float32)

        # Convert to normalized coordinates for interpolation
        # z is still in physical coordinates at this point
        z_norm = self.r_hat_func(z.astype(float), to_np=True)
        self.z = z.copy()  # Keep original physical z

        # Temperature interpolation using scipy CubicSpline with NORMALIZED coordinates
        self.temp_func = cs(z_norm, temp)

        # kappa interpolation using scipy RectBivariateSpline with NORMALIZED coordinates
        korg_wav_np = to_numpy(self.korg_wav)
        lambda_norm = (korg_wav_np - self.LAMBDA_MIN.item()) / (self.LAMBDA_MAX.item() - self.LAMBDA_MIN.item())
        self.kappa_func = RectBivariateSpline(z_norm, lambda_norm, korg_kappa_array)

        # Density interpolation using scipy CubicSpline with NORMALIZED coordinates
        self.rho_func = cs(z_norm, korg_rho)
        
        self.K_0 = torch.tensor(korg_kappa_array.mean(), dtype=torch.float32, device=self.device)
        self.RHO_0 = torch.tensor(korg_rho.mean(), dtype=torch.float32, device=self.device)

        # Compute S_0 by sampling 20×20 grid
        r_hat_min_val = to_numpy(self.r_hat_min) if torch.is_tensor(self.r_hat_min) else float(self.r_hat_min)
        r_hat_max_val = to_numpy(self.r_hat_max) if torch.is_tensor(self.r_hat_max) else float(self.r_hat_max)
        r_sample = np.linspace(r_hat_min_val, r_hat_max_val, 20)
        lambda_sample = np.linspace(0.0, 1.0, 20)  # lambda_hat range
        S_samples = []
        for r_hat in r_sample:
            for lambda_hat in lambda_sample:
                lam_phys = self.lambda_from_lambda_hat(lambda_hat)
                S_phys = self.S_lam(r_hat, lam_phys)
                if torch.is_tensor(S_phys):
                    S_samples.append(S_phys.item())
                else:
                    S_samples.append(float(S_phys))
        self.S_0 = torch.tensor(np.mean(S_samples), dtype=torch.float32, device=self.device)

        self.tau_0 = (self.RHO_0 * self.K_0 * self.R_0).item()
        print(f'tau_0: {self.tau_0}, RHO_0: {self.RHO_0}, K_0: {self.K_0}, R_0: {self.R_0}, S_0: {self.S_0}')

        # Initial condition function: I(r_min, lambda_hat) = S(r_min, lambda_phys)
        self.I_hat_0 = lambda lambda_hat: self.S_lam_hat(self.r_hat_min, self.lambda_from_lambda_hat(lambda_hat))

        # calculate analytic solution for use during training / plotting

        LAMBDA_MIN, LAMBDA_MAX = to_numpy(self.LAMBDA_MIN), to_numpy(self.LAMBDA_MAX)

        # Analytic blackbody function calculation

        num_bins = nbins(to_numpy(self.R_MIN), to_numpy(self.R_MAX), to_numpy(self.tau_0), tau_factor=1)
        if num_bins < 1000: num_bins = 1000
        num_bins = 100
        self.Rs = np.logspace(np.log10(to_numpy(self.R_MIN)), np.log10(to_numpy(self.R_MAX)), num_bins)
        self.wav_analytic = np.linspace(LAMBDA_MIN, LAMBDA_MAX, len(self.wav))*1e-8 # Angstroms to cm

        slam_start = datetime.now()
        
        S_lam_analytic = np.array([
            (2*self.h*self.c**2/self.wav_analytic[j]**5) * \
             1/(np.exp(self.h*self.c/(self.wav_analytic[j]*self.k*self.T(self.r_hat_func(self.Rs[i], to_np=True)))) - 1)
            for i in range(self.Rs.shape[0])
            for j in range(len(self.wav_analytic))])
        
        self.S_lam_analytic = S_lam_analytic.reshape(len(self.Rs), len(self.wav_analytic))
        self.S_lam_analytic *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A
        self.S_lam_analytic /= to_numpy(self.S_0)
        print('S_lam_analytic calculation took: ', (datetime.now()-slam_start).total_seconds()/60, ' minutes')
       
        taus_start = datetime.now()

        # Convert wav_analytic (in cm) to lambda_hat for kappa call
        wav_analytic_angstrom = self.wav_analytic * 1e8  # cm to Angstrom
        lambda_hat_analytic = (wav_analytic_angstrom - to_numpy(self.LAMBDA_MIN)) / (to_numpy(self.LAMBDA_MAX) - to_numpy(self.LAMBDA_MIN))

        taus = np.array([simpson(to_numpy(self.rho(self.r_hat_func(self.Rs[:i+1], to_np=True)) * self.kappa(self.r_hat_func(self.Rs[:i+1], to_np=True), lambda_hat_analytic[j])), x = self.Rs[:i+1]) for i in range(self.Rs.shape[0]) for j in range(len(self.wav_analytic))])
        taus = taus.reshape(len(self.Rs), len(self.wav_analytic))
 
        print('taus calculation took: ', (datetime.now()-taus_start).total_seconds()/60, ' minutes')
       
        Inu_start = datetime.now()

        # Convert wav_analytic (in Angstroms) to lambda_hat for I_hat_0 call
        self.I_nu = np.array([
                    simpson(self.S_lam_analytic[:i+1, j]*np.exp(-(taus[i, j] - taus[:i+1, j])), x=taus[:i+1, j]) + to_numpy(self.I_hat_0(lambda_hat_analytic[j]))*np.exp(-taus[i, j]) for i in range(self.Rs.shape[0])
                for j in range(len(self.wav_analytic))])
        print('I_nu calculation took: ', (datetime.now()-Inu_start).total_seconds()/60, ' minutes')
       
        self.I_nu = self.I_nu.reshape(len(self.Rs), len(self.wav_analytic))
        # Use scipy RectBivariateSpline for I_nu interpolation
        # RectBivariateSpline(x, y, z) where z has shape (len(x), len(y))
        # I_nu has shape (Rs, wavelengths) so x=Rs, y=wavelengths
        self.I_nu_func = RectBivariateSpline(
            self.r_hat_func(self.Rs, to_np=True),  # r_hat values (x-axis)
            self.wav_analytic*1e8,  # wavelengths in Angstroms (y-axis)
            self.I_nu  # shape: (len(Rs), len(wav_analytic))
        )

    def kappa(self, r_hat, lambda_hat):
        """
        Kappa interpolation with NORMALIZED coordinates (r_hat, lambda_hat).

        Args:
            r_hat: normalized radius in [0, 1]
            lambda_hat: normalized wavelength in [0, 1]
        """
        # Convert inputs to numpy
        if torch.is_tensor(r_hat): r_hat = to_numpy(r_hat)
        if torch.is_tensor(lambda_hat): lambda_hat = to_numpy(lambda_hat)

        kap = self.kappa_func(r_hat, lambda_hat, grid=False)

        return kap

    def rho(self, r):
        if torch.is_tensor(r): r = to_numpy(r)
        return self.rho_func(r)
    
    def T(self, r):
        return self.temp_func(r)

    def S_lam(self, r, wav):
        """
        Compute Planck function B_λ(T) directly (faster than astropy BlackBody).

        B_λ = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
        """
        # Convert inputs to numpy
        if torch.is_tensor(r):
            r = to_numpy(r)
        if torch.is_tensor(wav):
            wav = to_numpy(wav)

        # Handle scalar vs array
        r = np.atleast_1d(r)
        wav = np.atleast_1d(wav)

        # Get temperature
        T = self.T(r)

        # Convert wavelength from Angstroms to cm
        wav_cm = wav * 1e-8

        # Compute Planck function
        # B_λ in erg / s / cm² / cm / sr
        B_lambda = (2 * self.h * self.c**2 / wav_cm**5) / \
                   (np.exp(self.h * self.c / (wav_cm * self.k * T)) - 1)

        # Convert to erg / s / cm² / Angstrom / sr
        B_lambda *= 1e-8

        return torch.tensor(B_lambda, dtype=torch.float32)

    #-- Define non-dimensional functions

    def r_hat_func(self, r, to_np=False):
        if self.r_norm == 'minmax':
            if to_np: return (r - to_numpy(self.R_MIN))/(to_numpy(self.R_MAX) - to_numpy(self.R_MIN))
            else: return (r - self.R_MIN)/(self.R_0)
        if self.r_norm == 'log':
            if to_np: return np.log(r/to_numpy(self.R_MIN))/np.log(to_numpy(self.R_MAX)/to_numpy(self.R_MIN))
            else: return torch.log(r/self.R_MIN)/torch.log(self.R_MAX/self.R_MIN)
   
    def r_from_rhat(self, rhat, to_np=False):
        if to_np: return to_numpy(self.R_MIN*(self.R_MAX/self.R_MIN)**rhat)
        else: return self.R_MIN*(self.R_MAX/self.R_MIN)**rhat

    def lambda_hat_from_lambda(self, lam):
        """Convert physical wavelength to normalized wavelength [0,1]"""
        return (lam - self.LAMBDA_MIN) / (self.LAMBDA_MAX - self.LAMBDA_MIN)

    def lambda_from_lambda_hat(self, lambda_hat):
        """Convert normalized wavelength to physical wavelength"""
        return self.LAMBDA_MIN + lambda_hat * (self.LAMBDA_MAX - self.LAMBDA_MIN)

    def dr_dr_hat(self, r_hat):
        """Jacobian dr/dr_hat for log-space coordinate."""
        # Ensure R_MIN and R_MAX are on the same device as r_hat
        device = r_hat.device if torch.is_tensor(r_hat) else self.device
        R_MIN_val = self.R_MIN.to(device) if torch.is_tensor(self.R_MIN) else torch.tensor(self.R_MIN, dtype=torch.float32, device=device)
        R_MAX_val = self.R_MAX.to(device) if torch.is_tensor(self.R_MAX) else torch.tensor(self.R_MAX, dtype=torch.float32, device=device)

        r_phys = R_MIN_val * (R_MAX_val / R_MIN_val) ** r_hat
        return r_phys * torch.log(R_MAX_val / R_MIN_val)

    def rho_hat(self, r_hat):
        """Normalized density as function of normalized radius."""
        if torch.is_tensor(r_hat):
            r_hat_np = r_hat.detach().cpu().numpy()
        else:
            r_hat_np = r_hat
        rho_physical = self.rho_func(r_hat_np)
        result = torch.tensor(rho_physical / self.RHO_0.cpu().numpy(),
                            dtype=torch.float32)
        if torch.cuda.is_available():
            result = result.cuda()
        return result

    def kappa_hat(self, r_hat, lambda_hat):
        """Normalized opacity as function of normalized radius and wavelength."""
        if torch.is_tensor(r_hat):
            r_hat_np = r_hat.detach().cpu().numpy()
        else:
            r_hat_np = r_hat

        if torch.is_tensor(lambda_hat):
            lambda_hat_np = lambda_hat.detach().cpu().numpy()
        else:
            lambda_hat_np = lambda_hat

        # Evaluate 2D interpolator
        # Handle both scalar and array inputs
        if np.isscalar(r_hat_np) and np.isscalar(lambda_hat_np):
            kappa_physical = self.kappa_func(r_hat_np, lambda_hat_np)[0, 0]
        else:
            kappa_physical = self.kappa_func(r_hat_np, lambda_hat_np, grid=False)

        result = torch.tensor(kappa_physical / self.K_0.cpu().numpy(),
                            dtype=torch.float32)
        if torch.cuda.is_available():
            result = result.cuda()
        return result

    def S_lam_physical(self, r_hat, lambda_hat):
        """Physical source function: Planck function B_λ(T)."""
        # Convert to numpy for interpolation
        if torch.is_tensor(r_hat):
            r_hat_np = r_hat.detach().cpu().numpy().flatten()
        else:
            r_hat_np = np.atleast_1d(r_hat)

        if torch.is_tensor(lambda_hat):
            lambda_hat_np = lambda_hat.detach().cpu().numpy().flatten()
        else:
            lambda_hat_np = np.atleast_1d(lambda_hat)

        # Vectorized computation
        T = self.temp_func(r_hat_np)
        lam_physical = to_numpy(self.LAMBDA_MIN) + lambda_hat_np * (to_numpy(self.LAMBDA_MAX) - to_numpy(self.LAMBDA_MIN))
        wav_cm = lam_physical * 1e-8

        # Vectorized Planck function
        B_lambda = (2 * self.h * self.c**2 / wav_cm**5) / \
                   (np.exp(self.h * self.c / (wav_cm * self.k * T)) - 1)
        B_lambda *= 1e-8  # Convert to erg/s/cm²/Å/sr

        # Return numpy array (always, even for scalar inputs)
        return B_lambda

    def S_hat(self, r_hat, lambda_hat):
        """Normalized source function."""
        S_phys = self.S_lam_physical(r_hat, lambda_hat)
        # Convert S_phys to tensor maintaining shape
        if not isinstance(S_phys, torch.Tensor):
            S_phys_tensor = torch.tensor(S_phys, dtype=torch.float32, device=self.device)
        else:
            S_phys_tensor = S_phys.to(self.device) if S_phys.device != self.device else S_phys

        # Normalize
        result = S_phys_tensor / self.S_0

        # Ensure proper shape (match input)
        if torch.is_tensor(r_hat) and r_hat.ndim == 2:
            result = result.reshape(r_hat.shape)

        return result

    def S_lam_hat(self, r_hat, lam_phys):
        """Normalized source function: S_hat = S / S_0. (Legacy method for analytic solution)"""
        S_phys = self.S_lam(r_hat, lam_phys)
        if torch.is_tensor(S_phys):
            return (S_phys / self.S_0).cpu().numpy() if S_phys.is_cuda else S_phys / self.S_0.cpu().numpy()
        else:
            return S_phys / self.S_0.cpu().numpy()

    def physics_loss(self, r_hat, lambda_hat, mu):
        """
        Physics loss: residual of radiative transfer equation.
        PDE: μ * dI_hat/dr_hat = τ₀ * dr/dr_hat * ρ_hat * κ_hat * (S_hat - I_hat)
        """
        # Ensure requires_grad for r_hat
        r_hat = r_hat.requires_grad_(True)

        # Forward pass: I_hat = net(r_hat, lambda_hat, mu)
        inputs = torch.cat([r_hat, lambda_hat, mu], dim=1)
        I_hat = self.net(inputs)

        # Compute gradient dI_hat/dr_hat
        dI_hat_drhat = torch.autograd.grad(
            I_hat, r_hat,
            grad_outputs=torch.ones_like(I_hat),
            create_graph=True
        )[0]

        # Get dimensionless quantities (all take lambda_hat, not physical wavelength)
        rho_hat = self.rho_hat(r_hat)
        kappa_hat = self.kappa_hat(r_hat, lambda_hat)
        S_hat = self.S_hat(r_hat, lambda_hat)
        dr_drhat = self.dr_dr_hat(r_hat)

        # PDE residual: μ · dI/dr = ρ·κ·(S - I)
        residual = mu * dI_hat_drhat - self.tau_0 * dr_drhat * rho_hat * kappa_hat * (S_hat - I_hat)

        return torch.mean(residual ** 2)

    def ic_loss(self, lambda_hat, mu):
        """
        Initial condition at r_hat_min: I(r_min, λ, μ) = S(r_min, λ)

        This assumes optically thin regime at inner boundary.
        Holds for all μ values.
        """
        r_hat_min = torch.full_like(lambda_hat, self.r_hat_min)
        inputs = torch.cat([r_hat_min, lambda_hat, mu], dim=1)
        I_hat_pred = self.net(inputs)

        # Target: S_hat at r_min (independent of mu)
        S_hat_target = torch.stack([self.S_hat(self.r_hat_min, lh.item())
                                    for lh in lambda_hat]).reshape(-1, 1)

        return torch.mean((I_hat_pred - S_hat_target) ** 2)

    def predict(self, r_hat, lambda_hat, mu):
        """
        Predict intensity at given (r_hat, lambda_hat, mu) points.

        Args:
            r_hat: Normalized radius (scalar or array)
            lambda_hat: Normalized wavelength (scalar or array)
            mu: Directional cosine (scalar or array)

        Returns:
            I_physical: Physical intensity (erg/s/cm²/Å)
        """
        self.net.eval()

        # Convert to tensors if needed
        if not torch.is_tensor(r_hat):
            r_hat = torch.tensor(r_hat, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(lambda_hat):
            lambda_hat = torch.tensor(lambda_hat, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu, dtype=torch.float32, device=self.device)

        # Ensure proper shape
        if r_hat.dim() == 0:
            r_hat = r_hat.unsqueeze(0)
        if lambda_hat.dim() == 0:
            lambda_hat = lambda_hat.unsqueeze(0)
        if mu.dim() == 0:
            mu = mu.unsqueeze(0)
        if r_hat.dim() == 1:
            r_hat = r_hat.unsqueeze(1)
        if lambda_hat.dim() == 1:
            lambda_hat = lambda_hat.unsqueeze(1)
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)

        # Forward pass
        inputs = torch.cat([r_hat, lambda_hat, mu], dim=1)
        with torch.no_grad():
            I_hat = self.net(inputs)

        # Convert to physical units
        I_physical = I_hat.squeeze().cpu().numpy() * self.S_0.cpu().numpy()

        return I_physical

    def compute_flux(self, r_hat, lambda_hat_array, n_mu_samples=50):
        """
        Compute emergent flux by integrating over μ.

        F(λ) = 2π ∫₀¹ I(r, λ, μ) · μ · dμ

        Args:
            r_hat: Normalized radius (typically r_hat_max for surface flux)
            lambda_hat_array: Array of normalized wavelengths
            n_mu_samples: Number of μ samples for integration

        Returns:
            flux: Array of flux values at each wavelength [erg/s/cm²/Å]
        """
        self.net.eval()

        # Convert r_hat to numpy for consistency
        if torch.is_tensor(r_hat):
            r_hat = r_hat.cpu().numpy()
        r_hat = float(r_hat)

        # Sample μ uniformly
        mu_samples = np.linspace(self.mu_min, self.mu_max, n_mu_samples)

        flux = np.zeros(len(lambda_hat_array))

        for i, lambda_hat in enumerate(lambda_hat_array):
            # Ensure lambda_hat is float
            if torch.is_tensor(lambda_hat):
                lambda_hat = lambda_hat.cpu().numpy()
            lambda_hat = float(lambda_hat)

            # Compute I(r, λ, μ) for all μ
            intensities = []
            for mu in mu_samples:
                I = self.predict(r_hat, lambda_hat, mu)
                intensities.append(I * mu)  # Weight by μ for integration

            # Integrate using trapezoidal rule: ∫ I·μ dμ
            flux[i] = 2 * np.pi * np.trapz(intensities, mu_samples)

        return flux

    def adaptive_sampling(self, n_total, peak_fraction=0.3):
        """
        Adaptive sampling with concentration near 4000 Å peak.
        """
        n_peak = int(n_total * peak_fraction)
        n_uniform = n_total - n_peak

        # Uniform sampling across full range
        lambda_uniform = np.random.uniform(0.0, 1.0, n_uniform)

        # Concentrated sampling near 4000 Å
        LAMBDA_MIN_val = to_numpy(self.LAMBDA_MIN)
        LAMBDA_MAX_val = to_numpy(self.LAMBDA_MAX)
        peak_lambda_hat = (4000.0 - LAMBDA_MIN_val) / (LAMBDA_MAX_val - LAMBDA_MIN_val)
        peak_width = 0.071  # ±500 Å around peak in normalized coordinates
        lambda_peak = np.random.uniform(
            max(0.0, peak_lambda_hat - peak_width),
            min(1.0, peak_lambda_hat + peak_width),
            n_peak
        )

        lambdas = np.concatenate([lambda_uniform, lambda_peak])
        return torch.tensor(lambdas, dtype=torch.float32, device=self.device).reshape(-1, 1)

    def setup_NN(self):

        # Initialize network
        self.net = NN(n_hidden=self.n_hidden, n_layers=self.n_layers)

        # Move network to device
        self.net = self.net.to(self.device)

        # Xavier initialization
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                gain = torch.nn.init.calculate_gain('tanh') if hasattr(self.net, 'activation') else 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # Setup scheduler (ReduceLROnPlateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
        )

        # Training hyperparameters already set in __init__
        # self.n_collocation and self.n_boundary

    def train_NN(self, n_epochs=30000, early_stopping=True, patience=2000, use_adaptive_sampling=True, peak_fraction=0.3):

        print(f"\nTraining PINN (IVP)...", flush=True)
        print(f"Epochs: {n_epochs}, Collocation: {self.n_collocation}, Inner BC: {self.n_boundary}, LR: {self.lr}", flush=True)
        if use_adaptive_sampling:
            print(f"Adaptive sampling: {peak_fraction*100:.0f}% near 4000 Å", flush=True)

        best_loss = float('inf')
        no_improve_count = 0
        history = {'total': [], 'physics': [], 'ic': []}

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            # Sample collocation points
            r_hat_coll = torch.rand(self.n_collocation, 1, device=self.device)
            if use_adaptive_sampling:
                lambda_hat_coll = self.adaptive_sampling(self.n_collocation, peak_fraction)
            else:
                lambda_hat_coll = torch.rand(self.n_collocation, 1, device=self.device)

            # Sample mu uniformly in [mu_min, mu_max]
            mu_coll = torch.rand(self.n_collocation, 1, device=self.device) * (self.mu_max - self.mu_min) + self.mu_min

            # Sample boundary points
            lambda_hat_bc = torch.rand(self.n_boundary, 1, device=self.device)
            # Sample mu for boundary conditions too
            mu_bc = torch.rand(self.n_boundary, 1, device=self.device) * (self.mu_max - self.mu_min) + self.mu_min

            # Compute losses (IVP: physics + inner BC only)
            L_physics = self.physics_loss(r_hat_coll, lambda_hat_coll, mu_coll)
            L_ic = self.ic_loss(lambda_hat_bc, mu_bc)

            # Total loss (IVP formulation with fixed 10.0 weight for IC)
            loss = L_physics + 10.0 * L_ic

            loss.backward()
            self.optimizer.step()

            # Update scheduler
            self.scheduler.step(loss.item())

            history['total'].append(loss.item())
            history['physics'].append(L_physics.item())
            history['ic'].append(L_ic.item())

            # Early stopping
            if early_stopping:
                if loss.item() < best_loss - 1e-6:
                    best_loss = loss.item()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
                    break
            else:
                if loss.item() < best_loss:
                    best_loss = loss.item()

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                lr_current = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.6e} | "
                      f"LR: {lr_current:.2e} | Best: {best_loss:.6e}", flush=True)

        print(f"\nTraining complete! Best loss: {best_loss:.6e}", flush=True)
        self.history = history

        # Save model
        save_dict = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': history,
            'best_loss': best_loss
        }
        torch.save(save_dict, f'runs/{self.test_name}/RadAI_model.pt')
        print(f"Model saved: runs/{self.test_name}/RadAI_model.pt", flush=True)

    def make_plots(self):

        # Note: analytic solutions (if computed for multiple mu) would be in self.I_nu_analytic[mu_val]
        # For now, we'll focus on PINN predictions at multiple mu values and flux comparison

        R_MIN, R_MAX = to_numpy(self.R_MIN), to_numpy(self.R_MAX)
        ref_R = np.array([[R_MIN, R_MIN+(R_MAX/self.r_max_factor-R_MIN)/5], [R_MIN+(R_MAX/self.r_max_factor-R_MIN)/2, R_MAX/self.r_max_factor]])

        Inunn_start = datetime.now()

        # Compute I_nu_nn for multiple mu values to show angular dependence
        mu_values = [0.2, 0.5, 0.8, 1.0]
        I_nu_nn = {}  # Dictionary to store I(r,λ,μ) for each μ

        for mu_val in mu_values:
            print(f'Computing PINN predictions for mu={mu_val}...', flush=True)
            I_nu_nn[mu_val] = np.zeros((len(self.Rs), len(self.wav_nn)))
            self.net.eval()
            with torch.no_grad():
                for l, lmd_value in enumerate(self.wav_nn):
                    lambda_hat = self.lambda_hat_from_lambda(lmd_value) * torch.ones(len(self.Rs), 1)
                    r_hat = torch.tensor(self.r_hat_func(self.Rs, to_np=True), dtype=torch.float32).reshape(-1, 1)
                    mu_tensor = torch.full_like(r_hat, mu_val)
                    inputs = torch.cat([r_hat, lambda_hat, mu_tensor], dim=1).to(self.device)
                    I_hat_nn = self.net(inputs)
                    I_nu_nn[mu_val][:, l] = I_hat_nn.squeeze().cpu().numpy()
            I_nu_nn[mu_val] *= to_numpy(self.S_0)
        
        print('I_nu_nn calculation took: ', (datetime.now()-Inunn_start).total_seconds()/60, ' minutes')

        # Plot intensity at different radii for multiple mu values
        plt.rc('font', size=20)
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        colors_mu = {0.2: 'purple', 0.5: 'green', 0.8: 'orange', 1.0: 'red'}
        for i in [0, 1]:
            for j in [0, 1]:
                plt.rc('font', size=18)
                R_idx = np.argmin(np.abs(self.Rs-ref_R[i, j]))
                # Plot PINN predictions for each mu value
                for mu_val in mu_values:
                    axs[i, j].plot(self.wav_nn, I_nu_nn[mu_val][R_idx, :],
                                  c=colors_mu[mu_val], label=f'NN μ={mu_val}', alpha=0.8, linewidth=1.5)
                # Add Korg flux for reference at surface
                if i == j == 1:
                    axs[i, j].plot(to_numpy(self.korg_wav), to_numpy(self.korg_flux),
                                  c='blue', ls='--', alpha=0.5, label=r'$F_{\lambda, \rm{Korg}}$')
                axs[i, j].set_title(r'$R = {:4.2f}$ cm'.format(self.Rs[R_idx]))
                axs[i, j].set_xscale('log')
                axs[i, j].set_xlabel(r'$\lambda \ [\AA]$')
                axs[i, j].set_ylabel(r'$I_\lambda \ [\frac{erg}{s cm^2 \AA sr}]$')
                axs[i, j].set_xlim([to_numpy(self.LAMBDA_MIN)-500, to_numpy(self.LAMBDA_MAX)+500])
                axs[i, j].legend(fontsize=10, loc='best')
                plt.tight_layout()
                filters = np.array(glob.glob('filters/*'))
                if len(filters) > 0:
                    wavelengths = 'grizyJHKS'
                    colors_filt = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
                     "H": "orange", "K": "red", "S": "darkred"}
                    text_locs = {"g": 0.09, "r": 0.25, "i": 0.35, "z": 0.44, "y": 0.51, "J": 0.61, "H": 0.75, "K": 0.92}
                    for fltr in range(len(filters)):
                        filter_wavs = np.loadtxt(filters[fltr])
                        filter_wavs = filter_wavs[:, 0] # Angstroms
                        wav_low, wav_upp = filter_wavs[0], filter_wavs[-1]
                        fltr_band = filters[fltr].split('/')[-1][0]
                        if fltr_band == "S": continue
                        fltr_indx = wavelengths.find(fltr_band)
                        axs[i, j].axvspan(wav_low, wav_upp, alpha=0.2, color=colors_filt[wavelengths[fltr_indx]])

                axs[i, j].set_yscale('log')

        plt.suptitle(f"{self.test_name} - Angular Dependence", fontsize=22)
        plt.savefig(f'runs/{self.test_name}/I_vs_lambda.pdf')

        # Compute and plot flux comparison: PINN vs Korg
        print('Computing PINN flux by integrating over mu...', flush=True)
        flux_start = datetime.now()
        lambda_hat_surf = np.linspace(0, 1, len(self.wav_nn))
        pinn_flux = self.compute_flux(self.r_hat_max, lambda_hat_surf, n_mu_samples=50)
        print(f'Flux computation took: {(datetime.now()-flux_start).total_seconds()/60} minutes', flush=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.wav_nn, pinn_flux, 'r-', label='PINN flux', linewidth=2)
        ax.plot(to_numpy(self.korg_wav), to_numpy(self.korg_flux), 'b--', label='Korg flux', alpha=0.7, linewidth=2)
        ax.set_xlabel(r'$\lambda$ [Å]', fontsize=14)
        ax.set_ylabel(r'$F_\lambda$ [erg/s/cm²/Å]', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=12)
        ax.set_title(f'Emergent Flux: PINN vs Korg\n{self.test_name}', fontsize=16)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'runs/{self.test_name}/flux_comparison.pdf')
        print(f'Saved flux comparison plot to runs/{self.test_name}/flux_comparison.pdf', flush=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(self.history['total'], label='train loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'runs/{self.test_name}/loss.pdf')

        n_epochs = len(self.history['total']) # update n_epochs to be a user-supplied hyperparameter

        np.savetxt(f'runs/{self.test_name}/train_loss.dat', np.c_[np.arange(n_epochs), self.history['total']], fmt="%d \t %f")

    def lr_func(self, a, b, c): 
        # wrapper function to plot learning rate evolution on MonitorND
        return torch.tensor(self.optimizer.param_groups[0]['lr'], dtype=torch.float32)

class NN(torch.nn.Module):
    def __init__(self, n_hidden=512, n_layers=5):
        super(NN, self).__init__()
        self.input = torch.nn.Linear(3, n_hidden)  # Changed from 2 to 3 for [r_hat, lambda_hat, mu]
        self.linears = torch.nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden) for i in range(n_layers)])
        self.output = torch.nn.Linear(n_hidden, 1, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation(self.input(x))
        for linear in self.linears:
            x = self.activation(linear(x))
        x = self.output(x)
        return x
