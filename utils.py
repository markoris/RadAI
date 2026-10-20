#© 2026. Triad National Security, LLC. All rights reserved.
#This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

import sys
sys.path.insert(0, 'sim_data')
import dill
import torch
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy import constants
from torch.optim import Adam
from natsort import natsorted
from elem_abundance import abun_elem, abun_meteoric, elem_masses 

def to_numpy(x):
    ''' helper function to convert PyTorch tensor to NumPy array '''
    return x.detach().cpu().numpy()

def save(solver, fname="model_params.pt"):
    '''
    Saves a BundleIVP1D solver with the assumption that an Adam optimizer is used
    '''

    save_dict = solver.get_internals('all')
    save_dict['optimizer'] = solver.optimizer.state_dict()
    save_dict['type'] = solver.__class__
    save_dict['lr'] = solver.optimizer.param_groups[0]['lr']
    with open("model_params.pt", 'wb') as fname:
        dill.dump(save_dict, fname)

    return

def load(fname, train=False, return_optim=False):
    '''
    Loads a BundleIVP1D solver with the assumption of an Adam optimizer
    '''
    
    with open(fname, 'rb') as f:
        solver_params = dill.load(f)
    solver = solver_params["type"]( ode_system=solver_params['diff_eqs'], 
                                        conditions=solver_params['conditions'],
                                        t_min=solver_params['r_min'][0],
                                        t_max=solver_params['r_max'][0],
                                        nets=solver_params['nets'],
                                        theta_min=solver_params['r_min'][1:],
                                        theta_max=solver_params['r_max'][1:],
                                        n_batches_valid=solver_params['n_batches'],
                                  )
    for key in solver_params.keys():
        if not train: 
            if key in ["diff_eqs", "conditions", "r_min", "r_max", "n_batches"]: 
                continue
        if key in ["global_epoch"]: continue
        setattr(solver, key, solver_params[key])

    if return_optim:
        optim = Adam(solver_params['nets'][0].parameters(), lr=solver_params['lr'])
        optim.load_state_dict(solver_params["optimizer"])
        return solver, optim

    return solver

def train(solver, epochs=[500, 1000, 2000, 4000, 8000], lrs=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]):

    for epoch, lr in zip(epochs, lrs):

        for g in solver.optimizer.param_groups:
            g['lr'] = lr
        #setattr(solver, 'optimizer', Adam(solver.nets[0].parameters(), lr=lr))
        solver.fit(max_epochs=epoch)
        #save(solver, fname='solver.chkp')

    return solver

def nbins(Rmin, Rmax, tau0, tau_factor=10):
    if np.any(tau0 == tau_factor):
        print('Warning: At least one of the tau0 values is equal to tau_factor!')
        print('Warning: This will throw an error in the denominator term for log10(0).')
    # tau_factor is approximately the largest delta_tau value to expect for the returned nbins
    numerator = np.log10(Rmax/Rmin)
    denominator = np.log10(1 - (tau_factor/tau0))
    num_bins = np.array(np.abs(numerator/denominator).astype('int'), ndmin=1)
    num_bins[np.where(num_bins < 2)] = 2
    if len(num_bins) < 2: num_bins = num_bins[0]
    return num_bins
    
def get_N_elem(Y):
        Y -= 12
        Y = 10**Y           # assuming log base 10, yields N_el / N_H
        return Y

def load_korg_data():

        #-- Load temperature, opacity, and density data
         
        #korg_params = fits.open('sim_data/marcs_t6000_g400_f0.fits')
        korg_params = fits.open('sim_data/marcs_sph_t5770_g444_f0.fits')
        korg_params = np.array(korg_params[1].data)
        #spec = fits.open('sim_data/korgspec_t6000_g400_f0.fits')
        spec = fits.open('sim_data/korgspec_marcs_sph_5770_444.fits')
        korg_wav = torch.tensor(spec[1].data['wave'].flatten().astype(float), dtype=torch.float32)
        korg_flux = spec[1].data['flux'].astype(float)
        #korg_kappa = fits.open('sim_data/korgalpha_t6000_g400_f0.fits')[1].data
        korg_kappa = fits.open('sim_data/korgalpha_marcs_sph_5770_444.fits')[1].data

#        korg_flux *= u.erg / (u.cm**2 * u.s * u.cm * u.sr) # cgs: ergs / s / cm^2 / sr / cm ?
#        korg_flux = korg_flux.to('erg / (cm^2 s AA sr)').value
#        korg_flux = torch.tensor(korg_flux, dtype=torch.float32)

        n_e = korg_params['electron_number_density'].flatten()*u.cm**-3
        n_gas = korg_params['number_density'].flatten()*u.cm**-3
        temp = korg_params['temp'].flatten()
        z = korg_params['z'].flatten() # scale height
        tau_5000 = korg_params['tau_5000'].flatten()
        
        abun = abun_elem()      # log10(N_el / N_H) + 12
        masses = elem_masses()  # atomic mass units (amu)
        
        # sum of abundances transcribed from table does not add up to 0.013
        # we introduce a normalization factor w_i = 0.013 * Y_i / sum(Y_i) to ensure sum(w_i) = 0.013
        # we get the mass of the metals m_metals = sum( w_i * m_i )
        
        # [2:] to skip H and He in abundance sum; these are covered by X and Y in 1 = X + Y + Z (where Z are our metals)
        sum_abun = get_N_elem(np.array([a for a in abun.values()]))[2:].sum()
        norm_factor = (1 - 0.738 - 0.2485)/sum_abun
        
        amu = constants.u.to('gram')
        # calculate mass contribution from metals 
        for element in abun.keys():
            if element == 'H' or element == 'He': continue
            elem_abun = get_N_elem(abun[element])#/sum_abun # fractional abundance of the element
            elem_mass = elem_abun * norm_factor * masses[element] * amu # fractional abundance times norm_factor to scale to 0.013 = 1 - X - Y times the mass 
            try: mass_metals.append(elem_mass) 
            except NameError: mass_metals = [elem_mass]
        
        #m_H + m_He + m_z
        mass_gas = 0.738 * amu + 0.2485 * (4 * amu) + sum(mass_metals)
        rho_gas = mass_gas*n_gas
        rho_e = n_e * constants.m_e.to('gram')
        korg_rho = rho_gas + rho_e

        # trim
        z_0 = z.min() # model z of the deepest layer; the model's z = 0 is at tau_Ross = 1
        z -= z_0
        z = np.flip(z)
        z[0] += 1 # pretend we start at 1 cm for log-completeness
        temp = np.flip(temp)
        tau_5000 = np.flip(tau_5000)
        tau_cut = np.where(tau_5000 > 14)[0]
        z = np.delete(z, tau_cut)
        temp = np.delete(temp, tau_cut)
        tau_5000 = np.delete(tau_5000, tau_cut)
        
        # kappa setup
        #for name_idx in range(len(korg_kappa.names)):
        for name in natsorted(korg_kappa.names):
            #name = korg_kappa.names[name_idx]
            #opac = korg_kappa[name].flatten()*u.cm**-1/korg_rho[name_idx]
            opac = korg_kappa[name].flatten()*u.cm**-1/korg_rho[int(name[5:])]
            try: korg_kappa_array = np.concatenate((korg_kappa_array, opac[None, :]), axis=0)
            except NameError: korg_kappa_array = opac[None, :]
        korg_kappa_array = np.flip(korg_kappa_array, axis=0).astype('float')
        korg_kappa_array = np.delete(korg_kappa_array, tau_cut, axis=0)

        print(korg_kappa_array)
        
        korg_rho = np.flip(korg_rho.value)
        korg_rho = np.delete(korg_rho, tau_cut)

        return korg_wav, korg_flux, z, temp, korg_rho, korg_kappa_array, tau_5000, z_0

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        n_hidden = 512
        n_layers = 5
        self.input = torch.nn.Linear(2, n_hidden)
        self.bn = torch.nn.BatchNorm1d(2, n_hidden)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden) for i in range(n_layers)])
        self.output = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = torch.nn.SiLU()(self.input(x))
        for linear in self.linears:
            x = torch.nn.SiLU()(linear(x))
        x = self.output(x)
        return x
