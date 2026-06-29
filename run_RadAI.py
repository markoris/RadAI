#© 2026. Triad National Security, LLC. All rights reserved.
#This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

from RadAI import RadAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--r-norm', default='log')
parser.add_argument('--r-max-factor', default=1)
parser.add_argument('--test-name', default='RadAI')
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--n-hidden', default=512, type=int)
parser.add_argument('--n-layers', default=5, type=int)
parser.add_argument('--mu-min', default=0.01, type=float, help='Minimum mu for training')
parser.add_argument('--mu-max', default=1.0, type=float, help='Maximum mu for training')
parser.add_argument('--n-collocation', default=8192, type=int, help='Number of collocation points')
parser.add_argument('--n-boundary', default=100, type=int, help='Number of boundary points')

args = parser.parse_args()
PINN = RadAI(**vars(args))
PINN.setup_NN()
PINN.train_NN(n_epochs=40000, early_stopping=True, patience=3000, use_adaptive_sampling=True, peak_fraction=0.6)
PINN.make_plots()

