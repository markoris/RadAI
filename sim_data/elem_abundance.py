#© 2026. Triad National Security, LLC. All rights reserved.
#This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy import constants

def abun_elem():
    elem = {
    'H': 12.00, 
    'He': 10.93, 
    'Li': 1.05,    
    'Be': 1.38,   
    'B': 2.70 ,  
    'C': 8.39 ,  
    'N': 7.78 ,  
    'O': 8.66 ,  
    'F': 4.56 ,  
    'Ne': 7.84,  
    'Na': 6.17, 
    'Mg': 7.53, 
    'Al': 6.37, 
    'Si': 7.51, 
    'P': 5.36 , 
    'S': 7.14 , 
    'Cl': 5.50, 
    'Ar': 6.18,
    'K': 5.08 , 
    'Ca': 6.31, 
    'Sc': 3.17, 
    'Ti': 4.90, 
    'V': 4.00 , 
    'Cr': 5.64, 
    'Mn': 5.39, 
    'Fe': 7.45, 
    'Co': 4.92, 
    'Ni': 6.23, 
    'Cu': 4.21, 
    'Zn': 4.60, 
    'Ga': 2.88, 
    'Ge': 3.58, 
    'Kr': 3.25,     
    'Rb': 2.60,
    'Sr': 2.92,
    'Y': 2.21 ,
    'Zr': 2.58,
    'Nb': 1.42,
    'Mo': 1.92,
    'Ru': 1.84,
    'Rh': 1.12,
    'Pd': 1.66,
    'Ag': 0.94,
    'Cd': 1.77,
    'In': 1.60,
    'Sn': 2.00,
    'Sb': 1.00,
    'Xe': 2.24,
    'Ba': 2.17,
    'La': 1.13,
    'Ce': 1.70,
    'Pr': 0.58,
    'Nd': 1.45,
    'Sm': 1.00,
    'Eu': 0.52,
    'Gd': 1.11,
    'Tb': 0.28,
    'Dy': 1.14,
    'Ho': 0.51,
    'Er': 0.93,
    'Tm': 0.00,
    'Yb': 1.08,
    'Lu': 0.06,
    'Hf': 0.88,
    'W': 1.11 ,
    'Os': 1.25,
    'Ir': 1.38,
    'Au': 1.01,
    'Tl': 0.90,
    'Pb': 2.00,
    'Bi': 0.65,
    'Th': 0.06,
    'U': -0.47}
    return elem 


# meteoric:
def abun_meteoric():
    elem = {
        'H': 8.25, 
        'He':1.29,
        'Li':3.25 ,
        'Be': 1.38, 
        'B': 2.75 ,
        'C': 7.40 ,
        'N': 6.25 ,
        'O': 8.39 ,
        'F': 4.43 ,
        'Ne': -1.06,      
        'Na': 6.27,
        'Mg': 7.53,
        'Al': 6.43,
        'Si': 7.51,
        'P': 5.40 ,
        'S': 7.16 ,
        'Cl': 5.23,
        'Ar': -0.45,      
        'K': 5.06 ,
        'Ca': 6.29,
        'Sc': 3.04,
        'Ti': 4.89,
        'V': 3.97 ,
        'Cr': 5.63,
        'Mn': 5.47,
        'Fe': 7.45,
        'Co': 4.86,
        'Ni': 6.19,
        'Cu': 4.23,
        'Zn': 4.61,
        'Ga': 3.07,
        'Ge': 3.59,
        'As': 2.29,             
        'Se': 3.33,             
        'Br': 2.56,             
        'Kr': -2.27,
        'Rb': 2.33, 
        'Sr': 2.88, 
        'Y': 2.17,
        'Zr': 2.57, 
        'Nb': 1.39, 
        'Mo': 1.96, 
        'Ru': 1.77, 
        'Rh': 1.07, 
        'Pd': 1.67, 
        'Ag': 1.20, 
        'Cd': 1.71, 
        'In': 0.80, 
        'Sn': 2.08, 
        'Sb': 1.03, 
        'Te': 2.19,
        'I': 1.51 ,
        'Xe': -1.97,
        'Cs': 1.07, 
        'Ba': 2.16, 
        'La': 1.15, 
        'Ce': 1.58, 
        'Pr': 0.75, 
        'Nd': 1.43, 
        'Sm': 0.92, 
        'Eu': 0.49, 
        'Gd': 1.03, 
        'Tb': 0.28, 
        'Dy': 1.10, 
        'Ho': 0.46, 
        'Er': 0.92, 
        'Tm': 0.08, 
        'Yb': 0.91, 
        'Lu': 0.06, 
        'Hf': 0.74, 
        'Ta': -0.17,
        'W': 0.62 ,
        'Re': 0.23,
        'Os': 1.34,
        'Ir': 1.32,
        'Pt': 1.64,
        'Au': 0.80,
        'Hg': 1.13,
        'Tl': 0.78,
        'Pb': 2.02,
        'Bi': 0.65,
        'Th': 0.06,
        'U': -0.52}
    return elem

def elem_masses():
    amu = {
    'H': 1.0080     , 
    'He':4.00260    , 
    'Li':7.0        , 
    'Be':9.012183   , 
    'B': 10.81      ,  
    'C': 12.011     ,  
    'N': 14.007     ,  
    'O': 15.999     ,  
    'F': 18.99840316,  
    'Ne':20.180     ,        
    'Na':22.9897693 ,  
    'Mg':24.305     ,  
    'Al':26.981538  ,  
    'Si':28.085     ,  
    'P': 30.97376200,  
    'S': 32.07      ,  
    'Cl':35.45      ,  
    'Ar':39.9       ,        
    'K': 39.0983    ,  
    'Ca':40.08      ,  
    'Sc':44.95591   ,  
    'Ti':47.867     ,  
    'V': 50.9415    ,  
    'Cr':51.996     ,  
    'Mn':54.93804   ,  
    'Fe':55.84      ,  
    'Co':58.93319   ,  
    'Ni':58.693     ,  
    'Cu':63.55      ,  
    'Zn':65.4       ,  
    'Ga':69.723     ,  
    'Ge':72.63      ,  
    'As':74.92159   ,              
    'Se':78.97      ,              
    'Br':79.90      ,              
    'Kr':83.80      ,  
    'Rb':85.468     ,  
    'Sr':87.62      ,  
    'Y': 88.90584   ,  
    'Zr':91.22      ,  
    'Nb':92.90637   ,  
    'Mo':95.95      ,  
    'Ru':101.1      ,  
    'Rh':102.9055   ,  
    'Pd':106.42     ,  
    'Ag':107.868    ,  
    'Cd':112.41     ,  
    'In':114.818    ,  
    'Sn':118.71     ,  
    'Sb':121.760    ,  
    'Te':127.6      ,  
    'I': 126.9045   ,  
    'Xe':131.29     ,  
    'Cs':132.9054520,  
    'Ba':137.33     ,  
    'La':138.9055   ,  
    'Ce':140.116    ,  
    'Pr':140.90766  ,  
    'Nd':144.24     ,  
    'Sm':150.4      ,  
    'Eu':151.964    ,  
    'Gd':157.25     ,  
    'Tb':158.92535  ,  
    'Dy':162.500    ,  
    'Ho':164.93033  ,  
    'Er':167.26     ,  
    'Tm':168.93422  ,  
    'Yb':173.05     ,  
    'Lu':174.9667   ,  
    'Hf':178.49     ,  
    'Ta':180.9479   ,  
    'W': 183.84     ,  
    'Re':186.207    ,  
    'Os':190.2      ,  
    'Ir':192.22     ,  
    'Pt':195.08     ,  
    'Au':196.96657  ,  
    'Hg':200.59     ,  
    'Tl':204.383    ,  
    'Pb':207        ,  
    'Bi':208.98040  ,  
    'Th':232.038    ,  
    'U': 238.0289} 
    return amu









































 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
