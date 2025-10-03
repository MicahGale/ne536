# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:01:17 2021

@author: bgeiger3
"""
import matplotlib.pyplot as plt
plt.close('all')
## --------------------------------------------------------------
## ----------------- SPARC prediction ----------------------------
## --------------------------------------------------------------
## define parameters
params = {}
params['name']='SPARC' 
params['kind'       ] = 'tokamak'
params['R'          ] = 3.3
params['a'          ] = 1.13        # [m] minor radius
params['kappa_s'    ] = 1.84        # []  elongation
params['S'          ] = 2.63       # shaping factor
params['B_tor'      ] = 9.2        # [T] toroidal magnetic field strength
params['I_p'        ] = 7.8        # [MA]

params['H_factor'   ] = 1.8         # []     confinment scaling factor
params['f_alpha_loss'] = 0.02      # []     alpha particle loss fraction
params['fueling_mix']=[0.5,0.5,0.]

## Density profile settings
params['n_0'        ] = 3.0       # [10^20/m^3] core electron density
params['mu_n'       ] = 0.5        # []     electron density peaking
params['offset_n'   ] = 0.5        # [10^20/m^3]

## Temperature offset
params['offset_T'   ] = 0.05
    
## impurity settings
params['C_fraction' ] = 0.02       # []    Fraction of Carbon impurity
params['Fe_fraction'] = 0.0001     # []    Fraction of Fe impurity
params['rho_He'     ] = 5          # []    He confinment
params['rho_prad_core']=1/2      # radius upto which the radiated power is considered
params['R_sync']=1 ## don't consider synchrotron radiation.

## ECRH 
params['P_ecrh'     ] = 25.0       # MW
params['P_nbi'     ]  = 0          # MW

## Initial alpha particle heating power
params['P_alpha'    ] = 100.0        # MW



from toolbox import extend_parameters
params=extend_parameters(params)

## Evaluate the fusion reactor performance (calcualte the temperarture profile)
from toolbox import calc_profile_evolution
params=calc_profile_evolution(params)

## Plot the evoluation of some plasma parameters
from toolbox import plot_plasma_evolution
plot_plasma_evolution(params)

## plot the final profiles
from toolbox import plot_profiles
plot_profiles(params)
 
## print the results
from toolbox import print_results
print_results(params)
