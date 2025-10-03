# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:13:30 2022

@author: bgeiger3
"""
import numpy as np
import scipy.constants as consts
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import os
matplotlib.rcParams.update({'figure.max_open_warning':False})
plt.rcParams["mathtext.fontset"] = "dejavuserif"
matplotlib.rcParams.update({'font.size': 18})


## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------
## ------------------ initialization routine ----------------------------------
## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------
def extend_parameters(params,calc_cx=True):
       
    ## general switches
    if not 'plot_cooling_rates' in params:
        params['plot_cooling_rates']=False
    if not 'plot_fusion_rates' in params:
        params['plot_fusion_rates']=False
    if not 'verbose' in params:
        ## print out stat and results
        params['verbose']=False     
    if not 'time_step' in params:
        params['time_step']=0.1

    ## temperature profile offset
    if not 'offset_T' in params:
        params['offset_T']=1.e-5
    else:
        if params['offset_T'] <=0:
            ## set offset tempeature to a finite value
            params['offset_T']=1.e-5
        
    ## density profile settings
    if not 'slope_n' in params:
        ## additional slope of the density profile
        params['slope_n'] = 0
    
    
    ## confinment settings and fueling mix
    if not 'rho_He' in params:
        params['rho_He']=5. # 5 times energy confinement from literature
    if not 'f_alpha_loss' in params:
        ## alpha particle loss fraction
        params['f_alpha_loss']=0.
    if not 'rho_prad_core' in params:
        ## this defines the normalized radius upto which the radiated power is subtracted from the absorbed heating power 
        ## to obtain a net heating power (P_net) that is plugged into the 0D scaling laws
        params['rho_prad_core']=1
    if not 'fueling_mix' in params:
        ## main-ion fueling mix   D  T  H
        params['fueling_mix']  = [0.,0.,1.] ## only hydrogen
        
    if not 'R_sync' in params:
        ## this is the synchrotron radiation reflection coefficicent.
        params['R_sync']=0.9

    ## Inital Fusion Powers
    if not 'P_DT_alpha' in params:
         ## initial DT alpha particle power 
         params['P_DT_alpha']=0.
    if not 'P_DD_p' in params:
         ## initial DD proton power 
         params['P_DD_p']=0.
    if not 'P_DD_T' in params:
         ## initial DD T power 
         params['P_DD_T']=0.
    if not 'P_DD_He3' in params:
         ## initial DD He3 power 
         params['P_DD_He3']=0. 
    if not 'P_DHe3_p' in params:
         ## initial DHe3 proton power
         params['P_DHe3_p']=0.
    if not 'P_DHe3_alpha' in params:
         ## initial DHe3 alpha power
         params['P_DHe3_alpha']=0. 





    ## ------------------------------------------------------------------------
    ## Calcualte Q-values and fusion rates from cross-sections
    ## ------------------------------------------------------------------------
    ## First check if there is some deuterium fueled (skip in the case of an H-plasma)
    directory= os.path.dirname(__file__) + '/Tables_and_Data/'
    if params['fueling_mix'][0]>0:
        mass = {'D': 2.01410178, 'T': 3.01604928, 'He4':4.00260325, 'He3':3.0160293,'n':1.00866492, 'H':1.007825}
        ## --------------------------------------------------------------------
        ## DT fusion
        ## --------------------------------------------------------------------
        params['Q_DT']=(mass['D']+mass['T'] - (mass['He4'] + mass['n']))*consts.atomic_mass*consts.c**2/consts.e/1.e6
        params['DT_alpha_energy']=params['Q_DT']/(1+mass['He4']/mass['n'])
        params['DT_n_energy']=params['Q_DT']-params['DT_alpha_energy']
        ## --------------------------------------------------------------------
        ## DD fusion
        ## --------------------------------------------------------------------
        ## D+D --> p+T
        params['Q_DD_pT']=(mass['D']+mass['D'] - (mass['T'] + mass['H']))*consts.atomic_mass*consts.c**2/consts.e/1.e6
        params['DD_T_energy']=params['Q_DD_pT']/(1+mass['T']/mass['H'])
        params['DD_p_energy']=params['Q_DD_pT']-params['DD_T_energy']
        ## D+D --> n+He3
        params['Q_DD_nHe3']=(mass['D']+mass['D'] - (mass['He3'] + mass['n']))*consts.atomic_mass*consts.c**2/consts.e/1.e6
        params['DD_He3_energy']=params['Q_DD_nHe3']/(1+mass['He3']/mass['n'])
        params['DD_n_energy']=params['Q_DD_nHe3']-params['DD_He3_energy']
        ## --------------------------------------------------------------------
        ## DHe3 fusion
        ## --------------------------------------------------------------------
        params['Q_DHe3']=(mass['D']+mass['He3'] - (mass['H'] + mass['He4']))*consts.atomic_mass*consts.c**2/consts.e/1.e6
        params['DHe3_alpha_energy']=params['Q_DHe3']/(1+mass['He4']/mass['H'])
        params['DHe3_p_energy']=params['Q_DHe3']-params['DHe3_alpha_energy']
        ## print out results
        if params['verbose'] == True:
            print('---- DT fusion ----')
            print('Q-DT: %.4f'%params['Q_DT'],'MeV')
            print('Alpha energy: %.4f'%params['DT_alpha_energy'],'MeV')
            print('neutron energy: %.4f'%params['DT_n_energy'],'MeV')
            print('---- DD fusion ----')
            print('Q-DD_pT: %.4f'%params['Q_DD_pT'],'MeV')
            print('DD Tritium energy: %.4f'%params['DD_T_energy'],'MeV')
            print('DD proton energy: %.4f'%params['DD_p_energy'],'MeV')
            print('Q-DD_nHe3: %.4f'%params['Q_DD_nHe3'],'MeV')
            print('DD He3 energy: %.4f'%params['DD_He3_energy'],'MeV')
            print('DD neutron energy: %.4f'%params['DD_n_energy'],'MeV')
            print('---- DHe3 fusion ----')
            print('Q-DHe3: %.4f'%params['Q_DHe3'],'MeV')
            print('DHe3 alpha energy: %.4f'%params['DHe3_alpha_energy'],'MeV')
            print('DHe3 proton energy: %.4f'%params['DHe3_p_energy'],'MeV')
        ## ---------------------------------------------------------
        ## Calcualte fusion rates
        ## --------------------------------------------------------- 
        ## define Temperature array
        params['log10_Ti_arr_sigma_v']=np.linspace(-2,3,num=200) # log10(keV)
        ## ---------------------------------------------------------
        ## DT fusion
        ## ---------------------------------------------------------
        from fusion_cross_sections import read_sigma
        sigma,Egrid = read_sigma(directory+'D_T_-_4He_n.txt',mass['D'],mass['T'])
        ## calculate sigma_v on a log grid 
        from fusion_reaction_rates import calc_sigma_v_analytical
        params['sigma_v_D(T,n)He4']=calc_sigma_v_analytical(Egrid,sigma,10**params['log10_Ti_arr_sigma_v']*1.e3,mass['D'],mass['T'])
        ## sort out zero and negative values
        index=params['sigma_v_D(T,n)He4']==0.
        params['sigma_v_D(T,n)He4'][index]=np.min(params['sigma_v_D(T,n)He4'][params['sigma_v_D(T,n)He4']>0])
        ## ---------------------------------------------------------
        ## DD fusion
        ## ---------------------------------------------------------    
        ## D+D -> p + T
        sigma,Egrid = read_sigma(directory+'D_D_-_T_p.txt',mass['D'],mass['D'])
        params['sigma_v_D(D,p)T']=calc_sigma_v_analytical(Egrid,sigma,10**params['log10_Ti_arr_sigma_v']*1.e3,mass['D'],mass['D'])
        ## sort out zero and negative values
        index=params['sigma_v_D(D,p)T']==0.
        params['sigma_v_D(D,p)T'][index]=np.min(params['sigma_v_D(D,p)T'][params['sigma_v_D(D,p)T']>0])   
        ##----------------------------------------------------------
        ## D+D -> n + He3
        sigma,Egrid = read_sigma(directory+'D_D_-_3He_n.txt',mass['D'],mass['D'])
        params['sigma_v_D(D,n)He3']=calc_sigma_v_analytical(Egrid,sigma,10**params['log10_Ti_arr_sigma_v']*1.e3,mass['D'],mass['D'])
        ## sort out zero and negative values
        index=params['sigma_v_D(D,n)He3']==0.
        params['sigma_v_D(D,n)He3'][index]=np.min(params['sigma_v_D(D,n)He3'][params['sigma_v_D(D,n)He3']>0])   
        ## ---------------------------------------------------------
        ## D-He3 fusion
        ## ---------------------------------------------------------    
        ## D+He3 -> p + He4
        sigma,Egrid = read_sigma(directory+'D_3He_-_p_4He.txt',mass['D'],mass['He3'])
        params['sigma_v_D(He3,p)He4']=calc_sigma_v_analytical(Egrid,sigma,10**params['log10_Ti_arr_sigma_v']*1.e3,mass['D'],mass['He3'])
        ## sort out zero and negative values
        index=params['sigma_v_D(He3,p)He4']==0.
        params['sigma_v_D(He3,p)He4'][index]=np.min(params['sigma_v_D(He3,p)He4'][params['sigma_v_D(He3,p)He4']>0])   
        ## --------------------------------------------------------------------
        ## plot the fusion rates employed in this calcualtion
        ## --------------------------------------------------------------------
        if params['plot_fusion_rates'] == True:
            plt.figure()
            plt.grid()
            plt.loglog(10**params['log10_Ti_arr_sigma_v'],params['sigma_v_D(T,n)He4'],label=r'D(T,n)$^4$He')
            plt.loglog(10**params['log10_Ti_arr_sigma_v'],params['sigma_v_D(D,p)T'],label='D(D,p)T') 
            plt.loglog(10**params['log10_Ti_arr_sigma_v'],params['sigma_v_D(D,n)He3'],label='D(D,n)$^3$He')
            plt.loglog(10**params['log10_Ti_arr_sigma_v'],params['sigma_v_D(He3,p)He4'],label='D($^3$He,p)$^4$He')
            plt.legend(fontsize=16)
            plt.xlabel(r'T [keV]')
            plt.ylabel(r'$\mathrm{{<}\sigma\,v{>}}$ [m$^3$/s]')
            plt.xlim(1,1.e3)
            plt.ylim(1.e-25, 1.e-21)
            plt.tight_layout()
            plt.savefig('Plots/fusion_reaction_rates.png')



    ## read tabulated data on the mean impurity charge
    #from Tables_and_Data.read_meanz_data import read_meanz
    #Te_meanz,element_meanz,meanz_data=read_meanz(directory+'lz_puetti_meanz.dat')
    #params['Te_meanz']=Te_meanz/1.e3 #keV
    #params['meanz_Fe']=meanz_data[np.where(element_meanz == 'Fe')[0][0]]
    
    params['P_nbi_net']  =  params['P_nbi']
    ## absorbed power
    params['P_abs']=params['P_nbi_net']+params['P_ecrh']+params['P_DT_alpha'] \
        +params['P_DD_p']+params['P_DD_T']+params['P_DD_He3'] \
        +params['P_DHe3_p']+params['P_DHe3_alpha']   
    params['P_net']=params['P_abs']                      ## net heating power (radiation corrected)



    ## tokamak/stellarator specifics
    if params['kind']=='tokamak':
        ## kappa_s is the elongation
        params['a_eff']=params['a']*np.sqrt(params['kappa_s']) #[m] effective minor radius
        params['q_95']=calc_q95(params)
        ## We take the definition from Miyamoto (textbook) to get the circumference of the plasma boundary 
        ## Miamoto states: the length of the circumference is 2piGa with G=np.sqrt((1+params['kappa_s']**2)/2).
        ## Hence, the surface area is given by 2piR * 2piGa = 4pi**2 R a G
        ## --> S= 4 pi**2 R a G
        
        ## We define S= 4 pi**2 R aeff *K --> a G = aeff* K
        ## --> K = G * a /aeff
        ## --> K = G * a/ a*sqrt(kappa_s) = G /sqrt(kappa_s)
        ## --> K = np.sqrt((1+params['kappa_s']**2)/(2* kappa_s).
        params['K']=np.sqrt((1+params['kappa_s']**2)/(2*params['kappa_s']))
        print('Plasma shaping factor K: %.2f'%params['K'])
    else:
        params['q_95']=params['q_23']
        params['a']=params['a_eff']        
        
        

    ## define the radial grid for the profile calculation
    params['nrc']=140
    params['dra_c']=1./params['nrc']
    params['ra_c']=np.arange(params['nrc'])*params['dra_c'] +0.5*params['dra_c']
    params['rmin_arr_c']=params['a_eff']*params['ra_c'] #grid center positions
            
        
    ## The surface is 2piR * 2pi K aeff = 4pi**2 R aeff K
    params['surf']=4.*np.pi**2 * params['R'] * params['rmin_arr_c']*params['K']  
    params['plasma_surface']=4.*np.pi**2 * params['R'] * params['a_eff']*params['K']

    
    if not 'first_wall_surface' in params:    
        print('Set the first wall surface area to be equal to the plasma surface area --> needed for the NWL calculation')
        params['first_wall_surface'] = params['plasma_surface']
    
    params[ 'V'         ] =  2.*np.pi**2 * params['R'] * params['a_eff']**2
    vol= 2.*np.pi**2 * params['R'] * (params['a_eff']*(params['ra_c']+0.5*params['dra_c']))**2
    params['dvol']=np.diff(np.r_[0,vol])
    

    if not 'n_0' in params:  
        params['n_0']=1.0
    params['ne_c'] = radial_profile(params['ra_c'],params['n_0'],params['mu_n'],params['offset_n'],params['slope_n'])  #[10^20m^-3]


    ##-------------------------------------------------------------------------
    ## define the profile shape of the heat diffusivity profile
    ##-------------------------------------------------------------------------
    params['chi_profile_shape']=(1+params['ra_c']**3)

    ##-------------------------------------------
    ## read cooling rates and main charge
    ##-------------------------------------------
    ## Hydrogen Isotope Cooling rates
    directory= os.path.dirname(__file__) + '/Tables_and_Data/'
    params['T_rates']= np.loadtxt(directory+'lz_H_puetti_bolo.dat', skiprows = 1, unpack = True)
    params['H_rates']= np.loadtxt(directory+'lz_H_puetti_bolo.dat', skiprows = 1, unpack = True)
    params['D_rates']= np.loadtxt(directory+'lz_H_puetti_bolo.dat', skiprows = 1, unpack = True)
 
    ## Helium Cooling rates
    params['He3_rates']= np.loadtxt(directory+'lz_He_puetti_bolo.dat', skiprows = 1, unpack = True)
    params['He4_rates']= np.loadtxt(directory+'lz_He_puetti_bolo.dat', skiprows = 1, unpack = True)

    ## Charge and mass definitions
    params['species_names']=    ['D','T','H','He4','He3']
    params['species_charge']=   [1., 1., 1.,   2.,   2.]
    params['species_mass']=     [2., 3., 1.,   4.,   3.]
    params['fraction']=         [0., 0., 0.,   0.,   0.]
    
    
    if params['plot_cooling_rates']==True:
        plt.figure()
        plt.grid()
    ## fetch the cooling rates
    if 'W_fraction' in params:
        ## W Cooling rates
        params['W_rates']= np.loadtxt(directory+'lz_W_puetti_bolo.dat', skiprows = 1, unpack = True)
        if params['plot_cooling_rates']==True:
            plt.loglog(params['W_rates'][0]/1.e3, params['W_rates'][1],label='W')
        params['species_names'].append('W')
        params['species_charge'].append(46.) ## use some lower change state that is more reasonalbe
        params['species_mass'].append(184.)
        params['fraction'].append(params['W_fraction'])
    if 'Fe_fraction' in params:
        ## Fe Cooling rates
        params['Fe_rates']= np.loadtxt(directory+'lz_Fe_puetti_bolo.dat', skiprows = 1, unpack = True)    
        if params['plot_cooling_rates']==True:
            plt.loglog(params['Fe_rates'][0]/1.e3, params['Fe_rates'][1],label='Fe')  
        params['species_names'].append('Fe')
        params['species_charge'].append(26.)
        params['species_mass'].append(56.)  
        params['fraction'].append(params['Fe_fraction'])
    if 'N_fraction' in params:        
        ## Nitrogen Cooling rates
        params['N_rates']= np.loadtxt(directory+'lz_N_puetti_bolo.dat', skiprows = 1, unpack = True)
        if params['plot_cooling_rates']==True:
            plt.loglog(params['N_rates'][0]/1.e3, params['N_rates'][1],label='N')       
        params['species_names'].append('N')
        params['species_charge'].append(7.)
        params['species_mass'].append(14.)  
        params['fraction'].append(params['N_fraction'])
    if 'C_fraction' in params:
        ## Carbon Cooling rates
        params['C_rates']= np.loadtxt(directory+'lz_C_puetti_bolo.dat', skiprows = 1, unpack = True)
        if params['plot_cooling_rates']==True:
            plt.loglog(params['C_rates'][0]/1.e3, params['C_rates'][1],label='C')
        params['species_names'].append('C')
        params['species_charge'].append(6.)
        params['species_mass'].append(12.)   
        params['fraction'].append(params['C_fraction'])

    if params['plot_cooling_rates']==True: 
        ## plot Helium 
        plt.loglog(params['He4_rates'][0]/1.e3, params['He4_rates'][1],label=r'He')
        ## plot Hydrogen
        plt.loglog( params['H_rates'][0]/1.e3,  params['H_rates'][1],label='H')
        plt.xlabel(r'T$_e$ [keV]')
        plt.ylabel(r'L$_z$ [W cm$^3$]') 
        plt.legend(fontsize=14,loc=1)
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        plt.savefig('Plots/cooling_rates.eps')
        plt.savefig('Plots/cooling_rates.png')    
        
        
        
    params['nspecies']=len(params['species_names']) ## main ions, helium,impurities...
    params['species_charge']=np.array(params['species_charge'])
    params['species_mass']=np.array(params['species_mass'])
    params['fraction']=np.array(params['fraction'])
    
    
    
    ## ------------------------------------------------------------------------
    ## ion density array
    ## ------------------------------------------------------------------------    
    params['ion_dens_arr']=np.zeros((params['nspecies'],params['nrc']))
    ## add impurity densities (start at sixth element as there is T,D,H,He4,He3 always present)
    for iion in range(5,params['nspecies']):
        params['ion_dens_arr'][iion,:]=params['ne_c'] * params['fraction'][iion]

    ## ------------------------------------------------------------------------
    ## consider the fuelling
    ## ------------------------------------------------------------------------
    ## add the main ion density
    ## here, we use matrix-multiplication to calulate something like 
    ##nMain=params['ne_c'] - params['nHe_c']*2. - params['nC_c']*6. - params['nFe_c']*26. 
    impurity_index=np.arange(params['nspecies']-3)+3
    ## pure D fueling:
    if params['fueling_mix'][0] ==1:
        index=np.append([1,2],impurity_index)
        main_ion_density= params['ne_c']-np.matmul(np.transpose(params['ion_dens_arr'][index,:]), params['species_charge'][index]) 
        params['ion_dens_arr'][0,:]=main_ion_density
    ## pure H fueling
    elif params['fueling_mix'][2] == 1: 
        index=np.append([0,1],impurity_index)
        main_ion_density= params['ne_c']-np.matmul(np.transpose(params['ion_dens_arr'][index,:]), params['species_charge'][index]) 
        params['ion_dens_arr'][2,:]=main_ion_density
    ## D-T fuelling
    elif (params['fueling_mix'][0] > 0) & (params['fueling_mix'][1] > 0) & (params['fueling_mix'][2] == 0): 
        index=np.append([2],impurity_index)
        main_ion_density= params['ne_c']-np.matmul(np.transpose(params['ion_dens_arr'][index,:]), params['species_charge'][index]) 
        params['ion_dens_arr'][0,:]=   params['fueling_mix'][0]*main_ion_density ## D density
        params['ion_dens_arr'][1,:]=   params['fueling_mix'][1]*main_ion_density ## T density
    
    ## ------------------------------------------------------------------------
    ## total ion density
    ## ------------------------------------------------------------------------     
    params['nIon_c']=np.sum(params['ion_dens_arr'][:,:],axis=0)
    
    ## ------------------------------------------------------------------------
    ## calculate zeff
    ## ------------------------------------------------------------------------
    ## here, we use matrix-multiplication to calulate something like     
    ## zeff_prof=(params['nMain_c'] + params['nHe_c']*2.**2 + params['nC_c']*6.**2 + params['nFe_c']*26.**2)/params['ne_c']
    zeff_prof= np.matmul(np.transpose(params['ion_dens_arr'][:,:]), params['species_charge']**2) /params['ne_c']
    params['Zeff']=calc_volume_average(zeff_prof,params) 
    if params['verbose']:
        print('Initial Zeff: %.2f'%params['Zeff'])

    ## ------------------------------------------------------------------------
    ## calcualte average main ion mass: 
    ## ------------------------------------------------------------------------
    params['M'] = calc_volume_average(np.matmul(np.transpose(params['ion_dens_arr'][0:3,:]),\
                params['species_mass'][0:3])/np.sum(params['ion_dens_arr'][0:3,:],axis=0),params)
    print('average main-ion mass at start: %.2f'%params['M'])

    ## ------------------------------
    ## ---- STRAHL grid
    ## ------------------------------
    from strahl_solver import define_strahl_grid
    params['grid']=define_strahl_grid(rminor=params['a_eff']*100.,rmajor=params['R']*100.,dbound=10.,dr_0 = 5., dr_1 = 5.)


    ## ------------------------------
    ## ---- FIDASIM colrad table ----
    ## ------------------------------
    if params['P_nbi'] > 0:
        ## determine the injection angle of NBI
        if not 'NBI_gamma' in params:   
            params['NBI_gamma'] = np.deg2rad(60.) ## angle between NBI and Rmaj Tangency radius
        from calc_beam_deposition import find_nbi_angle_for_given_gamma
        xpos_source=2*params['R'] ## same definition in calc_beam_deposition
        params['NBI_alpha_angle']  =find_nbi_angle_for_given_gamma(xpos_source,params['NBI_gamma'],params['R'])


        ## load tables for the beam attenuation code
        from hdf5 import load_dict
        params['tables'] = load_dict(directory+'fidasim_tables_HeCFe.hdf5')
        ## ------------------------------
        ## ---- pyFIDASIM inputs
        ## ------------------------------
        if 'use_pyfidasim' in params and params['use_pyfidasim'] == True:
            del params['NBI_gamma']
            del params['NBI_alpha_angle']

            if 'collect_pyfidasim_inputs' in params:
                params=params['collect_pyfidasim_inputs'](params,doplt=True)
            else:
                from calc_beam_deposition_pyfidasim import collect_pyfidasim_inputs
                params=collect_pyfidasim_inputs(params,doplt=True)
    params['particle_source_nbi']=np.zeros(params['nrc'])

    ## power densities:
    ## note that we use capital letters for the overal power and small ones for the power density
    params['pe_ecrh']=calc_ecrh_power_density(params)
    params['pi_nbi']=np.zeros(params['nrc'])
    params['pe_nbi']=np.zeros(params['nrc'])
    params['ionization_power']=np.zeros(params['nrc'])
    params['prad']=np.zeros(params['nrc'])
    params['pei']=np.zeros(params['nrc'])
    params['cx_power']=np.zeros(params['nrc'])
    
    ## D-T fusion
    params['P_DT_fusion']=0
    params['P_DT_n']     =0
    params['pi_DT_alpha']=np.zeros(params['nrc'])
    params['pe_DT_alpha']=np.zeros(params['nrc'])
    ## D-D fusion
    params['P_DD_fusion']=0
    params['P_DD_n']     =0
    params['pi_DD_p']=np.zeros(params['nrc'])
    params['pe_DD_p']=np.zeros(params['nrc'])
    params['pi_DD_T']=np.zeros(params['nrc'])
    params['pe_DD_T']=np.zeros(params['nrc'])
    params['pi_DD_He3']=np.zeros(params['nrc'])
    params['pe_DD_He3']=np.zeros(params['nrc'])
    ## D-He3 fusion
    params['P_DHe3_fusion']=0
    params['pi_DHe3_p']=np.zeros(params['nrc'])
    params['pe_DHe3_p']=np.zeros(params['nrc'])
    params['pi_DHe3_alpha']=np.zeros(params['nrc'])
    params['pe_DHe3_alpha']=np.zeros(params['nrc'])
    return(params)



## --------------------------------
## --------------------------------
## ------ Empirical scalings ------ 
## --------------------------------
## --------------------------------
def calc_tauE(params):
    ## determine the 0D energy confinment time
    if params['kind']== 'stellarator':
        #energy confinement time according to ISS04 scaling law
        tau_E=(0.134 * params['a_eff']**2.28
                * params['R']**0.64
                * params['B_tor']**0.84
                * params['P_net']**-0.61 ## net heating power
                # * np.sign(params['P_net']) * (np.abs(params['P_net'])) ** (-0.61)
                * (n_lineavg(params)*10.)**0.54
                * params['q_23']**-0.41 )
        
    elif params['kind']== 'tokamak':
        #energy confinement time according to ITER98(y,2) scaling law
        tau_E= (0.144 * params['I_p']**0.93
                * params['B_tor']**0.15
                * params['P_net']**-0.69    ## net heating power
                * n_lineavg(params)**0.41
                * params['M']**0.19
                * params['R']**1.97
                * (params['a']/params['R'])**0.58
                * params['kappa_s']**0.78)
    else:
        raise Exception('device type not defined -- either stellarator or tokamak')
    return(tau_E*params['H_factor'])
    
def calc_density_limit(params):
    ## Density Limit Scalings
    if params['kind']== 'stellarator':
        ## determine the SUDO density limit (for stellarators)
        return ( 0.25 * np.sqrt( params['P_net']*params['B_tor'] 
                                /(params['a_eff']**2*params['R'])))
    elif params['kind']== 'tokamak':
        ## determine the Greenwald density limit (for tokamaks)
        return ( params['I_p'] / (np.pi * params['a']**2) )

     
## ----------------------------------------
## -------- Stored energy and pressure ----
## ----------------------------------------         
def calc_stored_energy(params):
    ## calculate the stored energy W = P_net * tau_e
    return(params['P_net']* 1e6 * calc_tauE(params)) ## in Joules

def calc_average_pressure_from_stored_energy(params):
    ## Calcualte the plasma pressure [Pascals] from stored energy
    p_avg=1./params['V'] * 2./3. *  calc_stored_energy(params)  ## Pa (eV*consts.e *m^-3)
    return(p_avg)


## ----------------------------------------
## -------- Profile routines --------------
## ----------------------------------------
def radial_profile( ra, f_0=1., mu=1. ,offset=0.,slope=0.):
    ## define a fusion-type radial profile    
    return ( (f_0-offset-slope) * (1.-ra**2)**mu + offset +(1-ra)*slope)

def calc_average_pressure_from_profiles(params):
    ## Calcualte the plasma pressure [Pascals] from profiles
    pe_avg=np.sum(params['ne_c']*1.e20*params['Te_c']*1.e3*(consts.e)*params['dvol'])/params['V']
    pi_avg=np.sum(params['nIon_c']*1.e20*params['Ti_c']*1.e3*(consts.e)*params['dvol'])/params['V']  
    return(pe_avg,pi_avg)
 
def n_lineavg(params):
    ## calculate the line average density
    line_avg=np.sum(params['ne_c']*params['dra_c'])
    return(line_avg)

def calc_volume_average(F,params):
    '''
    calculate the volume average of array F
    '''
    F[np.isnan(F)]=0
    average=np.sum(F*params['dvol'])/params['V']    
    return(average)


## -----------------------------------------------
## --------- ECRH routines -----------------------
## -----------------------------------------------    
def calc_ecrh_power_density(params):
    ## Consider ECRH depositin within r/a =0.2
    index=params['ra_c']<0.05
    pe_ecrh=np.zeros(params['nrc'])
    pe_ecrh[index]+=params['P_ecrh']/np.sum(params['dvol'][index]) #[MW/m^3]
    return(pe_ecrh)
    
def electron_cyclotron_freq( params ):
    ## calculate the electron cyclotron frequency
    return ( consts.e*params['B_tor'] /consts.m_e/(2.*np.pi))

def calc_cutoff_O( params, harmonic=1. ):
    ## ECRH O-mode cutoff 
    return ( 9.65e18 * harmonic**2 * params['B_tor']**2 )

def calc_cutoff_X( params, harmonic=1. ):
    ## ECRH X-mode cutoff 
    return ( 9.65e18 * harmonic*(harmonic-1) * params['B_tor']**2 )  


## ---------------------------------------------------------
## ----- fast-ion and alpha slowing down and pressure ------
## ---------------------------------------------------------   
def calc_coulomb_logarithm_NRL(ne,Te):
    '''
    This one assumes a thermal species
    calculation of the electron-ion coulomb logarithm using the NRL formulary (page 34) 
    This is valid for Ti *(m_e/m_i) < 10*Z^2 < Te
    inputs: ne: density in 10^20/m^-3
            Te: Temperature ion keV
            
    output: Coulomb logarithm []
    '''   
    ## input ne in 10^20/m^-3 
    ## --> ne*=1.e20 [m^-3]
    ## --> ne/=1.e6  [cm^-3]
    ##                                   cm^-3  [eV]
    coulomb_logarithm=24.-np.log(np.sqrt(ne*1.e14)/(Te*1.e3))         
    return(coulomb_logarithm)

def calc_coulomb_logarithm_Sauter(ne,Te):
    '''
    Coulomb logarithm as provided by O. Sauter, Physics of Plasmas 6, 2834 (1999);
    inputs: ne: density in 10^20/m^-3
            Te: Temperature ion keV
            
    output: Coulomb logarithm []
    '''                                               # m^-3  # eV    
    coulomb_logarithm= 31.3 - np.log(np.sqrt(ne*1.e20)/(Te*1.e3))
    return(coulomb_logarithm)

    
def calc_ii_coulomb_logarithm_Sauter(Z_i,Zeff,ni,Ti):
    '''
    Ion-Ion Coulomb logarithm as provided by O. Sauter, Physics of Plasmas 6, 2834 (1999);
    input: 
        Ion charge: Zi []
        Effective charge: Zeff
        Ion density: ni  [m^-3]
        Ion temperature: Ti [eV]
    output: 
        Coulomb logarithm []
    '''   
    coulomb_logarithm= 30 - np.log((Z_i*Zeff)**1.5*np.sqrt(ni)/Ti**1.5) 
    return(coulomb_logarithm)
       
def calc_coulomb_logarithm_weiland(params,energy,mass,charge):
    '''
    This one considers fast-ions
    calculation of the ion-ion coulomb logarithm based on the formulas of Weiland's Rabbit paper 2018    
    # Here, T and E are in keV and n are in m−3.
    #Ze = −1 and Ae = 1/1836.1 are supposed to be used for the electrons.
    ## see also https://w3.pppl.gov/~hammett/work/collisions/coulog.for 
    '''
    ##-------------------------------------------------------------------------
    ## determine rmax
    ##-------------------------------------------------------------------------    
    ## electrons
    A=1/1836.1
    omega2= 1.74 /A *params['ne_c']*1.e20  + 9.18e15 /A**2 *params['B_tor']**2
    vrel2_max = 9.58e10 * (params['Te_c']/A + 2*energy/mass)       
    denominator=omega2/vrel2_max    
    ## ions
    for i in range(params['nspecies']):
        omega2= 1.74* params['species_charge'][i]**2/params['species_mass'][i]*params['ion_dens_arr'][i,:]*1.e20 \
            + 9.18e15 * params['species_charge'][i]**2/params['species_mass'][i]**2 *params['B_tor']**2 
        vrel2_max = 9.58e10 * (params['Ti_c'][:]/params['species_mass'][i] + 2*energy/mass)              
        denominator+=omega2/vrel2_max
            
    rmax=np.sqrt(1./denominator)
    
    ##-------------------------------------------------------------------------
    ## determine rmin and coulomb logarithm
    ##-------------------------------------------------------------------------      
    ## electrons
    A=1/1836.1
    Z=-1.     
    vrel2_min= 9.58e10 *(3*params['Te_c']/A + 2*energy/mass)
    rmin_cl = 0.13793 * np.abs(Z * charge) * (A + mass)/(A*mass *vrel2_min)
    rmin_qu= 1.9121e-8 * (A + mass)/(A*mass*np.sqrt(vrel2_min))
    ## select the maximum rmin between rmin_cl and rmin_qu
    rmin = np.amax([rmin_cl, rmin_qu],axis=0) 
    ## electron coulomb logarithm
    e_coulomb_logarithm=np.log(rmax/rmin)
    ## ions
    ion_coulomb_logarithm=np.zeros((params['nspecies'],params['nrc']))
    for i in range(params['nspecies']):    
        A= params['species_mass'][i]
        Z= params['species_charge'][i]
        vrel2_min= 9.58e10 *(3*params['Ti_c']/A + 2*energy/mass)
        rmin_cl = 0.13793 * np.abs(Z * charge) * (A + mass)/(A*mass *vrel2_min)
        rmin_qu= 1.9121e-8 * (A + mass)/(A*mass*np.sqrt(vrel2_min))        
        ## select the maximum rmin between rmin_cl and rmin_qu
        rmin = np.amax([rmin_cl, rmin_qu],axis=0)     
        ion_coulomb_logarithm[i,:]=np.log(rmax/rmin)
    
    return(e_coulomb_logarithm, ion_coulomb_logarithm)

def calc_tau_spitzer(params,energy,mass,charge):
    ## calculate the spitzer time (NRL)
    coulomb_logarithm, dummy = calc_coulomb_logarithm_weiland(params,energy,mass,charge)
    ##                                                              #[eV]                     cm^-3    
    tau_s= 6.27e8 * mass/(charge**2*coulomb_logarithm) * (params['Te_c']*1.e3)**1.5 /(params['ne_c']*1.e14)
    return (tau_s)

def calc_critical_energy(params,energy,mass,charge):
    ## inputs
    ## energy [keV]
    ## mass in [amu]   
    ## charge in [C]      
    lambda_e,lambda_arr =calc_coulomb_logarithm_weiland(params,energy,mass,charge)
    '''    
    ## Determine the critical energy for below which fast ions mainly heat the ions.
    charge_mass_ratio=(params['ion_dens_arr'][0,:]*1**2   /params['M'] *lambda_main  +\
                       params['ion_dens_arr'][1,:]  *2**2   /4           *lambda_He    +\
                       params['ion_dens_arr'][2,:]   *6**2   /12          *lambda_C     +\
                       params['ion_dens_arr'][3,:] *26**2  /56          *lambda_Fe     )/(params['ne_c']*lambda_e)
  
    '''    
    charge_mass_ratio = np.matmul(np.transpose(params['ion_dens_arr']*lambda_arr), params['species_charge']**2/params['species_mass'])/(params['ne_c']*lambda_e)
    ## according to Weiland 2018 (Rabbit paper), the critical velocity is:
    ## vc=5.33*1.e4 *sqrt(Te(eV)) * <Zi**2/Ai>^1/3  # m/s
    ## This means, the critical energy becomes:
    ## Ec=0.5* mass * (5.33*1.e4 *sqrt(Te) * <Zi**2/Ai>^1/3)^2 ## [J]
    ## Ec= 14.20445e4 * mass*consts.amu *Te * <Zi**2/Ai>^2/3 ## [J]
    ## Ec= 14.20445e4 * mass*consts.amu *Te * <Zi**2/Ai>^2/3 /consts.e ## [eV]
    ## Ec= 14.7218 * mass *Te * <Zi**2/Ai>^2/3 ## [eV]        
    Ec=14.7218*params['Te_c']*mass*charge_mass_ratio**(2/3)
    return(Ec)

    
def calc_tau_sd(params,energy,mass,charge):    
    ## inputs
    ## energy [keV]
    ## mass in [amu]    
    ## charge in [C]     
    ## calculate the slowing down time
    Ec=calc_critical_energy(params,energy,mass,charge)
    tau_sd=calc_tau_spitzer(params,energy,mass,charge)/3. *np.log(1. +(energy/Ec)**1.5)
    return(tau_sd)
 
def Gfunc(params,energy,mass,charge):
    ## function to determine the fraction of the slowing down heating power going into ions
    Ec=calc_critical_energy(params,energy,mass,charge)
   
    y=energy/Ec
    integral = 0.33333333*(-2*np.log(np.sqrt(y)+1.) + np.log(y-np.sqrt(y)+1.) \
                           + 2*np.sqrt(3.)*np.arctan((2*np.sqrt(y)-1)/np.sqrt(3.))) \
                           + np.sqrt(3.)*np.pi/9.
    return(integral/y)    
 
    

## ----------------------------------------------
## ---------- Electron ion exchange terms -------
## ----------------------------------------------  
def calc_tau_ei(params):
    ## -- calcualte electron-ion exchange time based on Wesson eq. 2.14.10: --   
    ## Tau_ei= 3 * sqrt(2) *np.pi**1.5 * consts.epsilon_0**2 * m_i *m_e * (T_i/m_i + T_e/m_e)**1.5 /\
    ## (n_e * e^4 * Z_i**2*Z_e**2 * lambda )   
    ## -- Let's assume Te/me >> Ti/m_i and Z_e=1 --
    ## Tau_ei= 3 * sqrt(2) *np.pi**1.5 * consts.epsilon_0**2 * m_i *m_e * (T_e/m_e)**1.5 /\
    ## (n_e * e^4 * Z_i**2 * lambda )    
    ## -- This can be reduced to: --
    ## Tau_ei= 3 * 2**1/2**1 * 2**0.5 *np.pi**1.5 * consts.epsilon_0**2 * m_i *m_e**-0.5 * T_e**1.5 /\
    ## (n_e * e^4 * Z_i**2 * lambda )   
    ## -- and further to: --
    ## Tau_ei= 3/2  * (2*np.pi)**1.5 * \
    ## (consts.epsilon_0**2 * m_i * T_e**1.5) /\
    ## (sqrt(m_e) * n_e * e^4 * Z_i**2 * lambda )        
    Te_avg=calc_volume_average(params['Te_c'],params) ## [keV]
    ne_avg=calc_volume_average(params['ne_c'],params) ## [10^20/m^3]
    coulomb_logarithm=calc_coulomb_logarithm_NRL(ne_avg,Te_avg)
    m_i=params['M']*consts.atomic_mass ## ion mass [kg]
    Z=1
    tau_ei= 3/2  * (2*np.pi)**1.5 * \
            (consts.epsilon_0**2 * m_i * (Te_avg*1.e3*consts.e)**1.5) / \
            (np.sqrt(consts.m_e) * ne_avg*1.e20 * consts.e**4 * Z**2 *  coulomb_logarithm)    
    return(tau_ei)
    
def calc_pei(params):
    '''
    Electron-ion heat exchange term calculation
    '''
    ## See Wesson, page 93 (Braginski equation section)
    ## T in keV
    ## ni in m^-3   
    ## pei= 3 me/mi * ne/tau_e * (Te-Ti) [MW/m^3]    
    ## electron exchange time
    ## tau_e = 1.09e16 * Te**(3/2) /(ni * Z**2 * coulomb_log) [s]
    
  
    ## NRL page 36 + 37
    ## Te in keV
    ## ni in 10^20 m^-3
    ## -------------------------
    ## electron collisional time:
    ## -------------------------    
    ## tau_e=3.44e5 * (Te*1.e3)**(3/2)/(ni*1.e14 * coulomb_log)
    ## --> same as Wesson for Z=1
    ## -------------------------
    ## heat exchange
    ## -------------------------      
    ##                   cm^-3    J/eV 1/s     eV      = W/cm^3 = MW/m^3
    ## pei= 3 me/mi * (ne*1.e14)*k/tau_e * (Te-Ti)*1.e3
    ## k= 1.60e-12 erg/eV = 1.6e-12*1.e-7 J/eV = 1.6-19 J/eV = consts.e
    ## play around with constants:
    ## pei = 3* consts.m_e/consts.atomic_mass * consts.e * 1/Ai * (ne*1.e14) * 1/(3.44e5 * (Te*1.e3)**(3/2)/(ni*1.e14 *Z**2 * coulomb_log)) * (Te-Ti)*1.e3
    ## pei = 3* consts.m_e/consts.atomic_mass * consts.e * 1/3.44e5  (ne*1.e14) * (ni*1.e14) *Z**2 * coulomb_log * (Te-Ti)*1.e3  /Ai  /(Te*1.e3)**(3/2)
    ## pei = 3* consts.m_e/consts.atomic_mass * consts.e * 1/3.44e5 * 1.e14* 1.e14/1.e3**1.5*1.e3 *ne*ni *Z**2* coulomb_log * (Te-Ti)  /Ai  /(Te)**(3/2)      
    ## --> pei=0.2424 *ne*ni *Z**2 * coulomb_log * (Te-Ti)  /Ai  /(Te)**(3/2) [MW/m^3]    
    coulomb_logarithm=calc_coulomb_logarithm_NRL(params['ne_c'],params['Te_c'])
       
    pei=0.2424*coulomb_logarithm*params['ne_c']*(params['Te_c']-params['Ti_c'])/params['Te_c']**(1.5) *\
        np.matmul(np.transpose(params['ion_dens_arr']),params['species_charge']**2/params['species_mass'])
    return(pei)


def calc_p_sync(params):
                
    #phi_reabsorbed=0.99
    #p_sync = .327*params['B_tor']**2*params['ne_c']*params['Te_c']*(1-phi_reabsorbed)
    a_wall=params['first_wall_surface']/(4.*np.pi**2*params['R'])
    R=params['R_sync']


    p_sync= 1.28e-4*params['ne_c']**0.5*(params['Te_c']/10.)**2.5*a_wall**-0.5*params['B_tor']**2.5 \
        *(1+5.7/(params['R']/params['a_eff']*(params['Te_c']/10.)**0.5))**0.5*(1-R)**0.5
    # P_sync_rel = .327*Mag**2*n*T_arr*phi*(spc.kv(3, 511/T_arr)/spc.kv(2, 511/T_arr))
    # ax.loglog(T_arr, P_sync, figure=fig, label='classical')
    # ax.loglog(T_arr, P_sync_rel, figure=fig, label='relativistic')
    # ax.legend(fontsize = 14)
    #plt.figure()
    #plt.plot(p_sync)
    
    #P_sync = np.sum(p_sync *params['dvol']) #[MW]
    #print('Synchrotron power:',P_sync)
    return(p_sync)
## --------------------------------------------------------
## --------- calculation of power losses ------------------
## --------------------------------------------------------    
def calc_radiative_power(params):   
    '''
    Calculate Bremsstrahlung and line radiation for the different species considered
    This is based on the cooling factors provided by Th. Puetterich.
    '''
    ## determine the radiation density
    prad_density=np.zeros(params['nrc'])
    for iion in range(params['nspecies']):
        name=params['species_names'][iion]+'_rates'
        prad_density+=1.e28*params['ne_c']*params['ion_dens_arr'][iion,:]*np.interp(params['Te_c']*1.e3, params[name][0],params[name][1]) # [MW*m^3]
    ## add synchrotron radiation
    if params['R_sync'] < 1:
        prad_density+=calc_p_sync(params)
    return(prad_density)

      
## -----------------------------------------------------
## -------- Heat flux and diffusivities ----------------
## -----------------------------------------------------
def calc_qi(params): 
    ## ----------------------------
    ## Determine the ion heat flux
    ## ----------------------------    
    qi=np.cumsum((params['pi_nbi']+params['pei']+params['pi_DT_alpha'] \
                  +params['pi_DD_p']+params['pi_DD_T']+params['pi_DD_He3'] \
                  +params['pi_DHe3_p']+params['pi_DHe3_alpha'] \
                  -params['cx_power'])*params['dvol'])/params['surf'] ## MW/m^2
    return(qi)   
  
def calc_qe(params): 
    ## ---------------------------------  
    ## Determine the electron heat flux
    ## ---------------------------------     
    qe=np.cumsum((params['pe_ecrh']+params['pe_nbi']+params['pe_DT_alpha'] \
                  +params['pe_DD_p']+params['pe_DD_T']+params['pe_DD_He3'] \
                  +params['pe_DHe3_p']+params['pe_DHe3_alpha'] \
                  -params['pei']-params['ionization_power']-params['prad'])*params['dvol'])/params['surf'] ## MW/m^2
    return(qe)       
    
def calc_chi_e(params):  
    ## Calculate electron diffusivity
    qe=calc_qe(params) 
    ##    J/s     m^-2    m^3                  1/J      m^1    
    chi_e=(qe*1.e6)/(-params['ne_c']*1.e20*np.gradient(params['Te_c']*1.e3*consts.e,params['rmin_arr_c'])) # m^2/s
    ## make sure that chi_e is not getting too small.
    chi_e[chi_e<0.01]=0.01
    return(chi_e)
        
def calc_chi_i(params):
    ## Calculate Ion diffusivity
    qi=calc_qi(params)
    ##    J/s     m^-2    m^3                  1/J      m^1    
    chi_i=(qi*1.e6)/(-params['nIon_c']*1.e20*np.gradient(params['Ti_c']*1.e3*consts.e,params['rmin_arr_c'])) # m^2/s 
    ## make sure that chi_e is not getting too small.
    chi_i[chi_i<0.01]=0.01  
    return(chi_i)


## --------------------------------------------------------------------
## --------------------------------------------------------------------
## --------------- calculate the expected temperature profile ---------
## --------------------------------------------------------------------    
## --------------------------------------------------------------------
def calc_ti_from_chi(params,chi):
    ## Determine the Ion tempeature gradient and temperature profile
    qi=calc_qi(params) ## ion heat flux [MW/m^2]
    qi[qi<0]=0    
    dTdr=-(qi*1.e6/consts.e)/chi/(params['nIon_c']*1.e20) ## temperature derivative
    ti=np.zeros(params['nrc'])
    for i in range(params['nrc']):
        index=np.arange(i+1)
        ti[i]=np.trapz(dTdr[index],params['rmin_arr_c'][index])
    ti-=ti[-1] ## make it zero at r/a=1
    ti/=1.e3 ## convert to keV
    ti+=params['offset_T'] ## add offset       
    return(ti)
    
def calc_te_from_chi(params,chi):
    ## Determine the Ion tempeature gradient and temperature profile
    qe=calc_qe(params) ## electron heat flux [MW/m^2]
    qe[qe<0]=0
    
    dTdr=-(qe*1.e6/consts.e)/chi/(params['ne_c']*1.e20) ## temperature derivative
    te=np.zeros(params['nrc'])
    for i in range(params['nrc']):
        index=np.arange(i+1)
        te[i]=np.trapz(dTdr[index],params['rmin_arr_c'][index])
    te-=te[-1] ## make it zero at r/a=1
    te/=1.e3 ## convert to keV
    te+=params['offset_T'] ## add offset
    return(te)


def calc_fusion_rate(name,params):
    ''' 
    Input:
        name: fusion reaction formula
        params: dictionary
    Output:
        fusion reaction rate
    '''
    if name == 'D(T,n)He4':
        ## DT fusion
        dens1=params['ion_dens_arr'][0,:]*1.e20 ## Deterium density   /m^3   
        dens2=params['ion_dens_arr'][1,:]*1.e20 ## Tritium density /m^3
    elif (name == 'D(D,p)T') or (name == 'D(D,n)He3'):
        ## DD fusion
        dens1=params['ion_dens_arr'][0,:]*1.e20 ## Deterium density   /m^3   
        dens2=params['ion_dens_arr'][0,:]*1.e20 ## Deterium density   /m^3         
    elif name == 'D(He3,p)He4':
        ## DHe3 fusion
        dens1=params['ion_dens_arr'][0,:]*1.e20 ## Deterium density   /m^3   
        dens2=params['ion_dens_arr'][4,:]*1.e20 ## He3 density /m^3  
    else:
        raise Exception('Reaction name in calc_fusion_rate not well defined.')        
    ## perform a cubic interpolation to find the sigma-v values
    from scipy.interpolate import interp1d   
    f = interp1d(params['log10_Ti_arr_sigma_v'],np.log10(params['sigma_v_'+name]), kind='cubic', fill_value='extrapolate')
    sigma_v=10**f(np.log10(params['Ti_c']))
    fusion_rate=dens1*dens2*sigma_v #[1/m^3/s]
    fusion_rate[np.isnan(fusion_rate)]=0
    return(fusion_rate)


def get_heating_power_from_fusion_products(reaction_name,fusion_rate,params):
    '''
    Input:
        parameters  [dictionary]
        reaction_name: string describing the reaction
        fusion_rate [1/s]
    Output:
        smoothed heating power
        smoothed ion and electron power densities
    '''  
    energy=params[reaction_name+'_energy']
    initial_power=params['P_'+reaction_name]
        
    if reaction_name == 'DHe3_p':
        mass=1.
        charge=1.
    elif reaction_name == 'DHe3_alpha':
        mass=4.
        charge=2.    
    elif reaction_name == 'DT_alpha':
        mass=4.
        charge=2. 
    elif reaction_name == 'DD_p':
        mass=1.
        charge=1. 
    elif reaction_name == 'DD_T':
        mass=3.
        charge=1. 
    elif reaction_name == 'DD_He3':
        mass=3.
        charge=2.            
    
    ## determine the power density
    power_density=fusion_rate * energy *consts.e # [MW/m^3]
    # account for fast-ion losses
    power_density *=(1.-params['f_alpha_loss']) 
    ## get the volume integrated power
    Power = np.sum(power_density *params['dvol']) #[MW]

    ## smooth the evolution of the total power
    # get the smooth factor
    tau_avg=np.sum(calc_tau_sd(params,energy*1.e3, mass,charge)*fusion_rate)/np.sum(fusion_rate)
    #print(tau_avg)
    smooth_factor=tau_avg/params['time_step']
    Power_smoothed = (smooth_factor * initial_power +  Power)/(smooth_factor+1)   
    #print('1',Power_smoothed)        
    ## calcualte the electron and ion heating fractions
    fraction=Gfunc(params,energy*1.e3,mass,charge) 
    #print(fraction[0])
    power_density_ion     =  fraction     * power_density * Power_smoothed/Power
    power_density_electron= (1.-fraction) * power_density * Power_smoothed/Power  
    
    return(Power_smoothed, power_density_ion, power_density_electron)

def calc_fusion_product_density(params,initial_density,fusion_rate,doplt=False):

    ## ------------------------------------
    ## define the temporal simulation grid
    ## ------------------------------------
    nt=200 ## number of time points for the calculation
    time_arr=np.linspace(0,4.*params['time_step'],num=nt)
    nr=params['grid']['nr']
    
    ## define the particle flux profile
    if np.any(fusion_rate) > 0:
        alpha_birth_rate=np.interp(params['grid']['ra'],params['ra_c'],fusion_rate*(1.-params['f_alpha_loss'])/1.e6) #[1/s/cm^3]
        flx=np.zeros(nt)
        flx[0:int(nt/4)]=np.sum(alpha_birth_rate*params['grid']['dvol'])
        source=alpha_birth_rate/flx[0]
    else:
        flx=np.zeros(nt)
        source=np.zeros(nr)
        if not np.any(initial_density) >0:
            ## if the initial density is zero and the source is zero
            ## return zero (initial density)
            return initial_density

    # parallel loss time
    tau=np.ones(nr)*1.e99
    tau[params['grid']['ra']>1.]=0.00001
    
    nion=2
    # Set the initial temperature
    initial_dens = np.zeros((nr,nion))
    initial_dens[:,1]=np.interp(params['grid']['ra'],params['ra_c'],initial_density)*1.e20/1.e6


    ## define diffusion profile
    diffusion = np.interp(params['grid']['ra'],params['ra_c'],params['chi'])*1.e4/params['rho_He'] ## cm^2/s
    diffusion2d = np.outer(np.ones(nion),diffusion)
    ## define convection profile
    convection= np.zeros(params['grid']['nr'])
    #convection[params['grid']['ra']>0.7]=-1. # cm/s
    convection2d = np.outer(np.ones(nion),convection)# np.zeros((nion,nr))

    source_charge_state=1
    from strahl_solver import strahl_solver
    dens_arr =strahl_solver(nion,nr,nt,\
        diffusion2d,convection2d,\
        1./tau,\
        source,\
        np.zeros((nr,nion+1)),\
        np.zeros((nr,nion+1)),\
        params['grid']['rr'],params['grid']['pro'],\
        params['grid']['qpr'],params['grid']['der'],\
        time_arr,\
        flx,\
        initial_dens,source_charge_state)

    ## remove the charge dependence
    dens_arr=dens_arr[:,1,:]*1.e6/1.e20 ## 10^20/m^3
    index=np.arange(150,nt,dtype=int)

    if doplt:
        particle_count=np.sum(dens_arr[:,index]*np.outer(params['grid']['dvol'],np.ones(len(index))),axis=0)
        ben=np.polyfit(time_arr[index],np.log(particle_count),1)
        print('tau: \t\t%.3f'%(-1/ben[0]),'s')
        plt.figure()
        plt.grid()
        particle_count_full=np.sum(dens_arr[:,:]*np.outer(params['grid']['dvol'],np.ones(nt)),axis=0)
        plt.plot(time_arr,particle_count_full)
        plt.plot(time_arr[index], np.exp(time_arr[index]*ben[0]+ben[1]))
        plt.ylim(bottom=0)

        plt.ylabel('density [1/m^3]')
        plt.xlabel('time [s]')
        plt.tight_layout()

    return np.interp(params['ra_c'],params['grid']['ra'],dens_arr[:,int(nt/4)-1])
## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------
## --- Main routine to calculate the fusion plasma evolution
## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------    
def calc_profile_evolution(params,reduced_pECRH=-1,niter=2000): 
    ## main routine to calcualte the expected temperature profile considering
    ## fusion particle heating power, fuel dilution and radiation  
    
    
    ## ------------------------------------------------------------------------
    ## ----- make an initial guess of Te=Ti
    ## ------------------------------------------------------------------------
    ## For this, we employ a pre-define T shape and determine the height of the 
    ## temperature profile that fits the average pressure from the stored energy
    #<p> = 1/V * 2/3 * W
    p_avg=calc_average_pressure_from_stored_energy(params)/1.e3   ## kPa (keV*consts.e *m^-3)
    ## get the normalized core pressure
    dene=params['ne_c']*1.e20 #[m^-3] 
    deni=params['nIon_c']*1.e20 #[m^-3] ## all ions            
    ## Determine the shape of the temperature profile, normalized to one
    tprof_shape=radial_profile(params['ra_c'],f_0=1.,mu=1.) 
    ## Calculate the corresponding pressure            
    p_avg_prof=calc_volume_average((dene+deni)*tprof_shape*consts.e,params) 
    ## Scale the temperature profile by the ratio of the pressures
    params['Te_c']=tprof_shape* p_avg/p_avg_prof
    params['Te_c'][params['Te_c']<0.001]=0.001
    params['Ti_c']=params['Te_c']

    ## ------------------------------------------------------------------------
    ## Define array to store the evolution of parameters
    ## ------------------------------------------------------------------------
    ## temperature evolution storage
    params['Te_evol']=np.zeros(niter)
    params['Ti_evol']=np.zeros(niter)
    ## density evolution storage
    params['ne_evol']=np.zeros(niter)
    params['nT_evol']=np.zeros(niter)
    params['nD_evol']=np.zeros(niter)
    params['nH_evol']=np.zeros(niter)
    params['nHe4_evol']=np.zeros(niter)
    params['nHe3_evol']=np.zeros(niter)
    ## radiated power evolution storage
    params['Prad_evol']=np.zeros(niter)
    ## fusion power evolution storage
    params['P_DT_fus_evol']=np.zeros(niter)
    params['P_DD_fus_evol']=np.zeros(niter)
    params['P_DHe3_fus_evol']=np.zeros(niter)

    
    
    
    chi_fit_factor_guess=0.1 
    cc=0
    for i in range(niter):
        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        ## Calcualte a new Ti and Te profile
        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        # electron ion collisional exchange term power density profile  
        params['pei']=calc_pei(params)
        ## calculate the average pressure to be maintained by the Ti and Te profiles.
        params['p_avg']=calc_average_pressure_from_stored_energy(params)
        Te_old= np.copy(params['Te_c'])
        Ti_old= np.copy(params['Ti_c'])
        ## Use a least squares fit to find a Chi-profile that produces Te and Ti profiles
        ## that agree with params['p_avg']
        fit_param=np.zeros(1)
        fit_param[0]=chi_fit_factor_guess
        from scipy.optimize import least_squares
        def chi_square_fun(fit_param, params): 
            params['chi']=fit_param[0]*params['chi_profile_shape']
            params['Ti_c']=calc_ti_from_chi(params,params['chi'])
            params['Te_c']=calc_te_from_chi(params,params['chi'])
            if np.isnan(params['Ti_c']).any():
                params['Ti_c'][np.isnan(params['Ti_c'])]=0
                print('ben')
            if np.isnan(params['Te_c']).any():
                print('ben')
                params['Te_c'][np.isnan(params['Te_c'])]=0
            if np.isnan(params['chi']).any():
                print("Chi nan")
            if np.isnan(params['chi_profile_shape']).any():
                print("Chi profile shape nan")
            pe_avg,pi_avg=calc_average_pressure_from_profiles(params)
            target=(params['p_avg']-(pe_avg+pi_avg))**2
            return(target)
        
        fitresult = least_squares(chi_square_fun, fit_param,args = [(params)])
        chi_fit_factor_guess=fitresult['x'][0]
        smooth_factor=calc_tauE(params)/params['time_step']
        ## new Te profile
        params['Te_c']=(smooth_factor*Te_old + params['Te_c'])/(smooth_factor+1)
        params['Te_evol'][i]=params['Te_c'][0]
        ## new Ti profile
        params['Ti_c']=(smooth_factor*Ti_old + params['Ti_c'])/(smooth_factor+1)
        params['Ti_evol'][i]=params['Ti_c'][0]

        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        ## Calcualte the NBI heating power based on the new profiles
        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        if params['P_nbi'] >0:
            if 'use_pyfidasim' in params:
                if 'calc_beam_deposition' in params:
                    if not 'pyfidasim_nmarker' in params:
                        params['pyfidasim_nmarker'] = 15000
                    params['calc_beam_deposition'](params,nmarker=params['pyfidasim_nmarker'],doplt=False,monte_carlo=True)
                else:
                    from calc_beam_deposition_pyfidasim import calc_beam_deposition_pyfidasim
                    calc_beam_deposition_pyfidasim(params,monte_carlo=False,verbose=False)
            else:
                from calc_beam_deposition import calc_beam_deposition
                calc_beam_deposition(params,nmarker=1)

        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        ## ---- Determine fusion heating powers and neutron powers ------------
        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        fusion_rate_DT=0.
        fusion_rate_DHe3=0.
        fusion_rate_DD_pT=0. 
        fusion_rate_DD_nHe3=0.
        ## check if there is deterium in the plasma
        if np.sum(params['ion_dens_arr'][0,:]) >0.:       
            ## ----------------------------------------------------------------
            ## ---- D + T -> n + alpha  ---------------------------------------
            ## ----------------------------------------------------------------
            ## check if there is some tritium in the simluation
            if np.sum(params['ion_dens_arr'][1,:]) >0.:
                ## calculate DT fusion reaction rate
                fusion_rate_DT=calc_fusion_rate('D(T,n)He4',params)
                ## - - - - - - - - - - - - - - - - - - - -
                ## power going into the DT neutrons
                ## - - - - - - - - - - - - - - - - - - - -
                params['P_DT_n'] = np.sum(fusion_rate_DT * params['DT_n_energy'] * consts.e *params['dvol']) #[MW]     
                ## - - - - - - - - - - - - - - - - - - - - - -
                ## heating power by the fusion alpha particles
                ## - - - - - - - - - - - - - - - - - - - - - - 
                params['P_DT_alpha'],params['pi_DT_alpha'],params['pe_DT_alpha']= get_heating_power_from_fusion_products('DT_alpha',fusion_rate_DT,params)
                ## ------------------------------------------------------------
                ## determine total DT fusion power
                params['P_DT_fusion'] = np.sum(fusion_rate_DT * params['Q_DT'] * consts.e *params['dvol']) #[MW]
                params['P_DT_fus_evol'][i]=params['P_DT_fusion']

            ## ----------------------------------------------------------------
            ## ---- D + He3 -> p + alpha --------------------------------------
            ## ----------------------------------------------------------------
            if np.sum(params['ion_dens_arr'][4,:]) >0.:
                fusion_rate_DHe3=calc_fusion_rate('D(He3,p)He4',params)
                ## - - - - - - - - - - - - - - - - - - - -
                ## heating power by the DHe3 fusion protons
                ## - - - - - - - - - - - - - - - - - - - -  
                params['P_DHe3_p'],params['pi_DHe3_p'],params['pe_DHe3_p']= get_heating_power_from_fusion_products('DHe3_p',fusion_rate_DHe3,params)
                ## - - - - - - - - - - - - - - - - - - - - - - - - -
                ## heating power by the DHe3 fusion alpha particles
                ## - - - - - - - - - - - - - - - - - - - - - - - - -
                params['P_DHe3_alpha'],params['pi_DHe3_alpha'],params['pe_DHe3_alpha']= get_heating_power_from_fusion_products('DHe3_alpha',fusion_rate_DHe3,params)
                ## ------------------------------------------------------------
                ## determine total DHe3 fusion power
                params['P_DHe3_fusion'] = np.sum(fusion_rate_DHe3 * params['Q_DHe3'] * consts.e *params['dvol']) #[MW]
                params['P_DHe3_fus_evol'][i]=params['P_DHe3_fusion']
            ## ----------------------------------------------------------------
            ## ---- D + D -> p + T  -------------------------------------------
            ## ----------------------------------------------------------------
            fusion_rate_DD_pT=calc_fusion_rate('D(D,p)T',params)
            ## - - - - - - - - - - - - - - - - - - - -
            ## heating power by the DD fusion protons
            ## - - - - - - - - - - - - - - - - - - - -
            params['P_DD_p'],params['pi_DD_p'],params['pe_DD_p']= get_heating_power_from_fusion_products('DD_p',fusion_rate_DD_pT,params)
            ## - - - - - - - - - - - - - - - - - - - -
            ## heating power by the DD fusion Tritons
            ## - - - - - - - - - - - - - - - - - - - -
            params['P_DD_T'],params['pi_DD_T'],params['pe_DD_T']= get_heating_power_from_fusion_products('DD_T',fusion_rate_DD_pT,params)
            ## ----------------------------------------------------------------
            ## ---- D + D ->  n + He3 -----------------------------------------
            ## ----------------------------------------------------------------
            fusion_rate_DD_nHe3=calc_fusion_rate('D(D,n)He3',params)
            ## - - - - - - - - - - - - - - - - - - - -
            ## power going into the DD neutrons
            ## - - - - - - - - - - - - - - - - - - - -    
            params['P_DD_n']= np.sum(fusion_rate_DD_nHe3 * params['DD_n_energy'] * consts.e *params['dvol']) #[MW]
            ## - - - - - - - - - - - - - - - - - - - -
            ## heating power by the DD fusion He3 ions
            ## - - - - - - - - - - - - - - - - - - - -      
            params['P_DD_He3'],params['pi_DD_He3'],params['pe_DD_He3']= get_heating_power_from_fusion_products('DD_He3',fusion_rate_DD_nHe3,params)
            ## ----------------------------------------------------------------
            ## determine total DD fusion power
            params['P_DD_fusion'] = np.sum((fusion_rate_DD_pT*params['Q_DD_pT'] + fusion_rate_DD_nHe3*params['Q_DD_nHe3']) * consts.e*params['dvol']) #[MW]
            params['P_DD_fus_evol'][i]= params['P_DD_fusion']
            ## ----------------------------------------------------------------
            ## ----------------------------------------------------------------
            ## -------- Determine the densities of the fusion products --------
            ## ---------------------------------------------------------------- 
            ## ----------------------------------------------------------------
            ## define the overall fast-particle confinment time
            tau_p=calc_tauE(params)*params['rho_He']
            smooth_factor=tau_p/params['time_step']

            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            ## - - - Titium density  - - - - - - - - - - - - - - - - - - - - - 
            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
            ## if there is no Tritium fuelling, we calculate the T density self consistently
            if not params['fueling_mix'][1] > 0: 
                ## subtract the DT-fusion rate from the DD_pT rate
                #nT=(fusion_rate_DD_pT-fusion_rate_DT) * tau_p / 1.e20  ## [10^20/m^3]
                #nT*=(1.-params['f_alpha_loss']) ## account for fast-ion losses
                #params['ion_dens_arr'][1,:]=(smooth_factor*params['ion_dens_arr'][1,:] + nT)/(smooth_factor+1)
                #print('T')
                params['ion_dens_arr'][1,:]=calc_fusion_product_density(params,params['ion_dens_arr'][1,:],\
                                              params['particle_source_nbi']+fusion_rate_DD_pT-fusion_rate_DT)


            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            ## - - - proton density  - - - - - - - - - - - - - - - - - - - - - 
            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            ## add the fusion rates of the DHe3 and DD reactions
            #nproton=(fusion_rate_DD_pT+fusion_rate_DHe3) * tau_p / 1.e20  ## [10^20/m^3]
            #nproton*=(1.-params['f_alpha_loss']) ## account for fast-ion losses
            #params['ion_dens_arr'][2,:]=(smooth_factor*params['ion_dens_arr'][2,:] + nproton)/(smooth_factor+1)
            #print('p')
            params['ion_dens_arr'][2,:]=calc_fusion_product_density(params,params['ion_dens_arr'][2,:],fusion_rate_DD_pT+fusion_rate_DHe3)

            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            ## - - - He3 density - - - - - - - - - - - - - - - - - - - - - - -
            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            ## subtract the fusion rate of the DHe3 reaction from the D(D,n)He3 reaction
            #nHe3=(fusion_rate_DD_nHe3-fusion_rate_DHe3) * tau_p / 1.e20  ## [10^20/m^3]
            #nHe3*=(1.-params['f_alpha_loss']) ## account for fast-ion losses
            #params['ion_dens_arr'][4,:]=(smooth_factor*params['ion_dens_arr'][4,:] + nHe3)/(smooth_factor+1)
            #print('He3')
            params['ion_dens_arr'][4,:]=calc_fusion_product_density(params,params['ion_dens_arr'][4,:],fusion_rate_DD_nHe3-fusion_rate_DHe3)

            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            ## - - alpha particle density folowing the DT and DHe3 reactions -
            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            ## add the fusion rates of the DT and DHe3 reactions
            #nHe4=(fusion_rate_DT+fusion_rate_DHe3)* tau_p / 1.e20  ## [10^20/m^3]
            #nHe4*=(1.-params['f_alpha_loss']) ## account for fast-ion losses
            #params['ion_dens_arr'][3,:]=(smooth_factor*params['ion_dens_arr'][3,:] + nHe4)/(smooth_factor+1)
            #print('He4')
            params['ion_dens_arr'][3,:]=calc_fusion_product_density(params,params['ion_dens_arr'][3,:],fusion_rate_DT+fusion_rate_DHe3)


            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ##  - - modify main ion density according to helium, p,T content -
            ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            impurity_index=np.arange(params['nspecies']-3)+3
            ## pure D fueling:
            if params['fueling_mix'][0] ==1:
                index=np.append([1,2],impurity_index)
                main_ion_density= params['ne_c']-np.matmul(np.transpose(params['ion_dens_arr'][index,:]), params['species_charge'][index]) 
                params['ion_dens_arr'][0,:]=main_ion_density
            ## pure H fueling
            elif params['fueling_mix'][2] == 1:
                index=np.append([0,1],impurity_index)
                main_ion_density= params['ne_c']-np.matmul(np.transpose(params['ion_dens_arr'][index,:]), params['species_charge'][index]) 
                params['ion_dens_arr'][2,:]=main_ion_density
            ## D-T fuelling
            elif (params['fueling_mix'][0] > 0) & (params['fueling_mix'][1] > 0) & (params['fueling_mix'][2] == 0): 
                index=np.append([2],impurity_index)
                main_ion_density= params['ne_c']-np.matmul(np.transpose(params['ion_dens_arr'][index,:]), params['species_charge'][index]) 
                params['ion_dens_arr'][0,:]=   params['fueling_mix'][0]*main_ion_density ## D density
                params['ion_dens_arr'][1,:]=   params['fueling_mix'][1]*main_ion_density ## T density
            
            ## ----------------------------------------------------------------
            ## populate the arrays used to plot the density evolution 
            ## ----------------------------------------------------------------
            params['nD_evol'][i]=params['ion_dens_arr'][0,0]
            params['nT_evol'][i]=params['ion_dens_arr'][1,0]
            params['nH_evol'][i]=params['ion_dens_arr'][2,0]
            params['nHe4_evol'][i]=params['ion_dens_arr'][3,0]
            params['nHe3_evol'][i]=params['ion_dens_arr'][4,0]
            
            ## ----------------------------------------------------------------
            ## calcualte the total ion density
            ## ----------------------------------------------------------------
            params['nIon_c']=np.sum(params['ion_dens_arr'][:,:],axis=0)
            
            ## ----------------------------------------------------------------
            ## calcualte main ion average mass 
            ## ----------------------------------------------------------------
            params['M'] = calc_volume_average(np.matmul(np.transpose(params['ion_dens_arr'][0:3,:]),\
                        params['species_mass'][0:3])/np.sum(params['ion_dens_arr'][0:3,:],axis=0),params)
      

            ## ----------------------------------------------------------------
            ## calculate Zeff
            ## ----------------------------------------------------------------
            ## here, we use matrix-multiplication to calulate something like     
            ## zeff_prof=(params['nMain_c'] + params['nHe_c']*2.**2 + params['nC_c']*6.**2 + params['nFe_c']*26.**2)/params['ne_c']
            zeff_prof= np.matmul(np.transpose(params['ion_dens_arr'][:,:]), params['species_charge']**2) /params['ne_c']
            params['Zeff']=calc_volume_average(zeff_prof,params)
            '''
            ## determine the mean Z of Fe at a given temperature
            meanz_Fe=np.interp(params['Te_c'],params['Te_meanz'],params['meanz_Fe'])           
            '''
        
        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        ## ---- Determine radiatied power -------------------------------------
        ## --------------------------------------------------------------------
        ## --------------------------------------------------------------------
        prad= calc_radiative_power(params)   # [MW, MW/m^3]
        params['prad']=prad ## update radiated power density
        params['P_rad']=np.sum(prad*params['dvol']) ## [MW]
        params['Prad_evol'][i]=params['P_rad']
        ## get radiated power within a user defined minor radius
        index=params['ra_c']< params['rho_prad_core']
        params['P_rad_core']=np.sum(prad[index]*params['dvol'][index])
        
        ## --------------------------------------------------------------------
        ## subtract radiated power from total heating power
        ## --------------------------------------------------------------------
        params['P_abs']=params['P_nbi_net']+params['P_ecrh']+params['P_DT_alpha'] \
            +params['P_DD_p']+params['P_DD_T']+params['P_DD_He3'] \
            +params['P_DHe3_p']+params['P_DHe3_alpha']
            
        params['P_net']=params['P_abs'] - params['P_rad_core']

        params['ne_evol'][i]=params['n_0']

        if params['P_net' ] < 0:
            print('Net heating power gets negative -- stop the calculation as this does not converge')
            params['P_net']=0.001  
            params['Te_c']*=0
            params['Ti_c']*=0
            break
                        
        if i-cc > 50:
            ## Check if an equilibrium state (flat top) is reached
            ## For this evaluate the variation of Te (< 1 eV) and the variaion of Prad (< 10 kW)
            if  (np.std(params['Te_evol'][i-5:i]) < 0.001) & (np.std(params['Prad_evol'][i-5:i]) < 0.010):                    
                if (reduced_pECRH>=0) & (cc==0):
                    ## Reduce the ECRH power to the desired value (reduced_pECRH)
                    print(i,'now lets reduce the ECRH power to %.1f'%reduced_pECRH+' MW')
                    params['pe_ecrh']=np.zeros(params['nrc'])
                    params['P_ecrh'     ] = reduced_pECRH                      
                    params['P_abs']=params['P_nbi_net']+params['P_ecrh']+params['P_DT_alpha'] \
                        +params['P_DD_p']+params['P_DD_T']+params['P_DD_He3'] \
                        +params['P_DHe3_p']+params['P_DHe3_alpha']
                    params['P_net']=params['P_abs'] - params['P_rad_core']
                    cc=i
                else:
                    ## End the simulation at this point (Equilibrium is reached)
                    break

  
    if i == niter-1:
        if np.sum(params['ion_dens_arr'][0,:]) >0.:
            print('------------------------------------------------------------------')
            print('no stable solution found -- either oscillation or too little steps') 
            print('------------------------------------------------------------------')
            #params['Te_c']*=0
            #params['Ti_c']*=0
            #params['P_net']=0.001      

    if "calc_beam_deposition" in params:
        del params["calc_beam_deposition"]
    if "collect_pyfidasim_inputs" in params:
        del params["collect_pyfidasim_inputs"]
    
    params['time_evol']=np.linspace(0,i*params['time_step'],i)
    params['P_DT_fus_evol']=params['P_DT_fus_evol'][0:i]
    params['P_DD_fus_evol']=params['P_DD_fus_evol'][0:i]  
    params['P_DHe3_fus_evol']=params['P_DHe3_fus_evol'][0:i]      
    params['Te_evol']=params['Te_evol'][0:i]
    params['Ti_evol']=params['Ti_evol'][0:i]
    params['ne_evol']=params['ne_evol'][0:i]
    params['nT_evol']=params['nT_evol'][0:i]
    params['nD_evol']=params['nD_evol'][0:i]
    params['nH_evol']=params['nH_evol'][0:i]
    params['nHe4_evol']=params['nHe4_evol'][0:i]
    params['nHe3_evol']=params['nHe3_evol'][0:i] 
    params['Prad_evol']= params['Prad_evol'][0:i]




    print(params['P_DT_fusion'])
    print(params['P_DD_fusion'])
    print(params['P_DHe3_fusion'])
    params['P_fusion']=params['P_DT_fusion']+params['P_DD_fusion']+params['P_DHe3_fusion']
    params['NWL']=(params['P_DT_n']+ params['P_DD_n'])/params['first_wall_surface']
    if (params['P_nbi']+params['P_ecrh']) > 0:
        params['Q']=(params['P_DT_fusion']+params['P_DD_fusion'])/(params['P_nbi']+params['P_ecrh'])
    else:
        params['Q']=np.inf
    print(i,' iterations needed, DT Fusion power: %.1f'%params['P_DT_fusion'],'MW')      
    return(params)
    
    
    


## -----------------------------------------------------------------------------------
## ------ Plasma/Device characteristics, q95, rho*, nu*, L/R time, beta, BetaN -------
## -----------------------------------------------------------------------------------
def calc_q95(params):
    return (params['S'] * 5. * params['a']**2/params['R']
            * params['B_tor']/params['I_p'])
  
def calc_ion_rho_star_prof(params):
    #ion gyroradius r_larmor = sqrt(2*m_i*T_i)/(e*B) 
    r_larmor=np.sqrt(2. * params['M']*consts.atomic_mass * params['Ti_c']*1e3*consts.e)/(consts.e * params['B_tor'])
    return (r_larmor/params['a'])

def calc_electron_rho_star_prof(params):
    #electron gyroradius r_larmor = sqrt(2*m_e*T_e)/(e*B) 
    r_larmor=np.sqrt(2. * consts.m_e * params['Te_c']*1e3*consts.e)/(consts.e * params['B_tor'])
    return (r_larmor/params['a'])

def calc_nu_star_i(params):
    #normalised collision frequency
    Z_i=1
    T_i=calc_volume_average(params['Ti_c'],params)*1.e3  # eV  
    n_i=calc_volume_average(params['nIon_c'],params)*1.e20 # m^-3 
    lnA= 30 - np.log((Z_i*params['Zeff'])**1.5*np.sqrt(n_i)/T_i**1.5)
    return (4.9e-18 * params['R']*params['q_95']*(Z_i*params['Zeff'])**2*lnA
            /(params['a_eff']/params['R'])**1.5
            * n_i/T_i**2)

def calc_ion_nu_star_prof(params):
    #normalised collision frequency from O. Sauter, Phys. Plasmas, Vol. 6, No. 7, July 1999
    Z_i=1.
    T_i=params['Ti_c']*1.e3  # eV
    n_i=params['nIon_c']*1.e20 # m^-3 ## all ions
    index= T_i*n_i*params['rmin_arr_c'] > 0.
    
    lnA= 30 - np.log((Z_i*params['Zeff'])**1.5*np.sqrt(n_i[index])/T_i[index]**1.5) 
    nu_star=np.ones(params['nrc'])*np.nan
    nu_star[index]=4.9e-18 * params['R']*params['q_95']*(Z_i*params['Zeff'])**2*lnA \
            /(params['rmin_arr_c'][index]/params['R'])**1.5 * n_i[index]/T_i[index]**2
            
    return(nu_star)

def calc_nu_star_e(params):
    #normalised collision frequency
    T_e=calc_volume_average(params['Te_c'],params)*1.e3  # eV

    coulomb_logarithm=calc_coulomb_logarithm_NRL(params['n_avg'],T_e)    
    nu_star=(6.921e-18 * params['R']*params['q_95']*params['Zeff']* coulomb_logarithm
            /(params['a_eff']/params['R'])**1.5
            * params['n_avg']*1.e20/(T_e*1.e3)**2)
    return(nu_star)
    
def calc_electron_nu_star_prof(params):
    #normalised collision frequency from O. Sauter, Phys. Plasmas, Vol. 6, No. 7, July 1999 
    coulomb_logarithm=calc_coulomb_logarithm_NRL(params['ne_c'],params['Te_c'])

    nu_star=np.ones(params['nrc'])*np.nan
    nu_star=6.921e-18 * params['R']*params['q_95']*params['Zeff']*coulomb_logarithm \
            /(params['rmin_arr_c']/params['R'])**1.5 \
            * params['ne_c']*1.e20/(params['Te_c']*1.e3)**2     
    return(nu_star)
  
def calc_plasma_R( params ):
    '''
    ----
    The NRL formuary (page 30) provides the transverse spitzer Resistivity as:
    eta_s_perp= 1.03e-2 *Z * lnA /Te**1.5  in ohm cm, with Te in eV
    lnA is the Coulomb logarithm 
    -----
    Let's use Z=1, convert to ohm m and use keV instead of eV:
    eta_s_perp= 1.03e-2 * lnA * lnA /(Te*1.e3)**1.5/100.  in ohm m, with Te in keV
    eta_s_perp= 3.257e-9 * lnA /(Te)**1.5 in Ohm m, with Te in keV    
    ---
    Use Wesson formula 2.16.3 to convert to the parallel resistivity:
    eta_para=eta_perp/1.96  #Ohm m
    eta_para= 1.662e-9 *lnA /(Te)**1.5 in Ohm m, with Te in keV
    ---
    Calculate the Resistance from the resisitivy
    A=np.pi*params['a_eff']**2
    l=2.*np.pi*params['R']
    R=eta_para*l/A
              # m             m^-2              Ohm m           =  Ohm
    --> R= 2.*params['R']/params['a_eff']**2 * 1.662e-9 *lnA /(Te[keV])**1.5
    '''
    Te_avg=calc_volume_average(params['Te_c'],params) ## [keV]
    ne_avg=calc_volume_average(params['ne_c'],params) ## [10^20/m^3]    
    coulomb_logarithm=calc_coulomb_logarithm_NRL(ne_avg,Te_avg)
    R=2.*params['R']/params['a_eff']**2 * 1.662e-9 * coulomb_logarithm / (Te_avg)**1.5
    return(R)
    
def calc_inductance(params):
    ## M. Zanini et al 2020 Nucl. Fusion 60 106021
    ## This formula is appropriate for big top-down circular plasmas like W7-X (John Schmitt)
    ## Moreover, this is also given in the NRL formulary (which has an additional factor of 0.25 next to the "2")
    L=consts.mu_0*params['R']*(np.log(8*params['R']/params['a_eff'])-2)
    ## The factor "2" is a shaping factor (higher values for more peaked profiles)
    return (L) # Ohm s

def calc_beta(params):
    ## calculate the plasma beta: (2 mu0 <p>)/B**2
    p_avg=calc_average_pressure_from_stored_energy(params) # [Pa]
    return ( 2.*(4.*np.pi*1e-7)*p_avg/params['B_tor']**2 )

def calc_betaN(params):
    ## normalized beta is given by betaN= beta *a*B/Ip (See Miyamoto Textbook)
    ## A typical limit of betaN is 3.5%
    beta_N=calc_beta(params)*(params['a']*params['B_tor'])/ params['I_p']    
    return (beta_N)    
    
def calc_beta_fast(params,energy,mass,charge):
    '''
    calculate fast-ion beta. This only works for pure fast-ion heating
    '''
    ## inputs
    ## energy [keV]
    ## mass   [amu]       
    ## charge [C]        
    ## determine the fast-ion beta
    return (calc_beta(params)/(1.+calc_tauE(params)/calc_tau_sd(params,energy,mass,charge)))
    
def calc_v_fast(energy, mass):
    ## inputs:
    ## energy [keV]
    ## mass   [amu]
    return np.sqrt( 2.*energy*1e3*consts.e / (mass*consts.atomic_mass)) # m/s

def calc_fast_ion_larmor(params,energy, mass, charge):
    ## inputs:
    ## energy [keV]
    ## mass   [amu]  
    ## charge  [C]
    ## r= m*vperp/(qB)
    return (mass*consts.atomic_mass * calc_v_fast(energy, mass) / (charge* consts.e * params['B_tor']))
   
def calc_v_Alfven(params):
    ## Alfven velocity
    return(params['B_tor'] / np.sqrt(consts.mu_0 * params['M']*consts.atomic_mass * n_lineavg(params)*1.e20))    
   
    



def calc_net_electrical_power(params):
    '''
    Determine the net electrical output power: We use Zohm's 2010 paper of the "minimum size of Demo" for this.
    input: Fusion power in MW
    output: Electrical power and Aux Power in MW
    '''  
    eta_th = 0.43     ## thermodynamic efficiency for a helium Brayton cycle (X. R. Wang 2015)
    
    if (params['fueling_mix'][0] >0) &  (params['fueling_mix'][1] >0):
        Mn=1.16          ## Neutron Multiplyer (Zohm 2017)      
    else:
        Mn=1. 
 
    P_neutron=params['P_DT_n']+ params['P_DD_n']
    P_plasma=params['P_fusion']-P_neutron



    P_recirc = 0
    P_pump=0
    P_aux=0
    ## iterate to find a solution with the recirculating power
    for i in range(10):
        P_thermal=(Mn * P_neutron +  P_plasma + 0.9*P_pump + 0.3*P_aux)
        P_gross = eta_th * P_thermal

        ## consider an NBI efficiency of 33% and an ECRH efficiency of 50%   
        P_aux=  params['P_nbi']/0.33 + params['P_ecrh']/0.5
            
        ## Pumping power estiamted based on: 
        ## X. R. Wang,"Power Core Design and Engineering", Fusion Science and Technology, 67:1, 193-219, DOI: 10.13182/FST14-798
        P_pump=P_plasma*0.078 + Mn*P_neutron*0.037


        ## Infrastructure
        P_BOP=115. ## [MW] this considers 70 MW for operations, 30 MW for the cooling system and 15 MW for the Tritium  plant (Anđelka Kerekeša SOFT 2021)
        
        ## recriculating power
        P_recirc= P_aux + P_pump + P_BOP

    P_el= P_gross - P_recirc
    return(P_el,P_recirc)



## -------------------------------------------------------------
## -------- List the output of the reactor study ---------------
## -------------------------------------------------------------
def print_results(params):
    print(params['name'],params['kind'])
    print('---- '+params['name']+ ' '+params['kind']+' setup ----')
    print('{:<21s}'.format('R:'),'{:<.2f}'.format(params['R']),'\tm')
    print('{:<21s}'.format('a_eff:'), '{:<.2f}'.format(params['a_eff']),'\tm')
    print('{:<21s}'.format('Volume:'), '{:<.2f}'.format(params['V']),'\tm^3')    
    print('{:<21s}'.format('Surface:'), '{:<.2f}'.format(params['plasma_surface']),'\tm^2') 
    if params['first_wall_surface'] > 0:
        print('{:<21s}'.format('First Wall Surface:'), '{:<.2f}'.format(params['first_wall_surface']),'\tm^2')      
    print('{:<21s}'.format('aspect ratio:'), '{:<.1f}'.format(params['R']/params['a']))
    print('{:<21s}'.format('B:'), '{:<.1f}'.format(params['B_tor']),'\tT')
    if params['kind']== 'tokamak': 
        print('{:<21s}'.format('Ip:'), '{:<.2f}'.format( params['I_p']),'\tMA')         
        print('{:<21s}'.format('q95:'), '{:<.2f}'.format( params['q_95']))  
    
    print('{:<21s}'.format('P NBI (inj):'), '{:<.2f}'.format(params['P_nbi']),'\tMW') 
    print('{:<21s}'.format('P ECRH:'), '{:<4.2f}'.format(params['P_ecrh']),'\tMW')     
    print('{:<21s}'.format('P DT alpha:'), '{:<4.2f}'.format(params['P_DT_alpha']),'\tMW')  
    print('{:<21s}'.format('P DT neutron:'), '{:<4.2f}'.format( params['P_DT_n']),'\tMW') 
    print('{:<21s}'.format('P DHe3 alpha:'), '{:<4.2f}'.format(params['P_DHe3_alpha']),'\tMW')   
    print('{:<21s}'.format('P DHe3 proton:'), '{:<4.2f}'.format(params['P_DHe3_p']),'\tMW')       
    print('{:<21s}'.format('P DD proton:'), '{:<4.2f}'.format(params['P_DD_p']),'\tMW')   
    print('{:<21s}'.format('P DD neutron:'), '{:<4.2f}'.format(params['P_DD_n']),'\tMW')       
    print('{:<21s}'.format('P DD T:'), '{:<4.2f}'.format(params['P_DD_T']),'\tMW')    
    print('{:<21s}'.format('P DD He3:'), '{:<4.2f}'.format(params['P_DD_He3']),'\tMW')  
      
    #print('{:<21s}'.format('P_DT_alpha/surface:'), '{:<4.2f}'.format(params['P_DT_alpha']/params['surf'][-1]),' [MW/m^2]')    
    print('{:<21s}'.format('Radiated Power:'), '{:<.2f}'.format(params['P_rad']),'\tMW')

    P_sync=np.sum(calc_p_sync(params) *params['dvol']) #[MW]
    print('{:<21s}'.format('Synchrotron Power:'), '{:<.2f}'.format(P_sync),'\tMW')

   


    #print('{:<21s}'.format('P_net:'), '{:<.2f}'.format(params['P_net']),'\tMW')    
    if np.sum(params['ion_dens_arr'][0,:]) >0.: ## this is a DD simulation
       print('{:<21s}'.format('Total fusion power:'), '{:<.3f}'.format(params['P_fusion']/1.e3),'\tGW')
       P_el,P_recirc=calc_net_electrical_power(params)
       print('{:<21s}'.format('Electrical power out:'), '{:<.3f}'.format(P_el/1.e3),'\tGW')
       print('{:<21s}'.format('recirculated power:'), '{:<.3f}'.format(P_recirc/1.e3),'\tGW')
    print('----- Plasma Confinement Parameters ----')   
    print('{:<21s}'.format('H factor:'), params['H_factor'])  
    print('{:<21s}'.format('tauE:'), '{:<.3f}'.format(calc_tauE(params)),'\ts')
    print('{:<21s}'.format('tau_e-i:'), '{:<.3f}'.format(calc_tau_ei(params)),'\ts')   
    print('{:<21s}'.format('beta:'), '{:<.2f}'.format(calc_beta(params)*1.e2), '\t%')
    
    if not params['kind']== 'stellarator':
        print('{:<21s}'.format('betaN:'), '{:<.2f}'.format(calc_betaN(params)*100))    

    print('{:<22s}'.format('Resistance R:')+'{:<4.2e}'.format(calc_plasma_R(params)),'\tOhm')
    print('{:<22s}'.format('inductance L:')+'{:<4.2e}'.format(calc_inductance(params)),'\tH')
    print('{:<22s}'.format('L/R time:')+'{:<.2f}'.format(calc_inductance(params)/calc_plasma_R(params)),'\ts')
        
    print('----- Volume Averages ----')
    print('{:<21s}'.format('Stored energy:       '), '{:<4.2f}'.format(calc_stored_energy(params)/1.e3),'\tkJ')
    print('{:<21s}'.format('<Te>:       '), '{:<4.2f}'.format(calc_volume_average(params['Te_c'],params)),'\tkeV')
    print('{:<21s}'.format('<Ti>:       '), '{:<4.2f}'.format(calc_volume_average(params['Ti_c'],params)),'\tkeV')
    print('{:<22s}'.format('<n-T>-tau:')+'{:<.2e}'.format(calc_volume_average(0.5*(params['Te_c']+params['Ti_c'])*params['ne_c'],params)*1.e20*calc_tauE(params)),'\tm^-3 keV s')
    print('{:<21s}'.format(r'<nu_star>:'), '{:<4.2f}'.format(calc_volume_average(calc_ion_nu_star_prof(params),params)))   
    print('{:<21s}'.format(r'<rho_star>:'), '{:<4.3f}'.format(calc_volume_average(calc_ion_rho_star_prof(params),params)))  

    print('----- Plasma Density and Composition ----')
    print('{:<21s}'.format('core density:'), '{:<.2e}'.format(params['n_0']*1.e20),'\t1/m^3')  
    print('{:<21s}'.format('line-avg. density:'), '{:<.2e}'.format(n_lineavg(params)*1.e20), '\t1/m^3')
    if params['kind']== 'stellarator':    
        print('{:<21s}'.format('Sudo avg. dens limit:'), '{:<.2e}'.format(calc_density_limit(params)*1.e20), '\t1/m^3')
        
    else:
        print('{:<21s}'.format('Greenwald limit:'), '{:<3.1e}'.format(calc_density_limit( params )*1.e20), '\t1/m^3')
               
    print('{:<21s}'.format('Density limit fraction'), '{:<3.1f}'.format(n_lineavg(params)/calc_density_limit(params)*100.), '\t%')
    ## print the ion density fractions
    #print('{:<21s}'.format('Main Ion fraction:'), '{:<3.2f}'.format(calc_volume_average(np.sum(params['ion_dens_arr'][0:3,:],axis=1)/params['ne_c']*100,params)),'\t%')
    for iion in range(params['nspecies']):
        print('{:<21s}'.format(params['species_names'][iion]+' fraction:'), '{:<.3f}'.format(calc_volume_average(params['ion_dens_arr'][iion,:]/params['ne_c']*100,params)),'\t%')
       
    print('{:<21s}'.format('Zeff:'), '{:<3.2f}'.format(params['Zeff']),'')

    if np.sum(params['ion_dens_arr'][0,:]) >0.: ## this is a DT simulation
        print('{:<21s}'.format('neutron wall load (NWL):'), '{:<.3f}'.format(params['NWL']),'\tMW/m^2')

    if 'Ptot_cx' in params.keys():
        print('----- Charge exchange -----')
        print('{:<21s}'.format('Ptot_cx:     '), '{:<5.2f}'.format(params['Ptot_cx']*1.e3),'\tkW') 
        print('{:<21s}'.format('P_e-i:     '), '{:<5.2f}'.format(np.sum(calc_pei(params)*params['dvol'])*1.e3),'\tkW')
        
    print('----- ECRH parameters -----')
    print('{:<21s}'.format('Frequency (1st harmonic):'), '{:<4.0f}'.format(electron_cyclotron_freq(params)/1.e9),'\tGHz')
    #print('{:<21s}'.format('Frequency (2nd harmonic):'), '{:<4.0f}'.format(electron_cyclotron_freq(params)/1.e9*2),'\tGHz')
    #print('{:<21s}'.format('Frequency (3rd harmonic):'), '{:<4.0f}'.format(electron_cyclotron_freq(params)/1.e9*3),'\tGHz')
    print('{:<21s}'.format('O1 cutoff density:'), '{:<4.2e}'.format(calc_cutoff_O(params,harmonic=1)),'\t1/m^3')     
    print('{:<21s}'.format('X2 cutoff density:'), '{:<4.2e}'.format(calc_cutoff_X(params,harmonic=2)),'\t1/m^3')
    print('{:<21s}'.format('O2 cutoff density:'), '{:<4.2e}'.format(calc_cutoff_O(params,harmonic=2)),'\t1/m^3')     
    #print('{:<21s}'.format('X3 cutoff density:'), '{:<4.2e}'.format(calc_cutoff_X(params,harmonic=3)),'\t1/m^3')

    if params['P_nbi'] > 0:
        print('----- NBI parameters -----')
        print('{:<24s}'.format( 'Shine-through:')+'{:.2f}'.format((params['P_nbi']-params['P_nbi_net'])/params['P_nbi']*100.)+'\t%')
        print('{:<24s}'.format( 'Injection energy:')+'{:.2f}'.format(params['E_NBI'])+'\tkeV')
        print('{:<24s}'.format( 'Alfven velocity:')+'{:.1e}'.format(calc_v_Alfven(params))+'\tm/s')
        print('{:<24s}'.format( 'fast-ion velocity:')+ '{:.2f}'.format(calc_v_fast(params['E_NBI'],params['M_NBI'])/calc_v_Alfven(params))+'\tv_a')
        print('{:<24s}'.format( 'fast-ion larmor radius:')+'{:.2f}'.format(calc_fast_ion_larmor(params,params['E_NBI'],params['M_NBI'],params['Z_NBI'])*100.),'\tcm')
   # if params['M'] >2.: ## this is a DT simulation
    #    print('{:<21s}'.format('Alpha particle r_larmor:'), '{:<.1f}'.format(calc_fast_ion_larmor(params,params['DT_alpha_energy']*1.e3, 4., 2.)*100.), '\tcm') 
    print('{:<21s}'.format('Q:'), '{:<.0f}'.format(params['Q']))
    
    if params['P_DD_T'] > 0.1:
        fusion_rate_DD_pT=calc_fusion_rate('D(D,p)T',params)
        print(np.sum(params['particle_source_nbi']*params['dvol'])/np.sum((fusion_rate_DD_pT+params['particle_source_nbi'])*params['dvol'])*100,'% is T fueling')
        pabs=params['P_DT_alpha'] \
            +params['P_DD_p']+params['P_DD_T']+params['P_DD_He3'] \
            +params['P_DHe3_p']+params['P_DHe3_alpha']
        print(params['P_DT_alpha']/pabs*100,'% is DT alpha power')
        fusion_rate_DT=calc_fusion_rate('D(T,n)He4',params)
        print('DD production % of the T consumption:',np.sum((fusion_rate_DD_pT*params['dvol']))/np.sum(fusion_rate_DT*params['dvol'])*100,'%')
        print('burnup fraction: ', np.sum(fusion_rate_DT*params['dvol'])/np.sum((fusion_rate_DD_pT+params['particle_source_nbi'])*params['dvol'])*100,'%')

def calc_cost(params,cost_basis='HSX',inflation=2.):
    ## lets define the cost as a sum of the cost of the
    ## coils, support sturcture, vacuum vessel, heating systems
    
    if cost_basis == 'HSX':
    ## costing based on HSX:
    #Coils $1600k, Support $800k, Vacuum vessel $600k
        coils_ref=1.6e6
        support_ref=0.8e6
        vessel_ref=0.6e6
        power_ref=1.e6 ## we got a quote of super-caps for the HSX main coils
        time_ref=0.1 
        B_ref=1.37    # T
        a_ref=0.12  #m
        R_ref=1.2   #m

    cost=np.zeros(6)
    ## scale coil cost with B^2*a*R
    cost[0]=inflation*coils_ref*(params['a_eff']/a_ref)**2*(params['R']/R_ref)*(params['B_tor']/B_ref)**2
    ## scale support cost with B^2*a*R
    cost[1]=inflation*support_ref*(params['a_eff']/a_ref)**2*(params['R']/R_ref)*(params['B_tor']/B_ref)**2  
    ## scale vacuum vessel with V
    cost[2]=inflation*vessel_ref*(params['a_eff']/a_ref)*(params['R']/R_ref)
    ## power supply scales with the square root of time and current
    cost[3]=power_ref*np.sqrt(params['t_plasma']/time_ref*params['B_tor']/B_ref)
    
    ## HEATING power
    # let's consider 2 M$ per MW of installed power
    cost[4]=2.e6*params['P_abs']*np.sqrt(params['t_plasma'])
    #cost[3:5]=0


    cost=cost/1.e6 ## conversion to M$
    print()
    print('----- costing -----')
    print('{:<24s}'.format( 'coils:')+ '{:.2f}'.format(cost[0])+' M$')
    print('{:<24s}'.format( 'support structure:')+'{:.2f}'.format(cost[1])+' M$')
    print('{:<24s}'.format( 'vessel:')+ '{:.2f}'.format(cost[2])+' M$')
    print('{:<24s}'.format( 'power supply:')+ '{:.2f}'.format(cost[3])+' M$')
    print('{:<24s}'.format( 'NBI/ECRH systems:')+ '{:.2f}'.format(cost[4])+' M$')
    print('{:<24s}'.format( 'Total Cost:')+ '{:.2f}'.format(np.sum(cost))+' M$')     
def plot_plasma_evolution(params):
    ## -------------------------------------------
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, sharex=True,figsize=(7,9))
    plt.subplots_adjust(hspace=.1)
    #ax1.set_title(params['name'])
    ax1.grid()
    ax2.grid()
    ax3.grid()  
    ax4.grid()
    ax5.grid()   
    ax1.plot(params['time_evol'],params['P_DT_fus_evol']/1.e3,label=r'P$_\mathrm{DT}$')
    ax1.plot(params['time_evol'],params['P_DD_fus_evol']/1.e3,label=r'P$_\mathrm{DD}$')   
    ax1.plot(params['time_evol'],params['P_DHe3_fus_evol']/1.e3,label=r'P$_\mathrm{DHe^3}$')       
    ax1.set_ylabel(r'P$_\mathrm{fus}$ [GW]')    
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=16,loc=4,handlelength=1)       
    ax2.plot(params['time_evol'],params['Te_evol'],label=r'core T$_\mathrm{e}$')
    ax2.plot(params['time_evol'],params['Ti_evol'],label=r'core T$_\mathrm{i}$')
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('T [keV]')
    ax2.legend(fontsize=16,loc=4,ncol=2,handlelength=1)
    
    ax3.plot(params['time_evol'],params['ne_evol'],label=r'core n$_\mathrm{e}$')
    ax3.plot(params['time_evol'],params['nD_evol'],label=r'core n$_\mathrm{D}$')     
    ax3.set_ylabel(r'n$_\mathrm{e}$ [10$^{20}$/m$^{3}$]')    
    ax3.set_ylim([0,1.05*np.max(params['ne_evol'])])
    ax3.legend(fontsize=16,loc=4,handlelength=1)      
    
    ax4.plot(params['time_evol'],params['nHe4_evol']/params['ne_evol']*100,label=r'f$_\mathrm{He^4}$')   
    ax4.plot(params['time_evol'],params['nH_evol']/params['ne_evol']*100,label=r'f$_\mathrm{H}$')    
    ax4.plot(params['time_evol'],params['nT_evol']/params['ne_evol']*100,label=r'f$_\mathrm{T}$')   
    ax4.plot(params['time_evol'],params['nHe3_evol']/params['ne_evol']*100,label=r'f$_\mathrm{He^3}$')
    
    ax4.set_ylabel(r'f$_\mathrm{core}$ [%]')
    ax4.set_ylim(bottom=0)
    ax4.legend(fontsize=16,ncol=2,labelspacing = 0.2,handlelength=1)
    
    ax5.plot(params['time_evol'],params['Prad_evol'],label=r'P$_\mathrm{rad}$')
    ax5.set_ylabel(r'P$_\mathrm{rad}$ [MW]')
    ax5.set_ylim(bottom=0)
    ax5.legend(fontsize=16,loc=4,handlelength=1)
    
    
    ax5.set_xlabel('iteration [a.u.]')
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    fig.savefig('Plots/'+params['name']+'_plasma_evolution.eps')
    fig.savefig('Plots/'+params['name']+'_plasma_evolution.png')   
## -------------------------------------------------------------
## ------------ plot results -----------------------------------
## -------------------------------------------------------------    
def plot_profiles(params):
    ##------------------------------------------------
    ## plot radial density and temperature profiles
    ##------------------------------------------------ 
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True)
    plt.subplots_adjust(hspace=.1)
    ax1.grid()
    ax2.grid()
    
    
    #plot density profile
    ax1.plot(params['ra_c'],params['ne_c'],color='k', label=r'n$_\mathrm{e}$')
    '''
    for iion in range(0,params['nspecies']):
        string='$_\mathrm{'+params['species_names'][iion]+'}$'
        ax1.plot(params['ra_c'],params['ion_dens_arr'][iion,:],label=r'n'+string)
    '''
    ax1.legend(loc=1,fontsize=16,ncol=2,labelspacing = 0.1,handlelength=1)
    ax1.set_ylabel(r'n [$10^{20}\mathrm{m}^{-3}$]')
    ax1.set_ylim(bottom=0)    
        
    #plot temperature profile
    ax2.plot(params['ra_c'],params['Te_c'],color='k',label=r'T$_\mathrm{e}$')
    ax2.plot(params['ra_c'],params['Ti_c'],color='r',label=r'T$_\mathrm{i}$') 
    ax2.set_xlim([0,1])
    ax2.set_ylim(bottom=0)   
        

        
    ax2.set_ylabel(r'T [keV]')
    ax2.set_xlabel(r'$\mathrm{\rho}$')
    ax2.legend(loc=1,fontsize=16,labelspacing = 0.1,handlelength=1)    
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    fig.savefig('Plots/ne_Te_profiles.eps')
    fig.savefig('Plots/ne_Te_profiles.png') 

    with open('Te.npy', 'wb') as f:
        np.save(f, params['Te_c'])
  
    with open('Ti.npy', 'wb') as f:
        np.save(f, params['Ti_c'])

    with open('ra.npy', 'wb') as f:
        np.save(f, params['ra_c'])
    '''
    ## -------------------------------------------
    ## plot nu-star and rho-star profiles
    ## -------------------------------------------
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True,figsize=([6.,5.5]))
    plt.subplots_adjust(hspace=.1)
    ax1.grid()
    ax2.grid()
    
    ## plot nu-star
    ax1.semilogy(params['ra_c'],calc_ion_nu_star_prof(params),color='r', label='ion')
    ax1.semilogy(params['ra_c'],calc_electron_nu_star_prof(params),color='k', label=r'e$^-$')  
    ax1.set_ylabel(r'$\nu^*$') 
    ax1.set_ylim([1.e-2,1.e2])
    #ax1.set_yticks([1.e-2,0,1.e2])
    ax1.legend(fontsize=18,loc=1) 
    
    ## plot rho-star
    ax2.semilogy(params['ra_c'],calc_ion_rho_star_prof(params),color='r', label='ion')
    ax2.semilogy(params['ra_c'],calc_electron_rho_star_prof(params),color='k', label=r'e$^-$')   
    ax2.set_ylabel(r'$\rho^*$') 
    ax2.set_ylim([1.e-5,0.1])
    ax2.set_xlabel('r/a')
    ax2.set_xlim([0,1.])
    ax2.legend(fontsize=18,loc=1) 
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    fig.savefig('Plots/nu_star_and_rho_star_profiles.png')
 
    '''
    
    ## -------------------------------------------
    ## plot heat flux
    ## -------------------------------------------
    fig, (ax1) = plt.subplots(nrows=1)
    plt.subplots_adjust(hspace=.1)
    ax1.grid()
    #plot ion heat flux
    ax1.plot(params['ra_c'],calc_qe(params),color='k', label=r'$q_{e}$')
    ax1.plot(params['ra_c'],calc_qi(params),color='r', label=r'$q_{i}$')
    ax1.legend(fontsize=18,loc=1) 
    ax1.set_ylabel(r'MW/$\mathrm{m}^{2}$')
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel('r/a')
    ax1.set_xlim([0,1])
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    fig.savefig('Plots/Heat_flux.png')
    
    
    ## -------------------------------------------
    ## plot heating power and losses
    ## -------------------------------------------
    fig, (ax1) = plt.subplots(nrows=1, sharex=True)
    ax1.grid()

    fusion_ion_heating=params['pi_DT_alpha']+params['pi_DD_p']+params['pi_DD_T']+params['pi_DD_He3']+params['pi_DHe3_p']+params['pi_DHe3_alpha']
    fusion_electron_heating=params['pe_DT_alpha']+params['pe_DD_p']+params['pe_DD_T']+params['pe_DD_He3']+params['pe_DHe3_p']+params['pe_DHe3_alpha']
    ax1.plot(params['ra_c'],fusion_ion_heating,label=r'P$_{fus}$(ion)')
    ax1.plot(params['ra_c'],fusion_electron_heating,label=r'P$_{fus}$(e$^-$)')


    ## plot radiated power
    ax1.plot(params['ra_c'],params['prad'],label=r'P$_{rad}$')
    ## plot NBI power
    if params['P_nbi']>0:
        ax1.plot(params['ra_c'],params['pi_nbi'],label=r'$P_{NBI}$ (ion)')
        ax1.plot(params['ra_c'],params['pe_nbi'],label=r'$P_{NBI}$ (e$^-$)')

    if params['P_ecrh']>0:
        ax1.plot(params['ra_c'],params['pe_ecrh'],label=r'$P_{ECRH}$')

    #ax1.semilogy(params['ra_c'],params['pei'],label=r'$P_{ei}$')
    ax1.plot(params['ra_c'],params['pei'],label=r'$P_{ei}$')    
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel(r'power density [MW/m$^3$]')   
    ax1.set_xlim([0,1])
    #ax1.set_ylim([1.e-2,10.])    
    ax1.set_ylim(bottom=0)        
    ax1.legend(fontsize=14,loc=1) 
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    fig.savefig('Plots/heat_deposition_profiles.eps')
    fig.savefig('Plots/heat_deposition_profiles.png')
    if 'chi' in params:
        fig, (ax1) = plt.subplots(nrows=1)
        ax1.grid()
        ax1.plot(params['ra_c'],params['chi'])
        ax1.set_xlabel(r'$\rho$')
        ax1.set_ylabel(r'$\chi$ [m/s$^2$]')
        ax1.set_xlim([0,1])
        ax1.set_ylim(bottom=0.0)
        fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        fig.savefig('Plots/chi_plot.eps')
        fig.savefig('Plots/chi_plot.png')
    
    if params['P_DD_T'] > 0.1:
        ## -------------------------------------------
        ## plot Tritium fueling rates
        ## -------------------------------------------
        fig=plt.figure()
        plt.grid()
        fusion_rate_DT=calc_fusion_rate('D(T,n)He4',params)
        fusion_rate_DD_pT=calc_fusion_rate('D(D,p)T',params)
    
        plt.plot(params['ra_c'],params['particle_source_nbi'],label='NBI fueling')
        plt.plot(params['ra_c'],fusion_rate_DD_pT,label='D(D,p)T rate')
        plt.plot(params['ra_c'],-fusion_rate_DT,label='D(T,n)He4 rate')
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'T fueling rate [1/s/m$^3$]')
        plt.xlim([0,1])
        plt.legend(fontsize=14,loc=1) 
        fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)





if __name__ == '__main__':
    print('this is just the toolbox')
    import reactor_performance_evaluation
    reactor_performance_evaluation.run()
