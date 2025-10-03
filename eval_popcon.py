# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 22:43:38 2025

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
plt.close('all')
## ----------------------------------------------------------------------------
## ------------------------------- Subroutines --------------------------------
## ----------------------------------------------------------------------------
def calc_tauE(params):
    ## determine the 0D energy confinment time
    if params['kind']== 'stellarator':
        #energy confinement time according to ISS04 scaling law
        tau_E=(0.134 * params['a_eff']**2.28
                * params['R']**0.64
                * params['B_tor']**0.84
                * params['P_net']**-0.61 ## net heating power
                * (params['ne']*10.)**0.54
                * params['q_23']**-0.41)
    
    elif params['kind']== 'tokamak':
        #energy confinement time according to ITER98(y,2) scaling law
        tau_E= (0.144 * params['I_p']**0.93
                * params['B_tor']**0.15
                * params['P_net']**-0.69    ## net heating power
                * params['ne']**0.41
                * params['M']**0.19
                * params['R']**1.97
                * (params['a']/params['R'])**0.58
                * params['kappa_s']**0.78)
    else:
        raise Exception('device type not defined -- either stellarator or tokamak')
    return(tau_E*params['H_factor'])

def calc_stored_energy(params):
    ## calculate the stored energy W = P_net * tau_e
    return(params['P_net']* 1e6 * calc_tauE(params)) ## in Joules

def calc_average_pressure_from_stored_energy(params):
    ## Calcualte the plasma pressure [Pascals] from stored energy
    p_avg=1./params['V'] * 2./3. *  calc_stored_energy(params)  ## Pa (eV*consts.e *m^-3)
    return(p_avg)

def calc_temperature_from_pressure_and_density(p,params):
    T=p/((params['DT_dens']+params['he4_dens']+params['W_dens']+params['ne'])*1.e20) / (1.e3*(consts.e))
    return T

def calc_fusion_rate(params):
    ''' 
    Input:
        name: fusion reaction formula
        params: dictionary
    Output:
        fusion reaction rate
    '''
    name='D(T,n)He4'
    ## DT fusion
    dens=params['DT_dens']/2*1.e20 ## Deterium density   /m^3
    ## perform a cubic interpolation to find the sigma-v values
    from scipy.interpolate import interp1d   
    f = interp1d(params['log10_Ti_arr_sigma_v'],np.log10(params['sigma_v_'+name]), kind='cubic', fill_value='extrapolate')
    sigma_v=10**f(np.log10(params['T']))
    fusion_rate=dens**2*sigma_v #[1/m^3/s]
    return(fusion_rate)

def get_heating_power_from_fusion_products(fusion_rate,params):
    '''
    Input:
        parameters  [dictionary]
    Output:
        power
    '''  
    energy=params['DT_alpha_energy'] #MeV
    ## determine the power density
    power_density=fusion_rate * energy *consts.e # [MW/m^3]
    ## get the volume integrated power
    Power = power_density *params['V'] #[MW]
    return(Power)

def calc_radiative_power(params):
    '''
    Calculate Bremsstrahlung and line radiation for the different species considered
    This is based on the cooling factors provided by Th. Puetterich.
    '''
    ## determine the radiation density
    prad_density=0
    name='D_rates'
    prad_density+=1.e28*params['ne']*params['DT_dens']/2*np.interp(params['T']*1.e3, params[name][0],params[name][1]) # [MW*m^3]
    name='T_rates'
    prad_density+=1.e28*params['ne']*params['DT_dens']/2*np.interp(params['T']*1.e3, params[name][0],params[name][1]) # [MW*m^3]
    name='He4_rates'
    prad_density+=1.e28*params['ne']*params['he4_dens']*np.interp(params['T']*1.e3, params[name][0],params[name][1]) # [MW*m^3]
    name='W_rates'
    prad_density+=1.e28*params['ne']*params['W_dens']*np.interp(params['T']*1.e3, params[name][0],params[name][1]) # [MW*m^3]

    Prad=prad_density*params['V']
    params['P_rad']=Prad
    return(Prad)

def calc_beta(params):
    ## calculate the plasma beta: (2 mu0 <p>)/B**2
    p_avg=calc_average_pressure_from_stored_energy(params) # [Pa]
    return ( 2.*(4.*np.pi*1e-7)*p_avg/params['B_tor']**2 )

def calc_betaN(params):
    ## normalized beta is given by betaN= beta *a*B/Ip (See Miyamoto Textbook)
    ## A typical limit of betaN is 3.5
    beta_N=calc_beta(params)*(params['a']*params['B_tor'])/ params['I_p']
    return (beta_N)

def calc_q95(params):
    return (params['S'] * 5. * params['a']**2/params['R']
            * params['B_tor']/params['I_p'])

def calc_density_limit(params):
    ## Density Limit Scalings
    if params['kind']== 'stellarator':
        ## determine the SUDO density limit (for stellarators)
        return ( 0.25 * np.sqrt( params['P_net']*params['B_tor'] 
                                /(params['a_eff']**2*params['R'])))
    elif params['kind']== 'tokamak':
        ## determine the Greenwald density limit (for tokamaks)
        return ( params['I_p'] / (np.pi * params['a']**2) )

def calc_fusion_product_energyies(params):
    mass = {'D': 2.01410178, 'T': 3.01604928, 'He4':4.00260325,'n':1.00866492}
    ## --------------------------------------------------------------------
    ## DT fusion
    ## --------------------------------------------------------------------
    params['Q_DT']=(mass['D']+mass['T'] - (mass['He4'] + mass['n']))*consts.atomic_mass*consts.c**2/consts.e/1.e6
    params['DT_alpha_energy']=params['Q_DT']/(1+mass['He4']/mass['n'])
    params['DT_n_energy']=params['Q_DT']-params['DT_alpha_energy']
    ## ---------------------------------------------------------
    ## Calcualte fusion rates
    ## --------------------------------------------------------- 
    ## define Temperature array
    params['log10_Ti_arr_sigma_v']=np.linspace(-2,3,num=200) # log10(keV)
    ## ---------------------------------------------------------
    ## DT fusion
    ## ---------------------------------------------------------
    from fusion_cross_sections import read_sigma
    directory= os.path.dirname(__file__) + '/Tables_and_Data/'
    sigma,Egrid = read_sigma(directory+'D_T_-_4He_n.txt',mass['D'],mass['T'])
    ## calculate sigma_v on a log grid 
    from fusion_reaction_rates import calc_sigma_v_analytical
    params['sigma_v_D(T,n)He4']=calc_sigma_v_analytical(Egrid,sigma,10**params['log10_Ti_arr_sigma_v']*1.e3,mass['D'],mass['T'])
    ## sort out zero and negative values
    index=params['sigma_v_D(T,n)He4']==0.
    params['sigma_v_D(T,n)He4'][index]=np.min(params['sigma_v_D(T,n)He4'][params['sigma_v_D(T,n)He4']>0])

def get_cooling_rates(params):
    ## Cooling rates
    directory= os.path.dirname(__file__) + '/Tables_and_Data/'
    params['T_rates']= np.loadtxt(directory+'lz_H_puetti_bolo.dat', skiprows = 1, unpack = True)
    params['D_rates']= np.loadtxt(directory+'lz_H_puetti_bolo.dat', skiprows = 1, unpack = True)
    params['He4_rates']= np.loadtxt(directory+'lz_He_puetti_bolo.dat', skiprows = 1, unpack = True)
    params['W_rates']= np.loadtxt(directory+'lz_W_puetti_bolo.dat', skiprows = 1, unpack = True)


def calc_net_electrical_power(params):
    '''
    Determine the net electrical output power: We use Zohm's 2010 paper of the "minimum size of Demo" for this.
    input: Fusion power in MW
    output: Electrical power and Aux Power in MW
    '''  
    eta_th = 0.43     ## thermodynamic efficiency for a helium Brayton cycle (X. R. Wang 2015)
    Mn=1.16          ## Neutron Multiplyer (Zohm 2017)      

    P_neutron=params['P_DT_alpha']*params['DT_n_energy']/params['DT_alpha_energy']
    P_plasma=params['P_DT_alpha']


    P_recirc = 0
    P_pump=0
    P_aux=0
    ## iterate to find a solution with the recirculating power
    for i in range(10):
        P_thermal=(Mn * P_neutron +  P_plasma + 0.9*P_pump + 0.3*P_aux)
        P_gross = eta_th * P_thermal

        ## consider efficiency of 50% (an NBI efficiency of 33% and an ECRH efficiency of 50%)
        P_aux=  params['P_heat']/0.5
            
        ## Pumping power estiamted based on: 
        ## X. R. Wang,"Power Core Design and Engineering", Fusion Science and Technology, 67:1, 193-219, DOI: 10.13182/FST14-798
        P_pump=P_plasma*0.078 + Mn*P_neutron*0.037


        ## Infrastructure
        P_BOP=115. ## [MW] this considers 70 MW for operations, 30 MW for the cooling system and 15 MW for the Tritium  plant (Anđelka Kerekeša SOFT 2021)
        
        ## recriculating power
        P_recirc= P_aux + P_pump + P_BOP

    P_el= P_gross - P_recirc
    return(P_el,P_recirc)

def calc_neutron_wall_loading(params):
    P_neutron=params['P_DT_alpha']*params['DT_n_energy']/params['DT_alpha_energy']
    first_wall_area= 2.*np.pi * params['R'] * 2*np.pi * (params['a_eff']+params['distance_plasma_wall'])
    nwl=P_neutron/first_wall_area
    return nwl


def calc_helium_density(params):
    he4_dens=calc_fusion_rate(params)*calc_tauE(params)*params['rho_star']/1.e20
    return(he4_dens)


def find_temperature_solution(params):
    params['P_abs']=params['P_heat']
    params['P_net']=params['P_abs']
    ## main iteration loop
    params['T']=0
    for i in range(100):
        T=params['T']
        p=calc_average_pressure_from_stored_energy(params)
        params['T']=calc_temperature_from_pressure_and_density(p,params)
        fusion_rate_DT=calc_fusion_rate(params)
        params['he4_dens']=calc_helium_density(params)
        params['DT_dens']=params['ne']-2* params['he4_dens']-50*params['W_dens']
        
        params['P_DT_alpha']= get_heating_power_from_fusion_products(fusion_rate_DT,params)
        params['P_abs']=params['P_heat']+params['P_DT_alpha']
        Prad=calc_radiative_power(params)
        params['P_net']=params['P_abs']-Prad
        if params['P_net'] < 0:
            params['T']=-1
            return(-1)
        if abs(1-T/params['T'])<0.001:
            return(params['T'])
    if i == 99:
        print('no stable solution found!')
        params['T']=-1
        return(-1)

## ----------------------------------------------------------------------------
## ------------------------------- Device Settings ----------------------------
## ----------------------------------------------------------------------------
def main(params = None):
    if params is None:
        params = {}

        params['name']='ITER' 
        params['kind'       ] = 'tokamak'
        params['R'          ] = 6.2
        params['a'          ] = 2.0         # [m] minor radius
        params['distance_plasma_wall']=0.05 
        params['kappa_s'    ] = 1.7         # []  elongation
        params['S'          ] = 2.6        # shaping factor
        params['B_tor'      ] = 5.2        # [T] toroidal magnetic field strength
        params['I_p'        ] = 15.        # [MA]
        params['H_factor'   ] = 1.0       # []     confinment scaling factor
        params['M'   ] = 2.5  ## mass of the main plasma species mix (D+T = 2.5)
        params['ne'        ] = 1.0       # [10^20/m^3] core electron density
        params['he4_dens']=0.05*params['ne']
        params['W_dens']=1.e-9*params['ne']
        params['DT_dens']=params['ne']-2* params['he4_dens']-50*params['W_dens']
        params['P_heat'     ] = 60.0       # MW
        params['rho_star']=5

        params['name']='SPARC' 
        params['kind'       ] = 'tokamak'
        params['R'          ] = 3.3
        params['a'          ] = 1.13        # [m] minor radius
        params['distance_plasma_wall']=0.10 
        params['kappa_s'    ] = 1.84        # []  elongation
        params['S'          ] = 2.63       # shaping factor
        params['B_tor'      ] = 9.2        # [T] toroidal magnetic field strength
        params['I_p'        ] = 7.8        # [MA]
        params['H_factor'   ] = 1.78        # []     confinment scaling factor
        params['M'   ] = 2.5  ## mass of the main plasma species mix (D+T = 2.5)
        params['ne'        ] = 1.3       # [10^20/m^3] core electron density
        params['he4_dens']=0.05*params['ne']
        params['W_dens']=1.e-6*params['ne']
        params['P_heat'     ] = 143      # MW
        params["V"]           =141


        params['name']='Infinity Two' 
        params['kind'       ] = 'stellarator'
        params['R'          ] = 12.5
        params['a_eff'          ] = 1.25        # [m] minor radius
        params['distance_plasma_wall']=0.1 
        params['B_tor'      ] = 9.0        # [T] toroidal magnetic field strength
        params['q_23'       ] = 1.0#?????        # [MA]
        params['H_factor'   ] = 1.2        # []     confinment scaling factor
        params['M'   ] =        2.5        # mass of the main plasma species mix (D+T = 2.5)
        params['ne'        ] = 2.0         # [10^20/m^3] core electron density
        params['he4_dens']=0.05*params['ne']
        params['W_dens']=1.e-6*params['ne']
        params['DT_dens']=params['ne']-2* params['he4_dens']-50*params['W_dens']
        params['P_heat'     ] = 80.0 #????       # MW
        params['rho_star']=5

    ## ----------------------------------------------------------------------------
    ## ----------------- Print Results --------------------------------------------
    ## ----------------------------------------------------------------------------
    calc_fusion_product_energyies(params)
    get_cooling_rates(params)

    if params['kind']=='tokamak':
        params['a_eff']=params['a']*np.sqrt(params['kappa_s']) #[m] effective minor radius
        params['K']=np.sqrt((1+params['kappa_s']**2)/(2*params['kappa_s']))
    else:
        params['q_95']=params['q_23']
        params['a']=params['a_eff']
    params[ 'V'         ] =  2.*np.pi**2 * params['R'] * params['a_eff']**2
    print('---- '+params['name']+ ' '+params['kind']+' setup ----')
    print('{:<21s}'.format('R:'),'{:<.2f}'.format(params['R']),'\tm')
    print('{:<21s}'.format('a_eff:'), '{:<.2f}'.format(params['a_eff']),'\tm')
    print('{:<21s}'.format('Volume:'), '{:<.2f}'.format(params['V']),'\tm^3')    
    print('{:<21s}'.format('aspect ratio:'), '{:<.1f}'.format(params['R']/params['a']))
    print('{:<21s}'.format('B:'), '{:<.1f}'.format(params['B_tor']),'\tT')
    if params['kind']== 'tokamak': 
        print('{:<21s}'.format('Ip:'), '{:<.2f}'.format( params['I_p']),'\tMA')         
        print('{:<21s}'.format('q95:'), '{:<.2f}'.format(calc_q95(params)))
    print('{:<21s}'.format('H factor:'), params['H_factor'])
    print('{:<21s}'.format('DT density:'), '{:<.1f}'.format(params['DT_dens']/params['ne']*100),'%')
    print('{:<21s}'.format('W density:'), '{:<.1e}'.format(params['W_dens']/params['ne']*100),'%')
    if params['kind']== 'tokamak':
        print('{:<21s}'.format('Greenwald limit:'), '{:<3.1e}'.format(calc_density_limit( params )*1.e20), '\t1/m^3')



    nne=10
    ne_arr=np.linspace(0.5,2.5,num=nne,endpoint=True)
    nT=11
    T_target_arr=np.linspace(1,15,num=nT,endpoint=True)

    P_fusion_arr=np.zeros((nne,nT))
    Q_arr=np.zeros((nne,nT))
    P_heat_arr=np.zeros((nne,nT))
    T_arr=np.zeros((nne,nT))
    P_el_arr=np.zeros((nne,nT))

    npheat=2000
    pheat_arr_check=np.linspace(1.,2000,num=npheat)
    for ii,ne in enumerate(ne_arr):
        params['P_heat']=0.1
        for jj,T_target in enumerate(T_target_arr):
            params['ne']=ne
            params['he4_dens']=0.05*params['ne']
            params['W_dens']=1.e-6*params['ne']
            params['DT_dens']=params['ne']-2* params['he4_dens']-50*params['W_dens']
            T=0
            while T < T_target:
            #for kk,p_heat in enumerate(pheat_arr_check):
                params['P_heat']*=1.01#MW
                ## ----------------------------------------------------------------------------
                ## ----------------- Iteratively calcluate the solution------------------------
                ## ----------------------------------------------------------------------------
                T=find_temperature_solution(params)
                
                
            Q=params['P_DT_alpha']*5/params['P_heat']
            '''
            print(T_target)
            print('{:<21s}'.format('Q:'), '{:<.1f}'.format(Q))
            print('{:<21s}'.format('P heat:'), '{:<.2f}'.format(params['P_heat']),'\tMW') 
            print('{:<21s}'.format('Electron density:'), '{:<.2e}'.format(params['ne']*1.e20),'\t1/m^3')
            print('{:<21s}'.format('T:'),'{:<.2e}'.format(params['T']),' keV')
            print('{:<21s}'.format('He4 density:'), '{:<.1f}'.format(params['he4_dens']/params['ne']*100),'%')
           '''
            P_el,P_recirc=calc_net_electrical_power(params)
            #print('{:<21s}'.format('Electrical power out:'), '{:<.3f}'.format(P_el/1.e3),'\tGW')
            params['NWL']= calc_neutron_wall_loading(params)
            T_arr[ii,jj]=params['T']
            P_fusion_arr[ii,jj]=params['P_DT_alpha']*5
            Q_arr[ii,jj]=Q
            P_heat_arr[ii,jj]=params['P_heat']
            P_el_arr[ii,jj]=P_el




    plt.figure()
    cs2=plt.contour(T_target_arr,ne_arr,P_el_arr,colors='k',levels=5)
    plt.clabel(cs2,inline=True,inline_spacing=5, fmt="%.0f", fontsize=9)


    cs3=plt.contour(T_target_arr,ne_arr,Q_arr,colors='blue',levels=5)
    plt.clabel(cs3,inline=True,inline_spacing=5, fmt="%.2f", fontsize=9)


    cs3=plt.contour(T_target_arr,ne_arr,P_heat_arr,colors='r',levels=5)
    plt.clabel(cs3,inline=True,inline_spacing=5, fmt="%.0f", fontsize=9)



    handles = [plt.Line2D([0], [0], color="k", lw=1, label=r'$P_{el}$ [MW]'),
        plt.Line2D([0], [0], color="blue", lw=1, label="Q"),
        plt.Line2D([0], [0], color="r", lw=1, label=r'$P_{HCD}$ [MW]'),]
    plt.legend(handles=handles, loc="lower left",fontsize=9)

    plt.ylabel(r'$<n_e>$ [10$^{20}$m$^{-3}$]')
    plt.xlabel(r'$<T>$ [keV]')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    plt.show()

if __name__ == "__main__":
    main()
