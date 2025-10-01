# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:01:17 2021

@author: bgeiger3
"""

import numpy as np 

import numba   
def conditional_numba(skip_numba=False):
    def decorator(func):
        return numba.jit(func,cache=True, nopython=True, nogil=True)
    return decorator

## ---------------------------------------------------------------------------------
## define routine to calcualte the bi-maxwell average by the "brute Force Method"!
## ---------------------------------------------------------------------------------
@conditional_numba(skip_numba=False)              ## with this statement we "call" the numba routine 
def calc_sigma_v_monte_carlo(Egrid,sigma,T_arr,mass1,mass2,nv):    
    ec=1.6021766208e-19 ## Coulomb charge
    amu=1.66053904e-27 ## atomic mass unit
    mu=mass1*mass2/(mass1+mass2) ## reduced mass
    
    ntemp=len(T_arr)  ## get the number of temperature bins
    sigma_v_mean=np.zeros(ntemp)  ## predefine the output array
    for i in range(ntemp):
        ## generate Monte Carlo velocity distributions
        v_arr1 = np.sqrt(T_arr[i] * ec / (mass1*amu)) * np.random.randn(nv,3) #[m/s]
        v_arr2 = np.sqrt(T_arr[i] * ec / (mass2*amu)) * np.random.randn(nv,3) #[m/s]
        
        
        sigma_v=np.zeros((nv,nv)) ## storage array for the sigma*v values for all possibel conbinations of velocity vectors
        for j in range(nv): ## loop over the velocities of particle 1
            v1=v_arr1[j,:]
            for k in range(nv):  ## loop over the velocities of particle 2
                v2=v_arr2[k,:]   
                u=np.abs(v1-v2)  ## get the relative velocity
                u_abs=np.sqrt(u[0]**2+u[1]**2+u[2]**2) ## corresponding absolute velocity
                Erel=0.5*mu*amu*u_abs**2/ec/1.e3 ## in keV        ## get the relative collision energy
                sigma_v[j,k]=np.interp(Erel,Egrid,sigma)*u_abs*1.e-28  ## calcualte the product of sigma and u.
                
        sigma_v_mean[i]=np.mean(sigma_v) ## take the average for a given temperature
    return(sigma_v_mean)
            
## ---------------------------------------------------------------------------------
## define routine to calcualte the bi-maxwell average by the "Analytical formula"!
## ---------------------------------------------------------------------------------
@conditional_numba(skip_numba=False)              ## with this statement we "call" the numba routine 
def calc_sigma_v_analytical(Egrid,sigma,T_arr,mass1,mass2):
    amu=1.66053904e-27 ## atomic mass unit
    ec=1.6021766208e-19 ## Coulomb charge
    mu=mass1*mass2/(mass1+mass2) ## reduced mass
    
    nv=10000
    E_coll=np.linspace(0,1.e5,nv) ## keV ## Collision energy array
    dE=E_coll[1]-E_coll[0]

  
    ntemp=len(T_arr)  ## get the number of temperature bins
    sigma_v_mean=np.zeros(ntemp)  ## predefine the output array

    factor=4./(np.sqrt(2*amu*np.pi)*(1.e3*ec)**(3/2))*1.e-28*ec*1.e3*ec*1.e3
    for i in range(ntemp):
        T=T_arr[i]/1.e3 ## keV
        #print(T,np.sum(E_coll))
        #sigma_v_mean[i]=4./(np.sqrt(2*mu*amu*np.pi)*(T*1.e3*ec)**(3/2)) *\
        # np.sum(dE*ec*1.e3*E_coll*ec*1.e3*np.interp(E_coll,Egrid,sigma)*1.e-28*np.exp(-E_coll/(T)))
        sigma_v_mean[i]=factor/np.sqrt(mu*T**3)*np.sum(dE*E_coll*np.interp(E_coll,Egrid,sigma)*np.exp(-E_coll/(T)))
        
    return(sigma_v_mean)



if __name__ == "__main__":   
    
    import matplotlib.pyplot as plt
    import matplotlib as matplotlib
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    matplotlib.rcParams.update({'font.size': 18})    
    plt.close('all')

    # Reactant masses in atomic mass units (u).
    mass = {'D': 2.014, 'T': 3.016}

    
    ## -------------------------------------------
    ## -------------- Plot sigma-v ---------------
    ## -------------------------------------------
    ## define temperature array for the plot
    ntemp=50
    T_arr=np.logspace(3,6,num=ntemp) #[eV]
    
        
    ## D-T
    from fusion_cross_sections import read_sigma
    sigma_DT,Egrid_DT = read_sigma('Tables_and_Data/D_T_-_4He_n.txt',m1=mass['D'],m2=mass['T'])    
    sigma_v_DT=calc_sigma_v_analytical(Egrid_DT,sigma_DT,T_arr,mass['T'],mass['D'])
   

    # D + D -> T + p
    sigma_DD,Egrid_DD = read_sigma('Tables_and_Data/D_D_-_T_p.txt',m1=mass['D'],m2=mass['D']) 
    sigma_DD2,Egrid_DD2= read_sigma('Tables_and_Data/D_D_-_3He_n.txt',m1=mass['D'],m2=mass['D'])    
    sigma_v_DD=calc_sigma_v_analytical(Egrid_DD,sigma_DD+np.interp(Egrid_DD,Egrid_DD2,sigma_DD2),T_arr,mass['D'],mass['D'])
 

    
    
    ## define plot window
    fig, ax = plt.subplots()
    ax.grid() 
    ax.loglog(T_arr/1.e3,sigma_v_DT, label=r'D(T,n)$^4$He')
    ax.loglog(T_arr/1.e3,sigma_v_DD,  label=r'D(D,n)$^3$He + D(D,p)T')        
    ax.set_xlabel(r'T [keV]')
    ax.set_ylabel('$<\sigma\,v>$ [m$^3$/s]')     
    ax.set_xlim(1, 1000)
    xticks= np.array([1, 10, 100, 1000])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_ylim(1.e-25, 1.e-21)
    
    ax.legend(fontsize=18)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    plt.savefig('Plots/fusion_sigma_v.eps')   
    plt.savefig('Plots/fusion_sigma_v.png')     
    
    
    
    
    ## plot on linear scale
    fig, ax = plt.subplots()
    ax.grid()  
    ax.plot(T_arr/1.e3,sigma_v_DT, label=r'D(T,n)$^4$He')
    ax.plot(T_arr/1.e3,sigma_v_DD, label=r'D(D,n)$^3$He + D(D,p)T')

    ax.set_xlabel(r'T [keV]')
    ax.set_ylabel('$<\sigma\,v>$ [m$^3$/s]')     
    ax.set_xlim(1,100)
    ax.set_ylim(0, 1.e-21)
    
    ax.legend(fontsize=18)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    plt.savefig('Plots/fusion_sigma_v_linear_scale.eps')       
    plt.savefig('Plots/fusion_sigma_v_linear_scale.png')     
    
    