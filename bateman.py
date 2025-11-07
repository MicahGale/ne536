# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
plt.close('all')
matplotlib.rcParams.update({'font.size': 20})

## ----------------------------------------------------------------------------
## ---- plot the result of the Bateman equation for the Th232 decay chain -----
## ----------------------------------------------------------------------------

## make array of element names and half lifes
y = 1             ##  year
d = 1./365        ##  day
h = d/24          ##  hour
s = h/3600.       ##  second
name=['No256','Fm252','Cf248','Cm244','Pu240','U236' ,'Th232' ,'Ra228','Ac228','Th228','Ra224','Rn220']
thalf=[2.9*s   , 25*h  , 334*d , 29*y  , 6561*y,2.3e7*y,1.4e10*y,5.57*y ,6.15*h ,1.9*y  , 3.6*d ,55.6*s ]
nelements=len(thalf)
lam_arr=np.log(2)/thalf


## define time axis
t=10**(np.arange(-9,12,0.01))
ntime=len(t)




## ----------------------------------------------------------------------------
## --------  Apply Bateman equation -------------------------------------------
## ----------------------------------------------------------------------------
Nk=np.zeros([nelements,ntime])  ## number of particles in a given element
for ii in range(nelements):
    
    k=ii+1
    ## determine c_i coefficients as provided by the formula
    ci_arr=np.ones(k)
    for i in range(k):
        for j in range(k-1):
            ci_arr[i]*=lam_arr[j] ## product of lambdas from j=1 till k-1
        for j in range(k):
            if i==j:
                continue
            ci_arr[i]/=(lam_arr[j]-lam_arr[i])  ## divide by product of lambda
                                                ## differences from j=1 till k
            
    ## make the ci array and lambda array 2D (time and element)       
    ci_arr2=np.zeros([k,len(t)])
    lam_arr2=np.zeros([k,len(t)])
    for i in range(k):
        ci_arr2[i,:]=ci_arr[i]
        lam_arr2[i,:]=lam_arr[i]   
        
    ## now, apply the sum of the k-axis    
    Nk[ii,:]=np.sum(ci_arr2*np.exp(-lam_arr2*t),axis=0)
    



## finally plot the result
fig, ax = plt.subplots(figsize=[9,6])


## plot the number N as a function of t for all elements
for ii in range(nelements):
    ax.loglog(t,Nk[ii,:],label=name[ii])
    


## plot legend outsize of box  
box = ax.get_position()
ax.set_position([box.x0*1.3, box.y0*1.5, box.width * 0.8, box.height])
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels,loc=1,fontsize=14, bbox_to_anchor=(1.3, 1))
ax.set_xlabel('time [years]')
ax.set_ylabel(r'N(t)')
ax.set_xlim([1.e-9,1.e12])
ax.set_ylim([1.e-20,1.2])
plt.savefig('decay_chain_bateman.png')
plt.show()
