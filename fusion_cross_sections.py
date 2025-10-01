# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:01:17 2021

@author: bgeiger3
"""


def read_sigma(filename,m1=0,m2=0):
    import numpy as np
    """Read in cross section from filename and interpolate to energy grid."""
    '''
    cross-sections can be taken from:
    https://www-nds.iaea.org/exfor/endf.htm
    
    For D-T use e.g.
    Target = H-3
    Reaction= D,n'
    Quantity = sig
    
    '''
    Egrid, sigma = np.genfromtxt(filename, comments='#', skip_footer=2, unpack=True)
    if m1 !=0:       
        ## correct of the center of mass energy 
        Egrid *= m2 / (m1 + m2)
        #print('do nothing')
    Egrid*=1.e3 ## conversion from MeV to keV
    return(sigma,Egrid) ## return the cross-section in barn





if __name__ == "__main__":   
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as matplotlib
    matplotlib.rcParams.update({'font.size': 20})
    plt.close('all')

    # Reactant masses in atomic mass units (u).
    mass = {'H':1.008,'D': 2.014, 'T': 3.016, 'He3': 3.016, '11B':11.009}


    fig, ax = plt.subplots()
    ax.grid()
    directory='Tables_and_Data/'
    ## ---------------------------------------------------------
    ## DT fusion
    ## ---------------------------------------------------------
    sigma,Egrid = read_sigma(directory+'D_T_-_4He_n.txt',mass['D'],mass['T'])
    ax.loglog(Egrid,sigma,label=r'D(T,n)$^4$He')
    ## ---------------------------------------------------------
    ## DD fusion
    ## ---------------------------------------------------------    
    ## D+D -> p + T
    sigma,Egrid = read_sigma(directory+'D_D_-_T_p.txt',mass['D'],mass['D'])
    ax.plot(Egrid,sigma,label='D(D,p)T')
    ## D+D -> n + He3
    sigma,Egrid = read_sigma(directory+'D_D_-_3He_n.txt',mass['D'],mass['D'])
    ax.plot(Egrid,sigma,label='D(D,n)$^3$He')
    ## ---------------------------------------------------------
    ## D-He3 fusion
    ## ---------------------------------------------------------    
    ## D+He3 -> p + He4
    sigma,Egrid = read_sigma(directory+'D_3He_-_p_4He.txt',mass['D'],mass['He3'])
    ax.plot(Egrid,sigma,label='D($^3$He,p)$^4$He')


    ax.grid(True, which='both', ls='-')
    ax.set_xlim(1, 100)
    xticks= np.array([1, 10, 100, 1000])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_ylim(1.e-4, 1.e1)
    
    ax.set_xlabel(r'E [keV]')
    ax.set_ylabel('$\sigma$ [barn]')
    #ax.set_ylim(1.e-32, 1.e-27)
    
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('Plots/fusion_cross-sections.png')
