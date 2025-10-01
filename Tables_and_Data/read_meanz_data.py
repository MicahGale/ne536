# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:40:09 2021

@author: bgeiger3
"""

import numpy as np

def read_meanz(filename):

    f = open(filename, 'r')
    line = f.readline()
    dummy = line.split()
    
    nTe=int(dummy[0])   
    nelements=35
    ## define an array of strings
    element_arr=np.zeros((nelements)).astype(str)   
    meanz=np.zeros((nelements,nTe))
    Te_data=np.zeros(nTe)
    ## there are 8 colums of data
    nline=int(nTe/8)
    if nTe%8 >0:
        nline+=1
    
    for i in range(nline):
        line = f.readline()
        dummy = line.split()
        for ii in range(len(dummy)):
            ipos=8*i+ii
            Te_data[ipos]=float(dummy[ii])
            
    for j in range(nelements):
        line = f.readline()
        element_arr[j]=line.split()[0]
        for i in range(nline):
            line = f.readline()
            dummy = line.split()
            for ii in range(len(dummy)):
                ipos=8*i+ii
                meanz[j,ipos]=float(dummy[ii])            
    return(Te_data,element_arr,meanz)  


if __name__ == '__main__':
    
    import matplotlib as matplotlib
    import matplotlib.pyplot as plt
    plt.close('all')
    matplotlib.rcParams.update({'font.size': 20})
    filename='lz_puetti_meanz.dat'
    Te_data,element_arr,meanz=read_meanz(filename)

    plt.figure()
    element_plot=['H_','He','C_','Fe']
    for jj in range(len(element_plot)):
        print(element_plot[jj])
        j=np.where(element_arr == element_plot[jj])[0][0]
        print(j)
    #for j in range(len(element_arr)):
        plt.semilogx(Te_data,meanz[j,:],label=element_arr[j])
    plt.legend(fontsize=14)
    
