import numpy as np
import numba
def conditional_numba():
    def decorator(func):
        return numba.jit(func,cache=True,nopython=True,nogil=True)
    return decorator

def define_strahl_grid(rminor=53.,rmajor=550.,dbound=10, dr_0 = 1., dr_1 = 0.1):
    """
    

    Parameters
    ----------
    rminor : TYPE, optional
        DESCRIPTION. The default is 53..
    rmajor : TYPE, optional
        DESCRIPTION. The default is 550..
    dbound : TYPE, optional
        DESCRIPTION. The default is 10.
    dr_0 : TYPE, optional
        DESCRIPTION. The default is 1..
    dr_1 : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    None.

    """
    ## define radius vectors, and grid coordinates
    rx=rminor+dbound
    k=6. ## exponential factor to describe the higher resolution at the boundary
    dro = 1.
    a0=1./dr_0
    a1=1./dr_1

    ## define the number of grid points:
    nr=int(1.5+rx*(a0*k+a1)/(k+1.))+1
    a1=(k+1.)*(nr-1.5)/rx-a0*k

    ## radius
    rr=np.zeros(nr)
    ro=np.zeros(nr)
    rr[0]=0.
    for i in range(1,nr):
        temp1=0.
        temp2=rx*1.05
        ro[i] = i*dro
        for j in range(0,50):
            rr[i] = (temp1+temp2)/2.
            temp3 = a0*rr[i]+(a1-a0)*rx/(k+1.)*(rr[i]/rx)**(k+1.)
            if (temp3 >= ro[i]):
                temp2=rr[i]
            else:
                temp1=rr[i]

    # terms related with derivatives of ro
    temp1=.5/dro
    pro=np.zeros(nr)  
    qpr=np.zeros(nr)      
    pro[0]=2./dr_0**2
  
    ## pro = (drho/dr)/(2 d_rho) = rho’/(2 d_rho)  =?1/(2 dr)
    ## qpr = (dˆ2 rho/drˆ2)/(2 d_rho) = rho’’/(2 d_rho)
    
    ##pro and qpr are radial values utilized in the discretisation of the transport 
    for i in range(1,nr):
        pro[i]=(a0+(a1-a0)*(rr[i]/rx)**k)*temp1
        qpr[i]=pro[i]/rr[i]+temp1*(a1-a0)*k/rx*(rr[i]/rx)**(k-1.)

    grid={}
    grid['nr']=nr   
    grid['rr']=rr   
    grid['qpr']=qpr
    grid['pro']=pro         ## this is something like rho!?
    grid['der']=dr_0
    grid['dvol']=1./(2*dr_0)*2.*np.pi*rr*2.*np.pi*rmajor
    grid['ra']=rr/rminor 
    grid['rmajor']=rmajor
    grid['rminor']=rminor
    grid['dbound']=dbound
    return(grid)

@conditional_numba()
def strahl_solver(nion, nr, nt, dk, vd, dv, sint, s, al, rr, pro, qpr, der, time, flux, rn_t0, source_charge_state):
    """
    Parameters
    ----------
    nion : TYPE
        DESCRIPTION.
    nr : TYPE
        DESCRIPTION.
    nt : TYPE
        DESCRIPTION.
    dk : TYPE
        DESCRIPTION.
    vd : TYPE
        DESCRIPTION.
    dv : TYPE
        DESCRIPTION.
    sint : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    al : TYPE
        DESCRIPTION.
    rr : TYPE
        DESCRIPTION.
    pro : TYPE
        DESCRIPTION.
    qpr : TYPE
        DESCRIPTION.
    der : TYPE
        DESCRIPTION.
    time : TYPE
        DESCRIPTION.
    flux : TYPE
        DESCRIPTION.
    rn_t0 : TYPE
        DESCRIPTION.

    Returns
    -------
    rn_out : TYPE
        DESCRIPTION.

    """
    maxr=501
    maxz = 75
    rn = np.zeros((maxr,maxz))
    ra = np.zeros((maxr,maxz))
    rn_out = np.zeros((nr,nion,nt))
    #initial densities
    rn[0:nr, 0:nion] = rn_t0
    rn_out[0:nr, 0:nion, 0] = rn[0:nr, 0:nion] 
    
    if dv[-1] != 0:
        if dk.ndim == 3:
            dlen = np.sqrt(np.mean(dk[:,-1,:])/dv[-1])
        else:
            dlen = np.sqrt(np.mean(dk[:,-1])/dv[-1])
    else:
        dlen = abs(rr[-1] - rr[-2])



    #a, b, c, d1, bet, gam, temp1, temp2, tem3, and temp4 are expressions defined
    #to simplify numerical solver
    rnz = np.zeros((maxr,maxz))
    a=np.zeros((maxr,maxz))
    b = np.zeros((maxr,maxz))
    c = np.zeros((maxr,maxz))
    d1 = np.zeros(maxr)
    bet = np.zeros(maxr)
    gam = np.zeros(maxr)


    for it in range(1,nt):
        flx=flux[it-1]
        det = time[it]-time[it-1]
        dt = det/2

        ra[0:nr, 0:nion] = rn[0:nr, 0:nion]


        if source_charge_state ==0:
            #set neutral impurity densities before iteration
            rnz[0:nr, 0] = flx*sint
            rn[0:nr, 0] = flx*sint

        ##the following algorithm is described in detail in section 1.4 in the strahl manual
        ##by Ralph Dux

        ##first half time step direction
        for nz in range(1,nion):
            a[0,nz] = 0
            c[0,nz] = -2*dt*dk[nz,0]*pro[0]
            #c[0,nz] = -2*dt*pro[0]
            b[0,nz] = 1-c[0,nz]+2*dt*vd[nz,1]/der
            d1[0] = ra[0,nz]*(2-b[0,nz])-ra[1,nz]*c[0,nz]
            #r=r or at r+db respectively
            if dlen > 0:
                temp1 = 4*dt*(pro[nr-1]**2)*dk[nz,nr-1]
                temp2 = 0.5*dt*(qpr[nr-1]*dk[nz,nr-1]-pro[nr-1]*vd[nz,nr-1])
                temp3 = 0.5*dt*(dv[nr-1]+vd[nz,nr-1]/rr[nr-1])
                temp4 = 1/(pro[nr-1]*dlen)
                a[nr-1,nz] = -temp1
                b[nr-1,nz] = 1+(1+0.5*temp4)*temp1+temp4*temp2 + temp3
                c[nr-1,nz] = 0
                d1[nr-1] = -ra[nr-2,nz]*a[nr-1,nz]+ra[nr-1,nz]*(2-b[nr-1,nz])
                b[nr-1,nz] = b[nr-1,nz]+dt*s[nr-1,nz]
                d1[nr-1] = d1[nr-1]-dt*(ra[nr-1,nz]*al[nr-1,nz-1]-rnz[nr-1,nz-1]*s[nr-1,nz-1])
                if nz<nion:
                    d1[nr-1] = d1[nr-1] + dt*ra[nr-1,nz+1]*al[nr-1,nz]
            if dlen <0:
                a[nr-1,nz] = 0
                b[nr-1,nz] = 1
                c[nr-1,nz] = 0
                d1[nr-1] = ra[nr-1,nz]
            #normal coefficients
            temp5 = dt*pro[1:nr-1]*pro[1:nr-1]
            temp6 = 4*temp5* dk[nz,1:nr-1]
            temp7 = (dt/2)*(dv[1:nr-1] + pro[1:nr-1]*(vd[nz,2:nr] - vd[nz,0:nr-2]) + vd[nz,1:nr-1]/rr[1:nr-1])
            a[1:nr-1,nz] = 0.5*dt*qpr[1:nr-1]*dk[nz,1:nr-1]+ temp5[0:nr-2]*(0.5*(dk[nz,2:nr] - dk[nz,0:nr-2] - vd[nz,1:nr-1]/pro[1:nr-1]) - 2*dk[nz,1:nr-1])
            b[1:nr-1,nz] = 1+temp6[0:nr-2] + temp7[0:nr-2]
            c[1:nr-1,nz] = -temp6[0:nr-2] - a[1:nr-1,nz]
            d1[1:nr-1] = -ra[0:nr-2,nz]*a[1:nr-1,nz] + ra[1:nr-1,nz]*(2-b[1:nr-1,nz]) - ra[2:nr,nz]*c[1:nr-1,nz]
            
            b[0:nr-1, nz] = b[0:nr-1, nz] +dt*s[0:nr-1, nz]
            d1[0:nr-1] = d1[0:nr-1]-dt*(ra[0:nr-1,nz]*al[0:nr-1,nz-1] - rnz[0:nr-1,nz-1]*s[0:nr-1,nz-1])
    
            if nz < nion:
                d1[0:nr-1] = d1[0:nr-1]+dt*ra[0:nr-1,nz+1]*al[0:nr-1,nz]

            if nz == source_charge_state:
                d1[0:nr-1]+= dt*flx*sint[0:nr-1] ## (add input flux)

            #solution of tridiagonal equation system with
            bet[0] = b[0,nz]
            gam[0] = d1[0]/b[0,nz]
            x = a[1:nr,nz]*c[0:nr-1,nz]
            for i in range(1,nr):
                bet[i] = b[i,nz] - x[i-1]/bet[i-1]
            x = d1[1:nr]/bet[1:nr]
            y = a[1:nr,nz]/bet[1:nr]
            for i in range(1,nr):
                gam[i] = x[i-1] - y[i-1]*gam[i-1]
            rnz[nr-1,nz] = gam[nr-1]
            for i in range(nr-2,-1,-1):
                rnz[i,nz] = gam[i] -c[i,nz]*rnz[i+1,nz]/bet[i]
    
        ##second half time step
        for nz in range(nion-1,0,-1):
            a[0,nz] = 0
            c[0,nz] = -2*dt*dk[nz,0]*pro[0]
            b[0,nz] = 1-c[0,nz]+2*dt*vd[nz,1]/der
            d1[0] = rnz[0,nz]*(2-b[0,nz])-rnz[1,nz]*c[0,nz]
            #r=r or at r+db respectively
            
            if dlen > 0:
                temp1 = 4*dt*(pro[nr-1]**2)*dk[nz,nr-1]
                temp2=.5*dt*(qpr[nr-1]*dk[nz,nr-1]-pro[nr-1]*vd[nz,nr-1])
                temp3=.5*dt*(dv[nr-1]+vd[nz,nr-1]/rr[nr-1])
                temp4 = 1/pro[nr-1]/dlen
                a[nr-1,nz] = -temp1
                b[nr-1,nz] = 1+(1+.5*temp4)*temp1+temp4*temp2+temp3
                c[nr-1,nz] = 0
                d1[nr-1] = -rnz[nr-2,nz]*a[nr-1,nz]+rnz[nr-1,nz]*(2-b[nr-1,nz])
                b[nr-1,nz] = b[nr-1,nz] + dt*al[nr-1,nz-1]
                d1[nr-1] = d1[nr-1] - dt*(rnz[nr-1,nz]*s[nr-1,nz]-rnz[nr-1,nz-1]*s[nr-1,nz-1])
                if nz < nion:
                    d1[nr-1] = d1[nr-1]+dt*rn[nr-1,nz+1]*al[nr-1,nz]
                
            if dlen <=0:
                a[nr-1,nz] = 0
                b[nr-1,nz] = 1
                c[nr-1,nz] = 0
                d1[nr-1] = rnz[nr-1,nz]
            
            #normal coefficients
            temp5 = dt*pro[1:nr-1]* pro[1:nr-1]
            temp6 = 4*temp5* dk[nz,1:nr-1]
            temp7 = (dt/2)*(dv[1:nr-1] + pro[1:nr-1]*(vd[nz,2:nr] - vd[nz,0:nr-2]) + vd[nz,1:nr-1]/ rr[1:nr-1])
            a[1:nr-1,nz] = .5*dt*qpr[1:nr-1]*dk[nz,1:nr-1] + temp5*( .5*(dk[nz,2:nr] - dk[nz,0:nr-2] - vd[nz,1:nr-1]/ pro[1:nr-1]) - 2*dk[nz,1:nr-1])
            b[1:nr-1, nz] = 1+temp6 + temp7
            c[1:nr-1, nz] = -temp6 - a[1:nr-1, nz]
            d1[1:nr-1] = -rnz[0:nr-2, nz]* a[1:nr-1, nz] + rnz[1:nr-1,nz]*(2-b[1:nr-1,nz]) - rnz[2:nr,nz]* c[1:nr-1,nz]
            b[0:nr-1,nz] = b[0:nr-1,nz] + dt*al[0:nr-1,nz-1]
            d1[0:nr-1] = d1[0:nr-1] - dt*(rnz[0:nr-1,nz]* s[0:nr-1,nz] - rnz[0:nr-1,nz-1]* s[0:nr-1,nz-1])
            if nz+1 < nion:
                d1[0:nr-1] = d1[0:nr-1] + dt*rn[0:nr-1, nz+1]* al[0:nr-1,nz]
            
            if nz == source_charge_state:
                d1[0:nr-1]+= dt*flx*sint[0:nr-1] ## (add input flux)
                
            #solution of tridiagonal equation system
            bet[0] = b[0,nz]
            gam[0] = d1[0]/b[0,nz]
            x = a[1:nr,nz]*c[0:nr-1,nz]
            for i in range(1,nr):
                bet[i] = b[i,nz] - x[i-1]/bet[i-1]
            x = d1[1:nr]/bet[1:nr]
            y = a[1:nr,nz]/bet[1:nr]
            for i in range(1,nr):
                gam[i] = x[i-1] - y[i-1]*gam[i-1]
            rn[nr-1,nz] = gam[nr-1]
            for i in range(nr-2,-1,-1):
                rn[i,nz] = gam[i] - c[i,nz]*rn[i+1,nz]/bet[i]

        rn_out[0:nr, 0:nion, it] = rn[0:nr, 0:nion]
    return rn_out