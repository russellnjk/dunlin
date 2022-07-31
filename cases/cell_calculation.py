import numpy as np

od2g = 0.5

def doublingtime2mu(t):
    return np.log(2)/t

def mu2doublingtime(mu):
    return np.log(2)/mu

def dai(mu):
    gradient  = 5.78656638987421
    intercept = 0.03648482880435973
    
    y = gradient*mu + intercept
    
    return y

def rfrac2mu(rfrac, rsat=1, ract=1, r2aa=7459, kr=18):
    mu = r2aa*rsat*ract*rfrac/kr
    return mu

def rfp_od2rfrac(au, od, au2ug=2.771e-6, od2g=0.5, mw_fp=27e3, mw_r=820490, stoich=1, dcwfrac=0.5):
    '''
    Parameters
    ----------
    au : float
        Fluorescence in AU.
    od : float
        Biomass in OD.
    au2g : float
        ug FP per AU per L.
    od2g : float
        g DCW per OD per L.
    mw_fp : TYPE, optional
        DESCRIPTION. The default is 27e3.
    mw_r : TYPE, optional
        DESCRIPTION. The default is 820490.
    stoich : TYPE, optional
        DESCRIPTION. The default is 1.
    dcwfrac : float
        g protein per g dry cell weight
        
    Returns
    -------
    float
        Ribosome fraction

    '''
    fp_x =  fp_od_2_frac(au, od, au2ug, od2g, dcwfrac)
    fp2r = stoich*mw_r/mw_fp
    
    return fp_x*fp2r

def gfp_od2gfpfrac(au, od, au2g=7.667e-8, od2g=0.5, mw_fp=27e3, dcwfrac=0.55):
    fp_x =  fp_od_2_frac(au, od, au2g, od2g, dcwfrac)
    
    return fp_x

def fp_od_2_frac(au, od, au2g, od2g, dcwfrac=0.5):
    '''
    Converts fluorescence and OD reading into proteome fraction.    

    Parameters
    ----------
    au : float
        Fluorescence in AU.
    od : float
        Biomass in OD.
    au2g : float
        ug FP per AU per L.
    od2g : float
        g DCW per OD per L.
    dcwfrac : float
        g protein per g dry cell weight

    Returns
    -------
    Proteome fraction.
    '''
    fp = au*au2g
    x  = od*od2g*dcwfrac
    return fp/x
    