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

def rfrac2mu_max(rfrac, ract=0.8, vr=1575, aa_per_r=7459):
    mu_max = vr/aa_per_r *rfrac*ract
    return mu_max

def mu_max2rfrac(mu_max, rsat=1, ract=0.8, aa_per_r=7459, vr=1575):
    rfrac = mu_max*aa_per_r/vr /ract
    return rfrac

def rfp_od2rfrac(au, od, od2g=0.5, dcwfrac=0.55, medium='M9'):
    #fp in g/L
    #x in g/L
    # fp   = au*2.771e-6
    
    if medium =='M9':
        fp = -1.466485e-10*au**2 + 2.703022e-06*au + 2.545039e-05
    elif medium == 'LB':
        fp = 1.004170e-10*au**2 + 2.365363e-06*au - 8.975582e-04
    else:
        raise ValueError('Unrecognized medium')
        
    x    = od*od2g*dcwfrac
    fp_x = fp/x
    fp2r = 820490/27e3
    #820490 = 7459*110
    return fp_x*fp2r

def gfp_od2hfrac(au, od,  od2g=0.5, stoich=1, dcwfrac=0.55, medium='M9'):
    # fp   = 4.9770e-15*au**2 + 5.7066e-08*au
    
    if medium == 'M9':
        fp = 3.762672e-15*au**2 + 6.170191e-08*au + 8.885089e-04
    elif medium == 'LB':
        fp = 7.881538e-15*au**2 + 8.330307e-08*au + 3.028047e-03
    else:
        raise ValueError('Unrecognized medium')
    
    x    = od*od2g*dcwfrac
    fp_x = fp/x
    
    return fp_x*stoich

###############################################################################
#Plotting
###############################################################################
def plot_R_vs_mu(ax, x_axis='mu'):
    vr = 22*60/0.8/7459
    
    # vr = 
    A  = [1, .8, .4, .2]
    R  = np.linspace(0, 0.6, 5)
    
    for activity in A:
        mu = vr*R*activity
        
        if x_axis == 'mu':
            ax.plot(mu, R, ':', color='grey', label='_nolabel')
        
        elif x_axis == 'R':
            ax.plot(R, mu, ':', color='grey', label='_nolabel')
        else:
            raise ValueError('Unrecognized value for x_axis.')

def plot_synH_vs_mu(ax, x_axis='mu'):
    fHs = [1, .8, .4, .2, .1]
    mu  = np.linspace(0, 0.03, 6)
    
    color = 'grey'#np.array([216, 210, 224])/255
    
    for fH in fHs:
        synH = mu*fH
        
        if x_axis == 'mu':
            ax.plot(mu, synH, ':', color=color, label='_nolabel')
        
        elif x_axis == 'R':
            ax.plot(synH, mu, ':', color=color, label='_nolabel')
        else:
            raise ValueError('Unrecognized value for x_axis.')
