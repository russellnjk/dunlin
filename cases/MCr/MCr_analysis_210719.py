import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.simulate    as sim
import dunlin.curvefit      as cf
import dunlin.dataparser    as dp
import dunlin.traceanalysis as ta

add   = lambda x, y: x + y
minus = lambda x, y: x - y
mul   = lambda x, y: x * y
div   = lambda x, y: x / y
dxdt  = lambda x: x.diff().divide(np.diff(x.index, prepend=np.NAN), axis=0)/x

def apply2data(data, name, func, *states, **kwargs):
    if type(func) == str:
        if func == '+':
            func_ = add
        elif func == '-':
            func_ = minus
        elif func == '*':
            func_ = mul
        elif func == '/':
            func_ = div
        elif func == 'dxdt':
            func_ = dxdt
    elif isnum(func):
        func_ = lambda x: x*func
    elif type(func) == np.ndarray:
        func_ = lambda x: x.multiply(func, axis=0)
    else:
        func_ = func
    
    tables = [data[s] if type(s) == str else s for s in states]
    new    = func_(*tables, **kwargs)
    new    = pd.concat({name: new}, axis=1, names=data.columns.names)
    new    = pd.concat([data, new], axis=1)
    
    return new

def isnum(x):
    try:
        float(x)
        return True
    except:
        return False

def dai(mu):
    gradient  = 5.78656638987421
    intercept = 0.03648482880435973
    
    return mu*gradient + intercept

plt.close('all')
plt.style.use(dn.styles['light_style_multi'])
# plt.style.use(dn.styles['dark_style_multi'])

#Read the file
date = '210719'
df   = pd.read_excel(f'data_{date}_MCr.xlsx', index_col=0, header=[0, 1, 2, 3], sheet_name='Compiled')

#Thin and preprocess
idx  = list(range(50))[::2] + list(range(50, len(df)))[::5]
df   = df.iloc[idx]
time = df.index.values
data = {i: g.droplevel(axis=1, level=0) for i, g in df.groupby(axis=1, level=0)}

df = df.droplevel(3, axis=1)

#Wrap/shortcut
t  = time
w  = lambda name, func, *states, **kwargs: apply2data(df, name, func, *states, **kwargs)


df = w('RFP/OD', '/', 'RFP', 'OD600')
df = w('GFP/OD', '/', 'GFP', 'OD600')
df = w('mu', 'dxdt', 'OD600')

#Set up plots 
fig, AX     = sim.figure(3, 2, None)
# fig, AX_    = sim.figure(3, 2, None)
# AX          = AX + AX_
make_colors = lambda n, base, palette_type='light': sim.palette_types[palette_type](n, color=sim.colors[base])

#Plot time response
base_colors = [sim.colors[c] for c in ['cobalt', 'goldenrod', 'coral']]
col_lvl = 'Type'
sc_lvl  = 'Inducer'


for n0, [i0, g0] in enumerate(df.groupby(by=col_lvl, axis=1)):
    base_color = base_colors[n0]
    
    lvl1   = g0.groupby(by=sc_lvl, axis=1)
    lvl1   = sorted(lvl1, key=lambda x: x[0])
    colors = sim.palette_types['light'](len(lvl1), color=base_color)
    
    for n1, [i1, g1] in enumerate(lvl1):
        color = colors[n1]
        label = i1
        
        AX[0].plot(t, g1['OD600'], 'o', color=color, label=label)
        AX[1].plot(t, g1['RFP/OD'], 'o', color=color, label=label)
        AX[3].plot(t, g1['mu'], 'o', color=color, label=label)
        
        if i0 != 'MCR': 
            AX[2].plot(t, g1['GFP/OD'], 'o', color=color, label=label)
            
            start = 980
            stop  = 1300
            x     = g1['GFP/OD'].loc[start:stop]
            y     = g1['RFP/OD'].loc[start:stop]
            AX[4].plot(x, y, 'o', color=color, label=label)
        
        start = 40
        stop  = 180
        lag   = 30
        x     = g1['mu'].loc[start+lag:stop+lag]
        y     = g1['RFP/OD'].loc[start:stop]
        AX[5].plot(x, y, 'o', color=color, label=label)
        
        # start = 40
        # stop  = 200
        # lag   = 0
        # x     = g1['mu'].loc[start+lag:stop+lag]
        # y     = g1['GFP/OD'].loc[start:stop]
        # AX[6].plot(x, y, 'o', color=color, label=label)
        
AX[0].set_title(f'OD600 {date}')
AX[1].set_title('RFP/OD')
AX[2].set_title('GFP/OD')
AX[3].set_title('mu')
AX[4].set_title('RFP/OD vs GFP/OD (st)')
AX[5].set_title('RFP/OD vs mu (exp)')
# AX[6].set_title('GFP/OD vs mu (exp)')

AX[0].legend()