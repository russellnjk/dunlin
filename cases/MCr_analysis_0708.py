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
import dunlin.simulation    as sim
import dunlin.curvefit      as cf
import dunlin.dataparser    as dp
import dunlin.traceanalysis as ta


add   = lambda x, y: x + y
minus = lambda x, y: x - y
mul   = lambda x, y: x * y
div   = lambda x, y: x / y
dxdt  = lambda x: x.diff().divide(np.diff(x.index, prepend=np.NAN), axis=0)/x

def convert_rfp(au_od):
    au2M   = 4.35e9
    od2caa = 3.6e9*1e3*12e8
    au2gaa = 1/au2M*6e23*225
    return au_od*au2gaa/od2caa
convert_rfp(1800)

def convert_gfp(au_od):
    au2M   = 2.77e11
    od2caa = 3.6e9*1e3*12e8
    au2gaa = 1/au2M*6e23*236
    return au_od*au2gaa/od2caa
convert_gfp(2.5e6)

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

    data[name] = new
    
    return data

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
df   = pd.read_excel('data_210708_MCr.xlsx', index_col=0, header=[0, 1], sheet_name='Compiled')
df   = df.drop(['T7, M9', 'T7, M9+CA'], level=1, axis=1)

#Thin and preprocess
idx  = list(range(50))[::2] + list(range(50, len(df)))[::5]
df   = df.iloc[idx]
time = df.index.values
data = {i: g.droplevel(axis=1, level=0) for i, g in df.groupby(axis=1, level=0)}

#Shortcuts
t  = time
w  = lambda name, func, *states, **kwargs: apply2data(data, name, func, *states, **kwargs)

#Constants
RFP_OD2Molar   = 1/18.84/30/1e6
Molar2mol_cell = 1/2.4e9/1e3
OD2V           = 3.6e9*1e3 #fl of cell volume per L
V2aa           = 12e8

#Method 1
w('RFP/OD', '/', 'RFP', 'OD')
w('RFP_M', RFP_OD2Molar, 'RFP/OD')
w('RFP_cell', Molar2mol_cell, 'RFP_M')
w('Ribo', 6e23, 'RFP_cell')
w('R1', 7459+236, 'Ribo')
w('R_frac1', 1/4.16e8, 'R1')
w('R_frac1', 1/4.16e8, 'R1')
w('mu', 'dxdt', 'OD') 

#Method 2
w('V', OD2V, 'OD')
w('prot', V2aa, 'V') #aa/L
w('R2', 6e23*(7459+236)*RFP_OD2Molar, 'RFP')
w('R_frac2', '/', 'R2', 'prot')

#GFP
w('GFP/OD', '/', 'GFP', 'OD')

#Synthesis
w('dR', 'dxdt', 'RFP/OD')
w('synR', '+', 'dR', data['mu']*data['RFP/OD'])
w('synR/R', '/', 'synR', 'RFP/OD')
w('d(GFP/OD)', 'dxdt', 'GFP/OD')
w('synGFP', '+', 'd(GFP/OD)', data['mu']*data['GFP/OD'])

#Split by media
groups = {}
for key, frame in data.items():
    
    splitter = lambda i: i.split(',')[1].strip()
    for i, g in frame.groupby(splitter, axis=1):
        groups.setdefault(i, {})
        groups[i][key] = g
    
M9CA, M9 = groups['M9+CA'], groups['M9']


#Set up plots 
fig, AX     = sim.figure(3, 2, None)
make_colors = lambda n, base, palette_type='dark': sim.palette_types[palette_type](n, color=sim.colors[base])

#Plot for each media
colors = make_colors(len(M9CA['R_frac1'].columns), 'light blue')
colors = ['blue', 'orange', 'red', 'green', 'gray', 'brown']
labels = list(M9CA['OD'].columns)

for color, label in zip(colors, labels):
    
    AX[0].plot(t, M9CA['OD'][label],      '+', label=label, color=color)
    AX[1].plot(t, M9CA['mu'][label],      '+', label=label, color=color)
    AX[2].plot(t, M9CA['RFP/OD'][label], '+', label=label, color=color)
    AX[4].plot(t, M9CA['GFP/OD'][label],  '+', label=label, color=color)
    
    start = 500
    stop  = 1000
    
    x = M9CA['RFP/OD'][label].loc[start:stop]
    y = M9CA['OD'][label].loc[start+0:stop+0]
    AX[3].plot(x, y, '+', label=label, color=color)
    AX[3].set_xlim(0, 2000)
    
    start = 15
    stop  = 150
    x     = M9CA['mu'][label].loc[start+30:stop+30]
    y     = M9CA['RFP/OD'][label].loc[start:stop]
    AX[5].plot(x, y, '+', label=label, color=color)
    AX[5].set_xlim(0, 0.01)
    
    # AX[6].plot(t, M9CA['synR/R'][label],  '+', label=label, color=color)
    
    # start = 80
    # stop  = 200
    # x     = M9CA['mu'][label].loc[start:stop]
    # y     = M9CA['synR/R'][label].loc[start:stop]
    # AX[7].plot(x, y, '+', label=label, color=color)



# colors = make_colors(len(M9['R_frac1'].columns), 'sea')
# labels = list(M9['R_frac1'].columns)

# for color, label in zip(colors, labels):
#     AX[0].plot(t, M9['OD'][label].values, '+', markersize=10, label=label, color=color)
#     AX[1].plot(t, M9['mu'][label].values,      '+', label=label, color=color)
#     AX[2].plot(t, M9['RFP/OD'][label].values,      '+', label=label, color=color)
#     # AX[3].plot(t, M9['R_frac2'][label].values, '+', label=label, color=color)
#     AX[4].plot(t, M9['GFP/OD'][label].values,  '+', label=label, color=color)
    
#     m = 500
#     n = 700

#     x = M9['mu'][label].loc[m:n].values
#     y = M9['RFP/OD'][label].loc[m:n].values
#     AX[5].plot(x, y, '+', label=label, color=color)


AX[1].legend(loc='upper right')

AX[0].set_title('OD')
AX[1].set_title('mu')
AX[2].set_title('RFP/OD')
# AX[3].set_title('Rfrac_2')
AX[4].set_title('GFP/OD')
AX[5].set_title('RFP/OD vs mu')
# AX[6].set_title('synR/R')
# AX[7].set_title('synR/R vs mu')

new_M9CA_R_frac = pd.concat({'R_frac': 0.55*M9CA['R_frac2']}, axis=1)
new_M9_R_frac = pd.concat({'R_frac': 0.55*M9['R_frac2']}, axis=1)
new_M9CA_OD     = pd.concat({'x': M9CA['OD']}, axis=1)
new_M9_OD     = pd.concat({'x': M9['OD']}, axis=1)

new_df = pd.concat([new_M9CA_R_frac, new_M9_R_frac, new_M9CA_OD, new_M9_OD], axis=1)
t_cols = [col + ('Time', ) for col in new_df.columns]
df_t   = pd.DataFrame({col:t for col in t_cols})
df_t.index = t
new_df = pd.concat({'Data': new_df}, axis=1)
new_df.columns = new_df.columns.swaplevel(0, 1)
new_df.columns = new_df.columns.swaplevel(1, 2)

new_df = pd.concat([new_df, df_t], axis=1)
new_df = new_df.loc[:370]
new_df = new_df[np.all(new_df != 0, axis=1)]
print(new_df)

fig2, AX2 = sim.figure(2, 2, None)
final_gfp = M9CA['GFP/OD'].iloc[-1:]
final_rfp = M9CA['RFP/OD'].iloc[-1:]
final_gfp.T.plot.bar(ax=AX2[0], rot=0, fontsize=10)
final_rfp.T.plot.bar(ax=AX2[1], rot=0, fontsize=10)


