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
def visualize(date):
    if type(date) == str:
        df   = pd.read_excel(f'data_{date}_MCr.xlsx', index_col=0, header=[0, 1, 2, 3], sheet_name='Compiled')
    else:
        df  = date
        
    #Thin and preprocess
    idx  = list(range(20)) + list(range(20, len(df)))[::5]
    df   = df.iloc[idx]
    time = df.index.values
    cols = df.columns.get_level_values(0), df.columns.get_level_values(1), df.columns.map('{0[2]}, {0[3]}'.format)
    
    df.columns = pd.MultiIndex.from_tuples(zip(*cols), 
                                           names=df.columns.names[0:3]
                                           )

    # data = {i: g.droplevel(axis=1, level=0) for i, g in df.groupby(axis=1, level=0)}
    
    # df = df.droplevel(3, axis=1)
    
    #Wrap/shortcut
    t  = time
    w  = lambda name, func, *states, **kwargs: apply2data(df, name, func, *states, **kwargs)
    
    
    df = w('RFP/OD', '/', 'RFP', 'OD600')
    df = w('GFP/OD', '/', 'GFP', 'OD600')
    df = w('mu', 'dxdt', 'OD600')
    
    #Set up plots 
    fig, AX     = sim.figure(3, 2, None)
    
    #Plot time response
    base_colors  = ['cobalt', 'goldenrod', 'tomato red']
    base_colors2 = ['sea', 'purple', 'pink']
    col_lvl = 'Type'
    sc_lvl  = 'Inducer'
    
    
    for n0, [i0, g0] in enumerate(df.groupby(by=col_lvl, axis=1)):
        base_color  = base_colors[n0] 
        base_color2 = base_colors2[n0] 
        
        
        lvl1    = g0.groupby(by=sc_lvl, axis=1)
        lvl1    = sorted(lvl1, key=lambda x: x[0])
        colors  = sim.palette_types['light'](color=base_color,  n_colors=len(lvl1))
        colors2 = sim.palette_types['light'](color=base_color2, n_colors=len(lvl1))
         
        for n1, [i1, g1] in enumerate(lvl1):
            color = colors[n1] if 'M9' in i1 else colors2[n1]
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
            
            start = 50
            stop  = 180
            lag   = 0
            x     = g1['mu'].loc[start+lag:stop+lag]
            y     = g1['RFP/OD'].loc[start:stop]
            AX[5].plot(x, y, 'o', color=color, label=label)
            AX[5].set_ylim(0, 4000)
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
    AX[1].set_ylim(0, 1200)
    return df, AX

if __name__ == '__main__':
    dates = ['210715', '210719', '210721', '210729']
    dfs   = {}
    AXs   = {}
    
    #Apply the data analysis here
    for date in dates:
        df, AX = visualize(date)#Call here
        dfs[date] = df
        AXs[date] = AX
    
    #Other analysis
    dfs_     = {k:v for k,v in dfs.items() if '29' not in k}
    combined = list( dfs.values())
    n_rows   = min(combined, key=lambda x:x.shape[0]).shape[0]
    combined = [df.iloc[:n_rows, :] for df in combined]
    combined = pd.concat(dict(zip(dates, combined)), axis=0)
    combined = combined.swaplevel(axis=0)
    mean     = combined.mean(axis=0, level=0)
    sd       = combined.std(axis=0, level=0)
    
    
    t = mean.index
    
    #Set up plots 
    fig, AX     = sim.figure(3, 2, None)
    # fig, AX_    = sim.figure(3, 2, None)
    # AX          = AX + AX_
    make_colors = lambda n, base, palette_type='light': sim.palette_types[palette_type](n, color=sim.colors[base])
    
    #Plot time response
    base_colors = ['cobalt', 'goldenrod', 'tomato red']
    base_colors2 = ['sea', 'purple', 'pink']
    col_lvl = 'Type'
    sc_lvl  = 'Inducer'
    
    
    for n0, [i0, g0] in enumerate(mean.groupby(by=col_lvl, axis=1)):
        base_color  = base_colors[n0]
        base_color2 = base_colors2[n0]
        
        lvl1    = g0.groupby(by=sc_lvl, axis=1)
        lvl1    = sorted(lvl1, key=lambda x: x[0])
        colors  = sim.palette_types['light'](color=base_color,  n_colors=len(lvl1))
        colors2 = sim.palette_types['light'](color=base_color2, n_colors=len(lvl1))
        
        for n1, [i1, g1] in enumerate(lvl1):
            color = colors[n1] if 'M9' in i1 else colors2[n1]
            label = i1
            
            AX[0].errorbar(t.values, g1['OD600'].values,  yerr=sd['OD600'][i0][i1].values,  marker='o', linestyle='', color=color, label=label)
            AX[1].errorbar(t.values, g1['RFP/OD'].values, yerr=sd['RFP/OD'][i0][i1].values, marker='o', linestyle='', color=color, label=label)
            AX[3].errorbar(t.values, g1['mu'].values,     yerr=sd['mu'][i0][i1].values,     marker='o', linestyle='', color=color, label=label)
            
            if i0 != 'MCR': 
                AX[2].errorbar(t.values, g1['GFP/OD'].values, yerr=sd['GFP/OD'][i0][i1].values, marker='o', linestyle='', color=color, label=label)
                
                start = 980
                stop  = 1300
                x     = g1['GFP/OD'].loc[start:stop].values
                y     = g1['RFP/OD'].loc[start:stop].values
                xerr  = sd['GFP/OD'][i0][i1].loc[start:stop].values 
                yerr  = sd['RFP/OD'][i0][i1].loc[start:stop].values
                AX[4].errorbar(x, y, yerr=yerr, xerr=xerr, marker='o', linestyle='', color=color, label=label)
            
            # start = 40
            # stop  = 180
            # lag   = 30
            # x     = g1['mu'].loc[start+lag:stop+lag]
            # y     = g1['RFP/OD'].loc[start:stop]
            # AX[5].plot(x, y, 'o', color=color, label=label)
            
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
    
    # AX[0].legend()
    AX[0].legend(ncol=4, bbox_to_anchor=(1.08, 0.45))
    