import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin               as dn
import dunlin.simulate    as sim

add    = lambda x, y: x + y
minus  = lambda x, y: x - y
mul    = lambda x, y: x * y
div    = lambda x, y: x / y
dxdt   = lambda x: x.diff().divide(np.diff(x.index, prepend=np.NAN), axis=0)
dxdt_x = lambda x: dxdt(x)/x
var    = lambda x, v=-1 : pd.DataFrame([list(x.columns.get_level_values(v))]*x.shape[0], columns=x.columns, index=x.index)

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
        elif func == 'dxdt_x':
            func_ = dxdt_x
        elif func == 'var':
            func_ = var
        else:
            raise Exception('Unrecogonized func')
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

def light(color, n_colors, reverse=False):
    lst = sns.light_palette(color, n_colors+2, reverse)
    return lst[1:len(lst)-1]

plt.close('all')
plt.style.use(dn.styles['light_style_multi'])
# plt.style.use(dn.styles['dark_style_multi'])

#Read the file
color_map   = {}
base_colors = ['cobalt', 'goldenrod', 'tomato red', 'sea', 
               'purple', 'pink', 'olive', 'tangerine', 
               'pale brown', 'purpley blue', 'electric lime', 'brown',
               'cement', 'royal blue'
               ]

def thin_preprocess(df, gradient='Inducer', skip_scenario=None, skip_gradient=None, thin=True):
    if df is None:
        return None, None
    
    if thin:
        idx  = list(range(20)) + list(range(20, len(df)))[::5]
        df   = df.iloc[idx]
        
    time    = df.index.values
    grd_idx = df.columns.names.index(gradient)
    df      = df.swaplevel(grd_idx, axis=1)
    st_idx  = df.columns.names.index('State')
    df      = df.swaplevel(st_idx, 0, axis=1)
    
    if df.columns.nlevels > 2:
        collapsed  = [df.columns.get_level_values(i) for i in range(1, df.columns.nlevels-1)]
        collapsed  = [', '.join([str(i) for i in x]) for x in zip(*collapsed)]
        cols       = df.columns.get_level_values(0), list(collapsed), df.columns.get_level_values(-1)
        df.columns = pd.MultiIndex.from_tuples(zip(*cols), 
                                               names=('State', 'Scenario', gradient)
                                               )
    if skip_scenario:
        to_drop = [k for k in skip_scenario if k in df.columns.levels[1]]
        df      = df.drop(to_drop, axis=1, level=1)
    
    if skip_gradient:
        to_drop = []
        for col in df.columns:
            if (col[1], col[2]) in skip_gradient:
                to_drop.append(col)
        df = df.drop(to_drop, axis=1)
    
    return time, df

def visualize(date, 
              sd=None,
              gradient='Inducer', 
              palette=light, 
              skip_scenario=None, 
              skip_gradient=None,
              thin=True,
              header=(0, 1, 2, 3),
              AX=None
              ):
    global color_map
    global base_colors
    
    if type(date) == str:
        df   = pd.read_excel(f'data_{date}_MCr.xlsx', index_col=0, header=header, sheet_name='Compiled')
    else:
        df  = date
        
    #Thin and preprocess
    time, df  = thin_preprocess(df, gradient, skip_scenario, skip_gradient, thin=thin)
    _,    sdf = thin_preprocess(sd, gradient, skip_scenario, skip_gradient, thin=thin)
        
    #Wrap shortcut and add states here
    #Does not apply to sd
    t  = time
    if sd is None:
        w  = lambda name, func, *states, **kwargs: apply2data(df, name, func, *states, **kwargs)
        
        df = w('RFP/OD', '/', 'RFP', 'OD600')
        df = w('GFP/OD', '/', 'GFP', 'OD600')
        df = w('mu', 'dxdt_x', 'OD600')
        df = w('dGFP/OD', 'dxdt', 'GFP/OD')
        df = w('dRFP/OD', 'dxdt', 'RFP/OD')
        df = w('IPTG', 'var', 'OD600')
    
    #Set up plots 
    if not AX:
        title    = date if type(date) == str else 'Compiled'
        fig, AX0 = dn.figure(3, 2, None, title=title, fontsize=20)    
        AX       = AX0
    
    #Plot
    bar_x0 = []
    bar_y0 = []
    
    bar_x1 = []
    bar_y1 = []
    
    data_groups = df.groupby(by='Scenario', axis=1, sort=False)
    sd_groups   = None if sdf is None else dict(list(sdf.groupby(by='Scenario', axis=1)))
    
    order       = sorted(df.columns.levels[1], key=lambda x: (len (x), x))
    data_groups = dict(list(data_groups))
    data_groups = [(k, data_groups[k]) for k in order]
    
    for n0, [i0, g0] in enumerate(data_groups):
        if i0 in color_map:
            base_color = color_map[i0]
        else:
            base_color    = base_colors.pop(0)
            color_map[i0] = base_color
        
        
        df_    = g0.groupby(by=gradient, axis=1)
        sdf_   = None if sd_groups is None else dict(list(sd_groups[i0].groupby(by=gradient, axis=1)))
        colors = palette(color=dn.colors[base_color],  n_colors=len(df_))
        
        if g0['GFP/OD'].shape[1] > 1:
            bar_x0.append(str(i0))
            bar_y0.append(float(g0['GFP/OD'].iloc[-1,1]/g0['GFP/OD'].iloc[-1,0]))
            
            bar_x1.append(str(i0))
            bar_y1.append(g0['GFP/OD'].iloc[-1,0])
        
        for n1, [i1, g1] in enumerate(df_):
            g1s = None if sdf_ is None else sdf_[i1]
            
            kw = {'color' : colors[n1],
                  'label' : f'{i0}, {i1}',
                  'marker': 'o',
                  'linestyle' : '',
                  }
            
            plot_ = lambda *j, **k: plot(g1, g1s, t,*j, **k)
            
            plot_(AX[0], None, 'OD600', **kw)
            if 'CHS' not in i0 and 'Nar' not in i0:
                plot_(AX[1], None, 'GFP/OD', **kw)
            plot_(AX[2], None, 'RFP/OD', **kw)
            AX[2].set_ylim(0, 2000)
           
            if 'CHS' not in i0 and 'Nar' not in i0:
                plot_(AX[3], None, 'dGFP/OD', **kw)
                AX[3].set_xlim(0, None)
            
            plot_(AX[4], None, 'mu', **kw)
            AX[4].set_xlim(0, None)
            AX[4].set_ylim(0, None)
            
            interval = (1100, 1200) if 'Nar' in i0 else (900, 1000)
            plot_(AX[5], 'IPTG', 'RFP/OD', interval=interval, **kw)
            AX[5].set_ylim(400, 1200)
            
    # AX[4].bar(bar_x0, bar_y0, align='center')
    # AX[4].set_title('IPTG vs no IPTG')
    
    # AX[5].bar(bar_x1, bar_y1, align='center')
    # AX[5].set_title('No IPTG')
        
    AX[0].legend(loc='lower right', ncol=2)
    return df, AX

def plot(g1, g1s, time, ax, xstate, ystate, interval=None, title='', **kw):
    if interval:
        start, stop = interval
        
        g1_  = g1.loc[start:stop]
        g1s_ = None if g1s is None else g1s.loc[start:stop]
    else:
        g1_, g1s_ = g1, g1s
            
    x = time if xstate is None else g1_[xstate] 
    y = g1_[ystate]
    
    xerr = g1s_[xstate] if g1s_ is not None and xstate is not None else None
    yerr = g1s_[ystate] if g1s_ is not None else None 
    # print(xerr, yerr)
    if not title:
        title = ystate if xstate is None else f'{ystate} v {xstate}'
    ax.set_title(title)
    
    if xerr is not None and yerr is not None:
        return ax.errorbar(x.values, y.iloc[:,0].values, yerr=yerr.iloc[:,0].values, xerr=xerr.iloc[:,0].values, **kw)
    elif yerr is not None:
        return ax.errorbar(x, y.iloc[:,0], yerr.iloc[:,0], **kw)
    else:
        return ax.errorbar(x, y, **kw)
    
    
if __name__ == '__main__':
    dates = ['210715', '210719', '210721', '210820', '210824', '210825']#, '210804']
    dates = ['210820', '210824']
    dates = ['210821']
    dates = ['210817', '210820', '210824']
    dfs   = {}
    AXs   = {}
    
    skip_scenario = ['MCR, M9+CA', 'Nar1, M9+CA', 'Nar2, M9+CA', 'CHS32, M9+CA', 'CHS34, M9+CA']
    
    for date in dates:
        df, AX = visualize(date, skip_scenario=skip_scenario, header=[0, 1, 2, 3])#Call here
        dfs[date] = df
        AXs[date] = AX
        # plt.close(plt.gcf())
    
    
    #Other analysis
    combined = pd.concat(dfs, axis=0)
    mean     = combined.groupby(axis=0, level=1).mean()
    sd       = combined.groupby(axis=0, level=1).std().fillna(0)
    
    # cdf, cAX = visualize(mean, sd=sd, thin=False)
    
    
    # ###############################################
    # dates = ['210819', '210820', '210824', '210825']#CHS and Nar
    # dfs   = {}
    # AXs   = {}
    
    # skip_scenario = ['Nar2, M9+CA', 'C+Ind, M9', 'Ind, M9', 'C+Ind, M9+CA', 'Ind, M9+CA']#['Col 3', 'Col 4', 'Col 5', 'Col 10']
    
    # #Apply the data analysis here
    # for date in dates:
    #     df, AX = visualize(date, skip_scenario=skip_scenario, header=[0, 1, 2, 3])#Call here
    #     dfs[date] = df
    #     AXs[date] = AX
    #     plt.close(plt.gcf())

    # #Other analysis
    # combined = pd.concat(dfs, axis=0)
    # mean     = combined.groupby(axis=0, level=1).mean()
    # sd       = combined.groupby(axis=0, level=1).std().fillna(0)
    
    # cdf, cAX = visualize(mean, sd=sd, thin=False)
    # # cAX[1].legend(ncol=2)
    
   