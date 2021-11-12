import numpy   as np 
import pandas  as pd
import seaborn as sns
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_plot as upp

###############################################################################
#Raw Data 
###############################################################################
class TimeResponseData:
    consolidated_colors = {}
    
    ###########################################################################
    #Instantiation
    ###########################################################################
    def __init__(self, data, sd=None, base_colors=None, palette_type='light_palette', roll=2, 
                 thin=2, truncate=None, levels=None, consolidate_colors=True, 
                 ):
        
        def _2dict(df):
            return {i: g.droplevel(axis=1, level=0) for i, g in df.groupby(axis=1, level=0)}
        
        data   = self.preprocess(data, roll, thin, truncate, levels)
        sd     = None if sd is None else self.preprocess(sd, roll, thin, truncate, levels)
        colors = self.make_colors(data, base_colors, palette_type)
        
        self.colors  = colors
        self._data   = data
        self._sd     = sd
        self._dct    = _2dict(data)
        self._dct_sd = None if sd is None else _2dict(sd)
        self._t      = pd.DataFrame(dict.fromkeys(colors, data.index), index=data.index) 
    
    ###########################################################################
    #Supporting Methods
    ###########################################################################
    @classmethod
    def make_colors(cls, df, base_colors=None, palette_type='light_palette'):
        levels    = list(range(df.columns.nlevels))[1:]
        scenarios = sorted([i for i, g in df.groupby(axis=1, level=levels)])
        
        if palette_type == 'light_palette':
            colors = upp.make_light_scenarios(scenarios, base_colors)
        elif palette_type == 'dark_palette':
            colors = upp.make_dark_scenarios(scenarios, base_colors)
        else:
            colors = upp.make_color_scenarios(scenarios, base_colors)
        
        return colors
    
    @staticmethod
    def preprocess(df, roll=2, thin=2, truncate=None, levels=None, state_var='State', to_dict=True):
        if levels:
            to_drop = [lvl for lvl in df.columns.names if lvl not in levels]
            df      = df.droplevel(to_drop, axis=1)
            df      = df.reorder_levels(levels, axis=1)
        
        if truncate:
            lb, ub = truncate
            df = df.iloc[lb:ub]
        
        df = df.rolling(roll, min_periods=1).mean()
        df = df.iloc[::thin]
        
        return df
     
    ###########################################################################
    #Dict-like Behaviour
    ########################################################################### 
    def __contains__(self, key):
        return key in self._dct
    
    def __getitem__(self, key):
        return self._dct[key]
    
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f'Cannot add a state that already exists. Delete {key} first and add the new values.')
        elif type(key) != str:
            raise TypeError('Can only use strings as keys.')
            
        self._dct[key] = value
    
    def keys(self):
        return self._dct.keys()
    
    def values(self):
        return self._dct.values()
    
    def items(self):
        return self._dct.items()
    
    def __iter__(self):
        return iter(self._dct)
    
    def get_color(self, scenario):
        return self.colors[scenario]

    def getsd(self, key, ignore_none=False):
        if self._dct_sd is None:
            if ignore_none:
                return None
            else:
                raise AttributeError('No SD data.')
        
        return self._dct_sd[key]
    
    def setsd(self, key, value):
        if type(key) != str:
            raise TypeError('Can only use strings as keys.')
        if self._dct_sd:
            if key in self._dct_sd:
                raise ValueError(f'Cannot add a state that already exists. Delete {key} first and add the new values.')  
        if self._dct_sd is None:
            self._dct_sd = {}
            
        self._dct_sd[key] = value
    
    def get_size(self):
        tspan = self._data.index
        return {'bounds': (tspan[0], tspan[-1]),
                'tspan' : tspan
                }
    
    def _getvar(self, var):
        if type(var) == str:
            return self[var]
        else:
            return var
        
    def _getsd(self, var, sdvar):
        if sdvar is None:
            if type(var) == str:
                return self.getsd(var, ignore_none=True)
            else:
                return None
        else:
            if type(sdvar) == str:
                return self.getsd(sdvar)
            else:
                return sdvar 
            
    ###########################################################################
    #Representation
    ###########################################################################  
    def __str__(self):
        return f'type(self).__name__{tuple(self.keys())}'
    
    def __repr__(self):
        return self.__str__()
    
    ###########################################################################
    #Operators
    ###########################################################################  
    def dup(self, x, name):
        self[name] = self[x].copy()
    
    def evaluate(self, name, expr):
        result = eval(expr, self._dct)
        
        if name:
            self[name] = result
    
    def diff(self, x, name=None):
        x    = self._getvar(x)
        dt   = np.diff(x.index, prepend=np.NAN)
        dxdt = x.diff().divide(dt, axis=0)
        
        if name:
            self[name] = dxdt
        
        return dxdt
        
    def spec_diff(self, x, name=None):
        x    = self._getvar(x)
        dt   = np.diff(x.index, prepend=np.NAN)
        dxdt = x.diff().divide(dt, axis=0) / x
        
        if name:
            self[name] = dxdt
        
        return dxdt
    
    def add(self, x, *x_, name=None):
        x = self._getvar(x)
        
        for i in x_:
            i = self._getvar(i)
            x = x+i
        
        if name:
            self[name] = x
        return x
    
    def sub(self, x, *x_, name=None):
        x = self._getvar(x)
        
        for i in x_:
            i = self._getvar(i)
            x = x-i
        
        if name:
            self[name] = x
        return x
    
    def mul(self, x, *x_, name=None):
        x = self._getvar(x).copy()
        
        for i in x_:
            i = self._getvar(i)
            x = x*i
        
        if name:
            self[name] = x
        return x
    
    def div(self, x, *x_, name=None):
        x = self._getvar(x)
        
        for i in x_:
            i = self._getvar(i)
            x = x/i
        
        if name:
            self[name] = x
        return x
    
    def apply(self, func, *x_, name=None):
        x_     = [self._getvar(x) for x in x_]
        result = func(*x_) 
        
        if name:
            self[name] = result
        return result
    
    def first_order_gen(self, dx, x, decay, name=None):
        dx    = self._getvar(dx)
        x     = self._getvar(x)
        decay = self._getvar(decay)
        gen_x = dx + decay*x
        
        if name:
            self[name] = gen_x
        return gen_x
    
    ###########################################################################
    #Plotting
    ###########################################################################  
    def _set_axis_lim(self, ax, xlim, ylim):
        def helper(func, lim):
            if lim is None:
                pass
            elif hasattr(lim, 'items'):
                func(**lim)
            else:
                func(*lim)
                
        helper(ax.set_xlim, xlim)
        helper(ax.set_ylim, ylim)
            
    def plot(self, AX, yvar, bounds=None, **kwargs):
        xvar = self._t
        return self.plot2(AX, xvar, yvar, bounds, **kwargs)
    
    def plot2(self, AX, xvar, yvar, bounds=None, xsd=None, ysd=None, 
              skip=lambda scenario: False, title=None, xlim=None, ylim=None,
              **line_args):
        x          = self._getvar(xvar)
        y          = self._getvar(yvar)
        xsd        = self._getsd(xvar, xsd) 
        ysd        = self._getsd(yvar, ysd) 
        
        if bounds:
            lb, ub = bounds
            x      = x.loc[lb:ub]
            y      = y.loc[lb:ub]
            xsd    = None if xsd is None else xsd.loc[lb:ub]
            xsd    = None if ysd is None else ysd.loc[lb:ub]
        
        lines         = {}
        ax_with_title = set()
        for scenario, color in self.colors.items():
            if skip(scenario):
                continue
                
            x_vals = x[scenario].values 
            y_vals = y[scenario].values 
            xerr   = None if xsd is None else xsd[scenario].values
            yerr   = None if ysd is None else ysd[scenario].values
            ax     = AX[scenario] if hasattr(AX, 'items') else AX
            
            defaults   = {'marker': 'o',   'linestyle': 'None',
                          'color' : color, 'label'    : scenario
                          }
            line_args_ = {**defaults, **line_args}

            lines[scenario] = ax.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, **line_args_)
            
            if title is not None and ax not in ax_with_title:
                ax.set_title(title)
                ax_with_title.add(ax)
            
            self._set_axis_lim(ax, xlim, ylim)
                
        return lines
        
    def plot_linear(self, ax, xvar, yvar, bounds=None, 
                    skip=lambda scenario: False, title=None, xlim=None, ylim=None,
                    xspan=None,
                    **line_args):
        x          = self._getvar(xvar)
        y          = self._getvar(yvar)
        
        to_plot  = [c for c in x.columns if not skip(c)]
        
        if not to_plot:
            return
        
        x    = x.loc[:,to_plot]
        y    = y.loc[:,to_plot]
        
        if bounds:
            lb, ub = bounds
            x      = x.loc[lb:ub]
            y      = y.loc[lb:ub]
        
        x = x.values.flatten()
        y = y.values.flatten()
        
        idx = np.isnan(x) | np.isnan(y)
        idx = ~idx
        
        x = x[idx]
        y = y[idx]
        # ax.plot(x, y, '+')
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        xmax = max(x) if xspan is None else xspan[1]
        xmin = min(x) if xspan is None else xspan[0]
        x_ = np.linspace(xmin, xmax)
        
        line_args_ = {**{'label': '_nolabel'}, **line_args}
        plots      = ax.plot(x_, m*x_+c, '-', **line_args_) 
        
        if title is not None:
            ax.set_title(title)
        
        self._set_axis_lim(ax, xlim, ylim)
        
        return m, c, plots
    
    def average(self, ax, xvar, bounds, skip=lambda scenario: False, title=None, ylim=None, **bar_args):
        x          = self._getvar(xvar)
        
        series  = []
        heights = []
        for scenario, color in self.colors.items():
            if skip(scenario):
                continue
            
            if bounds:
                lb, ub = bounds
                x_     = x.loc[lb:ub]
                
            #Finish this part
            x_vals = x_[scenario].values 
            avr    = np.nanmean(x_vals)

            series.append(str(scenario))
            heights.append(avr)
        
        if title is not None:
            ax.set_title(title)
        
        self._set_axis_lim(ax, None, ylim)
        bar_args_ = {**{'color': self.colors.values()}, **bar_args}

        return series, heights, ax.bar(series, heights, **bar_args_)
    
    def time_taken(self, ax, xvar, bounds, rel=True, skip=lambda scenario: False, title=None, ylim=None, **bar_args):
        x          = self._getvar(xvar)
        
        series  = []
        heights = []
        
        lb, ub = bounds
        if lb is None:
            start = pd.Series(x.index[0], index=x.columns)
            lb_   = x.iloc[0].values
        else:
            start = x.sub(lb).abs().idxmin()
            lb_   = lb 

        
        if ub is None:
            ub_ = pd.Series(x.index[-1], index=x.columns)
        elif rel:
            ub_  = lb_*ub
            stop = x.sub(ub_).abs().idxmin()
        else:
            stop = x.sub(ub).abs().idxmin()

        time = stop - start
        
        colors = []
        for scenario, color in self.colors.items():
            if skip(scenario):
                continue

            series.append(str(scenario))
            heights.append(time[scenario])
            colors.append(color)
        
        if title is not None:
            ax.set_title(title)
        
        self._set_axis_lim(ax, None, ylim)
        bar_args_ = {**{'color': colors}, **bar_args}

        return series, heights, ax.bar(series, heights, **bar_args_)
        
    ###########################################################################
    #Export as Dataset
    ###########################################################################  
    def state2dataset(self, *states):
        for state in states:
            temp = self[state] 
            dct  = {}
            for column, values in temp.items():
                dct[(state, 'Data', column)] = values.values 
                dct[(state, 'Time', column)] = values.index.values
        
        return dct
        