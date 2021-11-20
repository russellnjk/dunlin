import numpy   as np 
import pandas  as pd
import seaborn as sns
from   mpl_toolkits import mplot3d
from   pathlib      import Path

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
    def __init__(self, data, base_colors=None, palette_type='light_palette', roll=2, 
                 thin=2, truncate=None, levels=None, drop_scenarios=None, consolidate_colors=True, 
                 ):
        
        def _2dict(df):
            return {i: g.droplevel(axis=1, level=0) for i, g in df.groupby(axis=1, level=0)}
        
        data   = self.preprocess(data, roll, thin, truncate, levels, drop_scenarios)
        colors = self.make_colors(data, base_colors, palette_type)
        
        self.colors  = colors
        self._data   = data
        self._dct    = _2dict(data)
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
    def preprocess(df, roll=2, thin=2, truncate=None, levels=None, drop_scenarios=None):
        if levels:
            to_drop = [lvl for lvl in df.columns.names if lvl not in levels]
            df      = df.droplevel(to_drop, axis=1)
            df      = df.reorder_levels(levels, axis=1)
        
        if truncate:
            lb, ub = truncate
            df     = df.loc[lb:ub]
        
        if drop_scenarios:
            lvls = df.columns.names[1:]
            temp = [g for i, g in df.groupby(level=lvls, axis=1) if i not in drop_scenarios]
            df   = pd.concat(temp, axis=1, sort=False)
           
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
            
    ###########################################################################
    #Representation
    ###########################################################################  
    def __str__(self):
        return f'{type(self).__name__}{tuple(self.keys())}'
    
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
    def _set_axis_lim(self, ax, xlim, ylim, zlim=None):
        def helper(func, lim):
            if lim is None:
                pass
            elif hasattr(lim, 'items'):
                func(**lim)
            else:
                func(*lim)
        
        if zlim is None:
            helper(ax.set_xlim, xlim)
            helper(ax.set_ylim, ylim)
        else:
            helper(ax.set_xlim, xlim)
            helper(ax.set_ylim, ylim)
            helper(ax.set_zlim, zlim)
    
    def _parse_color(self, args):
        color = args.get('color')
        
        if type(color) == str:
            args['color'] = upp.colors[color]
            return args
        else:
            return args
    
    def plot(self, AX, yvar, bounds=None, **kwargs):
        xvar = self._t
        return self.plot2(AX, xvar, yvar, bounds, **kwargs)
    
    def plot2(self, AX, xvar, yvar, bounds=None, xsd=None, ysd=None, 
              skip=lambda scenario: False, title=None, xlim=None, ylim=None,
              halflife=None, thin=1,
              **line_args):
        x          = self._getvar(xvar)
        y          = self._getvar(yvar)
        
        if bounds:
            lb, ub = bounds
            x      = x.loc[lb:ub]
            y      = y.loc[lb:ub]
            
        lines         = {}
        ax_with_title = set()
        for scenario, color in self.colors.items():
            if skip(scenario):
                continue
                
            x_vals = x[scenario]
            y_vals = y[scenario]
            
            if halflife is None:
                x_vals = x_vals
                y_vals = y_vals
            else:
                x_vals = x_vals.ewm(halflife=halflife, ignore_na=True).mean()
                y_vals = y_vals.ewm(halflife=halflife, ignore_na=True).mean()
            
            if x_vals.index.nlevels > 1:
                raise NotImplementedError()
            else:
                x_vals = x_vals.values[::thin]
                xerr   = None
            
            if y.index.nlevels > 1:
                raise NotImplementedError()
            else:
                y_vals = y_vals.values[::thin]
                yerr   = None
            
            ax     = AX[scenario] if hasattr(AX, 'items') else AX
            
            defaults   = {'marker': 'o',   'linestyle': 'None',
                          'color' : color, 'label'    : ', '.join([str(s) for s in scenario])
                          }
            line_args_ = self._parse_color({**defaults, **line_args})

            lines[scenario] = ax.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, **line_args_)
            
            if title is not None and ax not in ax_with_title:
                ax.set_title(title)
                ax_with_title.add(ax)
            
            self._set_axis_lim(ax, xlim, ylim)
                
        return lines
    
    def plot3(self, AX, xvar, yvar, zvar, bounds=None,
              skip=lambda scenario: False, title=None, xlim=None, ylim=None, zlim=None,
              halflife=None, thin=1,
              **line_args):
        x = self._getvar(xvar)
        y = self._getvar(yvar)
        z = self._getvar(zvar)
        
        if bounds:
            lb, ub = bounds
            x      = x.loc[lb:ub]
            y      = y.loc[lb:ub]
            z      = z.loc[lb:ub]
            
        lines         = {}
        ax_with_title = set()
        for scenario, color in self.colors.items():
            if skip(scenario):
                continue
                
            x_vals = x[scenario]
            y_vals = y[scenario]
            z_vals = z[scenario]
            
            if halflife is not None:
                x_vals = x_vals.ewm(halflife=halflife, ignore_na=True).mean()
                y_vals = y_vals.ewm(halflife=halflife, ignore_na=True).mean()
                z_vals = z_vals.ewm(halflife=halflife, ignore_na=True).mean()
                
            if x_vals.index.nlevels > 1:
                raise NotImplementedError()
            else:
                x_vals = x_vals.values[::thin]
                xerr   = None
            
            if y.index.nlevels > 1:
                raise NotImplementedError()
            else:
                y_vals = y_vals.values[::thin]
                yerr   = None
            
            if z.index.nlevels > 1:
                raise NotImplementedError()
            else:
                z_vals = z_vals.values[::thin]
                zerr   = None
                
            ax     = AX[scenario] if hasattr(AX, 'items') else AX
            
            defaults   = {'marker': 'o',   'linestyle': 'None',
                          'color' : color, 'label'    : ', '.join([str(s) for s in scenario])
                          }
            line_args_ = self._parse_color({**defaults, **line_args})

            lines[scenario] = ax.plot(x_vals, y_vals, z_vals, **line_args_)
            
            if title is not None and ax not in ax_with_title:
                ax.set_title(title)
                ax_with_title.add(ax)
            
            self._set_axis_lim(ax, xlim, ylim, zlim)
                
        return lines
    
    
    def plot_linear_average(self, ax, xvar, yvar, bounds=None, 
                    skip=lambda scenario: False, title=None, xlim=None, ylim=None,
                    xspan=None,
                    **line_args):
        if title is not None:
            ax.set_title(title)
        
        x = self._getvar(xvar)
        y = self._getvar(yvar)
        
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
        A = np.vstack([x, np.ones(len(x))]).T
        
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        xmax = max(x) if xspan is None else xspan[1]
        xmin = min(x) if xspan is None else xspan[0]
        x_ = np.linspace(xmin, xmax)
        
        line_args_ = {**{'label': '_nolabel'}, **line_args}
        self._set_axis_lim(ax, xlim, ylim)
        self._parse_color(line_args_)
        
        plots      = ax.plot(x_, m*x_+c, '-', **line_args_) 
        
        return plots, m, c
    
    def plot_average(self, ax, yvar, bounds, **kwargs):
        xvar = self._t
        return self.plot2(ax, xvar, yvar, bounds, **kwargs)
        
    def plot2_average(self, ax, xvar, yvar, bounds=None, 
                   skip=lambda scenario: False, title=None, xlim=None, ylim=None,
                   halflife=5, thin=1,
                   **line_args
                   ):
        if title is not None:
            ax.set_title(title)
        
        x = self._getvar(xvar)
        y = self._getvar(yvar)
        
        to_plot  = [c for c in x.columns if not skip(c)]
        
        if not to_plot:
            return
        
        x = x.loc[:,to_plot]
        y = y.loc[:,to_plot]
        
        if bounds:
            lb, ub = bounds
            x      = x.loc[lb:ub]
            y      = y.loc[lb:ub]
        
        x = x.mean(axis=1).ewm(halflife=halflife, ignore_na=True).mean()
        y = y.mean(axis=1).ewm(halflife=halflife, ignore_na=True).mean()
        
        if x.index.nlevels > 1:
            raise NotImplementedError()
        else:    
            x = x.iloc[::thin].values
        
        if y.index.nlevels > 1:
            raise NotImplementedError()
        else:
            y = y.iloc[::thin].values
        
        line_args_ = self._parse_color({**{'label': '_nolabel'}, **line_args})
        self._set_axis_lim(ax, xlim, ylim)
        
        plots = ax.plot(x, y, **line_args_)
        
        return plots, x, y
        
    def time_taken(self, ax, xvar, bounds, rel=True, skip=lambda scenario: False, title=None, ylim=None, **bar_args):
        if title is not None:
            ax.set_title(title)
        
        x = self._getvar(xvar)
        
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

            series.append(', '.join([str(s) for s in scenario]))
            heights.append(time[scenario])
            colors.append(color)
    
        bar_args_ = {**{'color': colors}, **bar_args}
        self._parse_color(bar_args_)
        self._set_axis_lim(ax, None, ylim)
        
        return ax.bar(series, heights, **bar_args_), series, heights
    
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
        