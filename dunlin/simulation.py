import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.model            as dml
import dunlin._utils_plot.plot as upp

###############################################################################
#Globals
###############################################################################
figure        = upp.figure
gridspec      = upp.gridspec
colors        = upp.colors
palette_types = upp.palette_types
fs            = upp.fs
make_AX       = upp.make_AX
scilimit      = upp.scilimit
save_figs     = upp.save_figs
truncate_axis = upp.truncate_axis

###############################################################################
#High-Level Functions
###############################################################################
def integrate_and_plot(models, multiply=True, overlap=True, include_events=True, _params=None, _exv_names=None, **kwargs):
    
    if type(models) == dml.Model:
        sim_results = integrate_model(models, multiply, overlap, include_events, _params, _exv_names)
        AX1         = plot_sim_results(sim_results, **kwargs)
        
        return AX1, sim_results
    else:
        all_sim_results = integrate_models(models, multiply, overlap, include_events, _params, _exv_names)
        AX1              = plot_all_sim_results(all_sim_results, **kwargs)

        return AX1, all_sim_results
    
###############################################################################
#Integration
###############################################################################
def integrate_models(models, multiply=True, overlap=True, include_events=True, _params=None, _exv_names=None):
    all_sim_results = {}
    to_iter         = models.values() if type(models) == dict else models
    
    for model in to_iter:
        model_key                  = model.model_key 
        _params_model              = _params    if _params is None else _params.get(model_key, None)
        _exv_names_model           = _exv_names if _exv_names is None else _exv_names.get(model_key, None)
        all_sim_results[model_key] = integrate_model(model, 
                                                     multiply, 
                                                     overlap, 
                                                     include_events, 
                                                     _params_model, 
                                                     _exv_names_model
                                                     )
    return all_sim_results

def integrate_model(model, multiply=True, overlap=True, include_events=True, _params=None, _exv_names=None, _tspan=None):
    states      = model.states
    params      = model.params if _params is None else _params
    tspan       = {} if _tspan is None else _tspan
    sim_results = {}
    
    if multiply:
        for scenario, y0 in zip(states.index, states.values):
            for estimate, p in zip(params.index, params.values):
                
                #Integrate
                t, y = model.integrate(scenario, y0, p, tspan=tspan.get(scenario))
                
                #Tabulate and evaluate exvs
                sim_results.setdefault(scenario, {})[estimate] = SimResult(model, t, y, p, scenario,  _exv_names)
                
    else:
        if len(states) != len(params):
            raise ValueError('Attempted to use non-multiplicative simulation but states and params are of different size.')
        
        for scenario, y0, estimate, p in zip(states.index, states.values, params.index, params.values):
            #Integrate
            t, y = model.integrate(scenario, y0, p, tspan=tspan.get(scenario))
            
            #Tabulate and evaluate exvs
            sim_results.setdefault(scenario, {})[estimate] = SimResult(model, t, y, p, scenario,  _exv_names)
            
    return sim_results

###############################################################################
#SimResult Class
###############################################################################
class IntResult():
    @classmethod
    def tabulate_params(cls, model, t, y, p, scenario):
        '''
        Make an array for p that can be concatenated with the states
        :meta private:
        '''
        
        y0          = y[:, 0]
        y0, p_array = model._modify(y0, p, scenario) if model._modify else (y0, p)
        record      = cls.get_event_record(model)
        
        if record:
            p_matrix         = np.zeros( (y.shape[1], len(p_array)) )
            n                = 0
            t_event, p_event = record[n]
            p_curr           = p_array
            
            for i, timepoint in enumerate(t):
                if timepoint < t_event:
                    p_matrix[i] = p_curr
                else:
                    p_curr      = p_event
                    p_matrix[i] = p_curr
                    
                    n += 1
                    
                    if n < len(record):
                        t_event, p_event  = record[n]
                    else:
                        p_matrix[i:] = p_curr
                        break
        else:
            p_matrix = np.tile(p_array, (y.shape[1], 1))
        return p_matrix.T
    
    @staticmethod
    def get_event_record(model):
        record = [pair for event in model._events for pair in event.record]
        record = sorted(record, key=lambda x: x[0])
        return record
    
    @staticmethod
    def evaluate_exv(exv, t, y, *args):
        '''
        :meta private:
        '''
        
        try:
            result = exv(t, y, *args)
        except Exception as e:
            msg    = 'Error in evaluating exv function "{}".'.format(exv.__name__)
            args   = (msg,) + e.args
            e.args = ('\n'.join(args),)
            
            raise e
        
        if hasattr(result, '__iter__'):
            try:
                x, y = result
                return result
            except:
                try:
                    x, y, z = result
                    return result
                except:
                    raise ValueError('Invalid exv')
        else:
            return result
    @classmethod
    def evaluate_exvs(cls, model, t, y, p, scenario, _exv_names=None):
        p_table = cls.tabulate_params(model, t, y, p, scenario)
        
        exv_results = {}
        exv_names   = model._exvs if _exv_names is None else _exv_names
        for exv_name in exv_names:
            exv                   = model._exvs[exv_name]
            exv_results[exv_name] = cls.evaluate_exv(exv, t, y, p_table)
        
        return exv_results, p_table
        
    def __init__(self, model, t, y, p, scenario, _exv_names=None):
        self.model_key = model.model_key
        self.t         = t
        self.y         = dict(zip(model._states, y))
        
        #Evaluate exvs
        exv_results, p_table = self.evaluate_exvs(model, t, y, p, scenario, _exv_names)
        
        p_table_         = dict(zip(model._params, p_table))
        self.y           = {**self.y, **p_table_}
        self.exv_results = exv_results
        
    def __getitem__(self, key):
        if key == 't':
            return self.t
        
        try:
            return self.y[key]
        except:
            try:
                return self.exv_results[key]
            except:
                raise KeyError(key)
    
    def get1d(self, key):
        try:
            return self.y[key]
        except:
            try:
                _, y, *__ = self.exv_results[key]
                return y
            except KeyError:
                raise KeyError(key)
            
    def get2d(self, key):
        try:
            return self.t, self.y[key]
        except:
            try:
                x, y, *_ = self.exv_results[key]
                assert len(x) == len(y)
                return x, y
            except KeyError:
                raise KeyError(key)
            except AssertionError:
                raise ValueError('exv does not have the same length as timespan OR exv is missing an "x" axis component.')
    
    def get_scatter(self, key):
        try:
            return self.t, self.y[key]
        except:
            try:
                x, y, *args = self.exv_results[key]
                if len(args) >= 2:
                    s, c = args
                elif len(args) == 1:
                    s, c = args[0], None
                else:
                    s, c = None, None
                    
                assert len(x) == len(y)
                return x, y, s, c
            except KeyError:
                raise KeyError(key)
            except AssertionError:
                raise ValueError('exv does not have the same length as timespan OR exv is missing an "x" axis component.')
    
    def tabulate(self):
        df       = pd.DataFrame.from_dict(self.y)
        df.index = self.t 
        return df
    
    def variables(self):
        return [*self.y.keys(), *self.exv_results.keys()]

class SimResult(IntResult):
    def __init__(self, model, t, y, p, scenario, _exv_names=None):
        super().__init__(model, t, y, p, scenario, _exv_names)
        
        self.line_args = getattr(model, 'sim_args', {}).get('line_args', {})

###############################################################################
#Supporting Functions for Plotting
###############################################################################
def plot_all_sim_results(all_sim_results, AX, **line_args):
    AX1 = AX
    for model_key, sim_results in all_sim_results.items():
        AX_model = AX.get(model_key, None)
        
        if not AX_model:
            continue
        line_args_model = {k: v.get(model_key, {}) if type(v) == dict else v for k,v in line_args.items()}
        plot_sim_results(sim_results, AX_model, **line_args_model)
    return AX1

def plot_sim_results(sim_results, AX, palette=None, repeat_labels=False, **line_args):
    AX1         = AX
    seen_labels = set()
    for scenario in sim_results:
        for estimate, sim_result in sim_results[scenario].items(): 
            for var, ax_ in AX1.items():
                
                ax             = upp.recursive_get(ax_, scenario, estimate) 
                line_args_     = {**sim_result.line_args, **line_args}
                line_args_     = {k: upp.recursive_get(v, scenario, estimate) for k, v in line_args_.items()}
                
                #Process special keywords
                color = line_args_.get('color')
                if type(color) == str:
                    line_args_['color'] = colors[color]
                    
                label_scheme   = line_args_.get('label', 'scenario, estimate')
                if label_scheme == 'estimate':
                    label = f'{estimate}'
                if label_scheme == 'scenario':
                    label = f'{scenario}'
                elif label_scheme == 'model_key':
                    label = f'{sim_result.model_key}'
                elif label_scheme == 'state':
                    label = f'{var}'
                elif label_scheme == 'scenario, estimate':    
                    label = f'{scenario}, {estimate}'
                elif label_scheme == 'model_key, scenario':
                    label = f'{sim_result.model_key}, {scenario}'
                elif label_scheme == 'model_key, estimate':
                    label = f'{sim_result.model_key}, {estimate}'
                elif label_scheme == 'state, scenario':    
                    label = f'{var}, {scenario}'
                elif label_scheme == 'state, estimate':    
                    label = f'{var}, {estimate}'
                elif label_scheme == 'model_key, state':    
                    label = f'{var}, {var}'
                elif label_scheme == 'model_key, scenario, estimate':
                    label = f'{sim_result.model_key}, {scenario}, {estimate}'
                elif label_scheme == 'state, scenario, estimate':
                    label = f'{var}, {scenario}, {estimate}'
                elif label_scheme == 'model_key, state, scenario':
                    label = f'{sim_result.model_key}, {var}, {scenario}'
                elif label_scheme == 'model_key, state, estimate':
                    label = f'{sim_result.model_key}, {var}, {estimate}'
                elif label_scheme == 'model_key, state, scenario, estimate':
                    label = f'{sim_result.model_key}, {var}, {scenario}, {estimate}'
                else:
                    label = f'{scenario}, {estimate}'
                
                l_key = (ax, label)
                if l_key in seen_labels and not repeat_labels:
                    label = '_nolabel'
                else:
                    seen_labels.add(l_key)
                    
                line_args_['label'] = label
                plot_type           = line_args_.get('plot_type', 'line')
                
                #Plot
                if plot_type == 'line':
                    if line_args_.get('marker', None) and 'linestyle' not in line_args_:
                        line_args_['linestyle'] = 'None'
                    
                    x_vals, y_vals = sim_result.get2d(var)
                    ax.plot(x_vals, y_vals, **line_args_)
                elif plot_type == 'scatter':
                    x_vals, y_vals, s, c = sim_result.get_scatter(var)
                    ax.scatter(x_vals, y_vals, s, c, **line_args_)
                else:
                    raise ValueError(f'Unrecognized plot_type {plot_type}')
                
    return AX1

def make_palette_for_models(all_sim_results, palette_type, base_colors=None, **kwargs):
    return {model_key: make_palette_for_model(sim_results, palette_type, base_colors, **kwargs) for model_key, sim_results in all_sim_results.items()}
import seaborn as sns

def make_palette_for_model(sim_results, palette_type, base_colors=None, **kwargs):
    if base_colors is None:
        base_colors = palette_types['color']('deep', len(sim_results)) 
    elif type(base_colors) == dict:
        base_colors = [colors[ base_colors[s] ] if base_colors[s] == str else base_colors[s] for s in sim_results]
    else:
        base_colors = [colors[c] if type(c) == str else c for c in base_colors]
        
    palette = {}
    for scenario, base_color in zip(sim_results, base_colors):
        
        if palette_type in ['light', 'dark']:
            helper   = palette_types[palette_type]
            palette_ = helper(base_color, len(sim_results[scenario]), **kwargs)
        elif palette_type in ['cubehelix']:
            helper   = palette_types[palette_type]
            palette_ = helper(len(sim_results[scenario]), base_color, **kwargs) 
        else:
            raise ValueError('Invalid palette.')
            
        sns.palplot(palette_)
        palette_ = dict(zip( sim_results[scenario].keys(), palette_) )
        palette[scenario] = palette_
    return palette
