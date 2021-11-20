import numpy  as np

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_model.ode_coder as odc
import dunlin._utils_model.events    as uev
import dunlin._utils_model.exvs      as uex
import dunlin._utils_model.ivp       as ivp

class ODEModel:
    '''
    This class stores the auto-generated functions for ODE integration. During 
    numerical integration, dunlin's Model class will delegate the computation to 
    this class. It is not meant to be used in isolation from the front-end. 
    '''
    default_tspan = np.linspace(0, 1000, 21)
    
    def __init__(self, **model_data):
        
        func_data  = odc.make_ode_data(model_data)
        event_objs = uev.make_events(func_data, model_data)
        exv_objs   = uex.make_exvs(func_data, model_data)
        
        self.rhs    = func_data['rhs']
        self.sim    = func_data['sim']
        self.exvrhs = func_data['exvrhs']
        self.exvs   = exv_objs
        self.events = event_objs
        self.modify = func_data['modify']
        self.eqns   = func_data['eqns']
        
        self.state_names = tuple(model_data['states'].keys())
        self.param_names = tuple(model_data['params'].keys())
        self.model_key   = model_data['model_key']
        
    ###########################################################################
    #Integration
    ###########################################################################
    def __call__(self, *args, **kwargs):
        return self.integrate(*args, **kwargs)
    
    def integrate(self,         y0_dct,     p_dct, tspan,    
                  overlap=True, raw=False,  include_events=True, 
                  **int_args
                  ):
        
        #Reassign and/or extract
        events        = self.events
        exvs          = self.exvs
        state_names   = self.state_names
        param_names   = self.param_names
        model_key     = self.model_key
        default_tspan = self.default_tspan
        
        for scenario in y0_dct:
            t, y = ivp.integrate(self.rhs, 
                                 tspan          = tspan.get(scenario, default_tspan), 
                                 y0             = y0_dct[scenario], 
                                 p              = p_dct[scenario], 
                                 events         = events, 
                                 overlap        = overlap, 
                                 include_events = include_events,
                                 **int_args
                                 )
            
            if raw:
                yield t, y
            else:
                
                yield ODEResult(t, y, p_dct[scenario], state_names, param_names, events, exvs, scenario, model_key)

###############################################################################
#Integration Results
###############################################################################
class ODEResult:
    ###########################################################################
    #Instantiators
    ###########################################################################
    def __init__(self, t, y, p, state_scenarios, param_scenarios, events, exvs, scenario='', model_key=''):
        p              = self.tabulate_params(events, t, y, p)
        p_dict         = dict(zip(param_scenarios, p)) 
        y_dict         = dict(zip(state_scenarios, y))
        self.eval_a    = False
        self.evaluated = {**y_dict, **p_dict, **{'t': t}}
        self.exvs      = exvs
        self._args     = [t, y, p]
        self.scenario  = scenario
        self.model_key = model_key
        self.t         = t
        self.y         = y
        self.p         = p
        
    @classmethod
    def tabulate_params(cls, events, t, y, p):
        '''
        Makes a 2-D array out of a 1-D p. Each row corresponds to one parameter.
        :meta private:
        '''
        record      = cls.get_event_record(events)
        
        if record:
            p_matrix         = np.zeros( (y.shape[1], len(p)) )
            n                = 0
            t_event, p_event = record[n]
            p_curr           = p
            
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
            p_matrix = np.tile(p, (y.shape[1], 1))
            
        return p_matrix.T
    
    @staticmethod
    def get_event_record(events):
        record = [pair for event in events for pair in event.record]
        record = sorted(record, key=lambda x: x[0])
        return record
    
    ###########################################################################
    #Accessors/Lazy Evaluators
    ###########################################################################
    def to_df(self):
        pass
        
    def __getitem__(self, var):
        if type(var) in [tuple, list]:
            return [self.get(v) for v in var]
        else:
            return self.get(var)
    
    def __setitem__(self, var, vals):
        self.evaluated[var] = vals
    
    def get(self, var):
        if var in self.evaluated:
            return self.evaluated[var]
        elif var in self.exvs:
            return self.evaluate(var)
        elif not self.eval_a and var != 'all__':
            temp = self.evaluate('all__', update=False)
            self.evaluated.update(temp)
            return self.evaluated[var]
        else:
            raise KeyError(f'Invalid var: {var}')
    
    def _evaluate_all__(self):
        temp = self.evaluate('all__', update=False)
        for key in temp:
            self.evaluated[key] = temp
            
    def evaluate(self, var, update=True):
        try:
            exv_obj = self.exvs[var]
            result  = exv_obj(*self._args)
            
        except Exception as e:
            msg    = f'Error in evaluating "{var}".'
            args   = (msg,) + e.args
            e.args = ('\n'.join(args),)
            
            raise e
        
        if update:
            self.evaluated[var] = result
        return result
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        unevaluated = [exv for exv in self.exvs if exv not in self.evaluated and exv != 'all__']
        evaluated   = list(self.evaluated.keys())
        scenario    = f'{type(self).__name__}({self.model_key}|{self.scenario})'
        s           = f'{scenario}{{evaluated: {evaluated}, unevaluated: {unevaluated}}}'
        return s
    
    def __repr__(self):
        return self.__str__()
    
    