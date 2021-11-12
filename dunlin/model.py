import numpy  as np
import pandas as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_model.ode_classes as umo
import dunlin.standardfile           as stf

###############################################################################
#Main Instantiation Algorithm
###############################################################################
def read_file(*filenames, **kwargs):
    
    dun_data   = stf.read_file(*filenames)
    models     = make_models(dun_data, **kwargs)
    
    return dun_data, models

def make_models(dun_data, _check_sub=True):
    models = {section['model_key'] : Model(**section) for section in dun_data if 'model_key' in section}
    
    if _check_sub:
            [model._check_sub(model.model_key) for model in models.values()]
    return models

###############################################################################
#Dunlin Model
###############################################################################        
class Model:
    '''
    This is the front-end class for representing a model.
    '''
    #Hierarchy management
    _cache = {}
    _sub   = {}
    
    #Attribute management
    _checkkw = True
    _locked  = ['model_key', 'rxns', 'vrbs', 'funcs', 'rts']
    _df      = ['states', 'params']
    _kw      = {'int_args'          : {'method'  : 'LSODA'},
                'sim_args'          : {},
                'optim_args'        : {},
                'strike_goldd_args' : {},
                }
    
    ###############################################################################
    #Hierarchy Tracking
    ###############################################################################
    @staticmethod
    def _find_submodels(model_data):
        rxns = model_data['rxns']
        subs = [] 
        if not rxns:
            return subs
        for rxn_args in rxns.values():
            if hasattr(rxn_args, 'items'):
                sub = rxn_args.get('submodel')
                if sub:
                    sub = [sub, len(rxn_args['substates']), len(rxn_args['subparams'])]
                    subs.append(sub)
                elif 'submodel ' in rxn_args.get('rxn', ''):
                    sub = [rxn_args['rxn'][9:], len(rxn_args['substates']), len(rxn_args['subparams'])]
                    subs.append(sub)
            else:
                if 'submodel ' in rxn_args[0]:
                    sub = rxn_args[0][9:], len(rxn_args[1]), len(rxn_args[2])
                    subs.append(sub)
        return subs
    
    @classmethod
    def _check_sub(cls, model_key, _super=()):
        if model_key in _super:
            raise SubmodelRecursionError(*_super, model_key)
            
        subs = cls._sub[model_key]
        
        for (sub, y_args, p_args) in subs:
            #Check if submodel exists
            if sub not in cls._cache:
                raise MissingSubmodelError(model_key, sub)
            
            #Check number of substates and subparams
            if len(cls._cache[sub].get_state_names()) != y_args:
                raise SubmodelLenError(model_key, sub, 'states(y)')
            elif len(cls._cache[sub].get_param_names()) != p_args:
                raise SubmodelLenError(model_key, sub, 'params(p)')
            
            cls._check_sub(sub, _super=_super+(model_key,))
    
    ###############################################################################
    #Instantiation
    ###############################################################################  
    def __init__(self,      model_key,   states,       params, 
                 rxns=None, vrbs=None,   funcs=None,   rts=None, 
                 exvs=None, events=None, tspan=None,
                 **kwargs
                 ):
        
        #Set the locked attributes using the super method
        tspan_ = {} if tspan is None else {}
        
        super().__setattr__('model_key', model_key)
        super().__setattr__('_states_tuple', tuple(states.keys()))
        super().__setattr__('_params_tuple', tuple(params.keys()))
        super().__setattr__('rxns',      rxns     )
        super().__setattr__('vrbs',      vrbs     )
        super().__setattr__('funcs',     funcs    )
        super().__setattr__('rts',       rts      )
        super().__setattr__('exvs',      exvs     )
        super().__setattr__('events',    events   )
        super().__setattr__('tspan',     tspan_   )
        
        #Set property based attributes
        self.states      = states
        self.params      = params
        
        #Set analysis settings
        for k, v in {**self._kw, **kwargs}.items():
            if k not in self._kw and self._checkkw:
                msg = f'Attempted to instantiate Model with invalid attribute: {k}'
                raise AttributeError(msg)
            super().__setattr__(k, v)
        
        #Check types
        if any([type(x) != str for x in self._states_tuple]):
            raise NameError('States can only have strings as names.')
        if any([type(x) != str for x in self._params_tuple]):
            raise NameError('Params can only have strings as names.')
        
        #Prepare dict to create functions
        model_data = self.to_dict()
        
        #Create functions
        super().__setattr__('ode', umo.ODEModel(**model_data))
        
        #Set mode
        self._mode = 'ode'
        
        #Track model and submodels 
        self._sub[model_key]   = self._find_submodels(model_data)
        self._cache[model_key] = self 
        
    def new(self, **kwargs):
        args = self.to_dict()
        args = {**args, **kwargs} 
        return type(self)(**args)
    
    ###############################################################################
    #Attribute Management
    ###############################################################################
    def _df2dict(self, attr, value):
        if type(value) in [dict, pd.DataFrame]:
            df = pd.DataFrame(value)
        elif type(value) == pd.Series:
            df = pd.DataFrame(value).T
        else:
            raise TypeError(f"Model object's '{attr} attribute can be assigned using dict, DataFrame or Series.")
        
        #Check values
        if df.isnull().values.any():
            raise ValueError('Missing or NaN values.')
        
        #Extract values
        keys = list(getattr(self, '_' + attr + '_tuple'))
        try:
            df = df[keys]
        except KeyError:
            raise ModelMismatchError(keys, df.keys())
        
        #Save as dict
        return dict(zip(df.index, df.values))
    
    def _dict2df(self, attr):
        dct = getattr(self, '_'+attr)
        # df  = pd.DataFrame(dct).from_dict(dct, 'index')
        df  = pd.DataFrame(dct).T
        
        df.columns = getattr(self, '_'+attr+'_tuple')
        return df
    
    @property
    def states(self):
        return self._dict2df('states')
    
    @states.setter
    def states(self, df):
        self._states = self._df2dict('states', df)
    
    @property
    def params(self):
        return self._dict2df('params')
    
    @params.setter
    def params(self, df):
        self._params = self._df2dict('params', df)
    
    @property
    def state_names(self):
        return self._states_tuple
    
    @state_names.setter
    def state_names(self):
        return AttributeError('State names are locked.')
    
    @property
    def param_names(self):
        return self._params_tuple
    
    @param_names.setter
    def param_names(self):
        return AttributeError('Parameter names are locked.')
    
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ['ode']:
            self._mode = value
        else:
            raise ValueError(f'Invalid mode: {value}')
        
    def __setattr__(self, attr, value):
        if attr in self._locked:
            raise AttributeError(f'{attr} attribute is locked.')
        else:
            super().__setattr__(attr, value)
    
    def get_param_names(self):
        return self._params_tuple
    
    def get_state_names(self):
        return self._states_tuple
    
    ###############################################################################
    #Dict-like Behaviour
    ###############################################################################
    def to_dict(self):
        result           = {k: v for k, v in self.__dict__.items() if k[0] != '_' }
        result['states'] = self.states
        result['params'] = self.params
        return result
    
    def keys(self):
        return self.to_dict().keys()
    
    def values(self):
        return self.to_dict().values()
    
    def items(self):
        return self.to_dict().items()
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def setdefault(self, key, default=None):
        try:
            return getattr(self, key)
        except:
            setattr(self, key, default)
        return default
    
    ###############################################################################
    #Representation
    ###############################################################################
    def __repr__(self):
        return f'{type(self).__name__} {self.model_key}{{states: {self.get_state_names()}, params: {self.get_param_names()}}}'
    
    def __str__(self):
        return self.__repr__()
    
    def __len__(self):
        return len(self._states), len(self._params)
    
    ###########################################################################
    #Integration
    ###########################################################################
    def __call__(self, *args, **kwargs):
        if self._mode == 'ode':
            return self.integrate_ode(*args, **kwargs)
    
    def integrate_ode(self,        y0=None,     p=None,     overlap=True,  
                  raw=False,  
                  include_events=True
                  ):
        
        #Reassign and/or extract
        y0_dct   = self._states      if y0     is None else y0
        p_dct    = self._params      if p      is None else p  
        int_args = self.int_args
        tspan    = self.tspan
        
        return self.ode(y0_dct, p_dct, tspan, overlap, raw, include_events, **int_args)
        
class ModelMismatchError(Exception):
    def __init__(self, expected, received):
        super().__init__(f'Required keys: {list(expected)}. Recevied: {list(received)}')

class MissingSubmodelError(Exception):
    def __init__(self, model_key, submodel_key):
        super().__init__(f'{model_key} calls {submodel_key} but the submodel is missing.')

class SubmodelLenError(Exception):
    def __init__(self, model_key, submodel_key, arg):
        super().__init__(f'{model_key} calls {submodel_key} but the {arg} argument is of the wrong length.')

class SubmodelRecursionError(Exception):
    def __init__(self, *chain):
        chain_ = ' -> '.join([str(c) for c in chain])
        super().__init__('Recursive model hierarchy: ' + chain_)
        
