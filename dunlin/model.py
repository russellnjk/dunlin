import numpy  as np
import pandas as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin._utils_model.ode_coder       as odc
import dunlin._utils_model.events          as uev
import dunlin._utils_model.base_error      as dbe
import dunlin._utils_model.dun_file_reader as dfr
import dunlin._utils_model.ivp             as ivp

###############################################################################
#Main Instantiation Algorithm
###############################################################################
def read_file(filename, _check_sub=True, _parse=True):
    if _parse:
        dun_data   = dfr.read_file(filename, _parse=_parse)
        models     = make_models(dun_data, _check_sub=_check_sub)
        
        return dun_data, models
    else:
        return dun_data

def make_models(dun_data, _check_sub=True):
    models     = {}
    for model_key, model_data in dun_data.items():
        model             = Model(**model_data, _check_sub=False)
        models[model_key] = model
    
    if _check_sub:
            for model_key in models:
                Model._check_sub(model_key)
    return models

###############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinModelError(dbe.DunlinBaseError):
    @classmethod
    def locked(cls, attr):
        msg = "Model object's {} attribute is locked. Instantiate a new object if you wish to change it.".format(attr)
        return cls.raise_template(msg, 0)
    
    @classmethod
    def mismatch(cls, attr, expected, received):
        msg = f'Expected keys/columns in {attr}: {expected}\nReceived: {received}'
    
    @classmethod
    def invalid_attr(cls, msg):
        return cls.raise_template(msg, 1)
    
    @classmethod
    def submodel_missing(cls, model_key, submodel_key):
        return cls.raise_template(f'{model_key} calls {submodel_key} but the submodel is missing.', 10)
                     
    @classmethod
    def submodel_len(cls, model_key, submodel_key, arg):
        return cls.raise_template(f'{model_key} calls {submodel_key} but the {arg} argument is of the wrong length.', 11)
    
    @classmethod
    def submodel_recursion(cls, *chain):
        chain_ = ' -> '.join([str(c) for c in chain])
        return cls.raise_template('Recursive model hierarchy: ' + chain_, 12)

###############################################################################
#Dunlin Models
###############################################################################        
class Model():
    #Hierarchy management
    _cache = {}
    _sub   = {}
    
    #Attribute management
    _locked  = ['model_key', 'rxns', 'vrbs', 'funcs']
    _df      = ['states', 'params']
    _default = {'int_args': {'method'  : 'LSODA'}
                }
    _tspan   = np.linspace(0, 100, 11)
    
    
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
            raise DunlinModelError.submodel_recursion(*_super, model_key)
            
        subs = cls._sub[model_key]
        
        for (sub, y_args, p_args) in subs:
            #Check if submodel exists
            if sub not in cls._cache:
                raise DunlinModelError.submodel_missing(model_key, sub)
            
            #Check number of substates and subparams
            if len(cls._cache[sub]._states) != y_args:
                raise DunlinModelError.submodel_len(model_key, sub, 'states(y)')
            elif len(cls._cache[sub]._params) != p_args:
                raise DunlinModelError.submodel_len(model_key, sub, 'params(p)')
            
            cls._check_sub(sub, _super=_super+(model_key,))
            
    def __init__(self, model_key, states, params, rxns=None, vrbs=None, funcs=None, rts=None, exvs=None, modify=None, events=None, tspan=dict(), _check_sub=True, **kwargs):
        #Set the locked and constrained attributes
        self._states     = tuple(states.keys()) 
        self._params     = tuple(params.keys())
        self._states_set = set(states.keys()) 
        self._params_set = set(params.keys())
        
        self.states = states
        self.params = params
        
        self._record = []
        
        #Set the remaining attributes
        super().__setattr__('model_key', model_key)
        super().__setattr__('rxns',      rxns     )
        super().__setattr__('vrbs',      vrbs     )
        super().__setattr__('funcs',     funcs    )
        super().__setattr__('rts',       rts      )
        super().__setattr__('exvs',      exvs     )
        super().__setattr__('modify',    modify   )
        super().__setattr__('events',    events   )
        super().__setattr__('tspan',     tspan    )
                
        for k, v in {**self._default, **kwargs}.items():
            super().__setattr__(k, v)
        
        #Create functions
        #In the future, extend functionality by using model_type
        model_data = self.to_dict()
        func_data  = odc.make_ode_data(model_data)
        event_objs = uev.make_events(func_data, model_data)
        
        self._rhs_code, self._rhs = func_data['rhs']
        self._events              = event_objs
        self._modify              = func_data['modify'][1] if func_data['modify'] else None
        self._exvs                = func_data['exvs'][1]
        
        #Track model and submodels 
        self._sub[model_key]   = self._find_submodels(model_data)
        self._cache[model_key] = self 
        
        if _check_sub:
            self._check_sub(model_key)
        
    def __setattr__(self, attr, value):
        if attr in self._locked:
            raise DunlinModelError.locked(attr)
        
        elif attr in self._df:
            _keys = getattr(self, '_' + attr)
            _set  = getattr(self, '_' + attr + '_set')
            try:
                keys = value.keys()
            except:
                raise TypeError('Not a dict/DataFrame.')
            
            if len(_set.intersection(keys)) != len(_set):
                raise DunlinModelError.mismatch(attr, list(_keys), list(keys))
            
            data = {k: value[k] for k in _keys}
            try:
                result = pd.DataFrame.from_dict(data, orient='columns')
            except:
                try:
                    result = pd.DataFrame.from_dict({0: data}, orient='index')
                except:
                    raise DunlinModelError.value('states')
            super().__setattr__(attr, result)        
        else:
            super().__setattr__(attr, value)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        return self.__setattr__(key, value)
    
    def __len__(self):
        return len(self._states), len(self._params)
    
    def __repr__(self):
        return f'{type(self).__name__} {self.model_key}<states: {self._states}, params: {self._params}>'
    
    def __str__(self):
        return self.__repr__()
    
    #MIGHT CHANGE THIS IN THE FUTURE
    def __call__(self, *args, **kwargs):
        return self._rhs(*args, **kwargs)
    
    def get_exv(self, exv_name):
        return self._exvs[exv_name]
    
    def get_param_names(self):
        return self._params
    
    def get_state_names(self):
        return self._states
    
    def to_dict(self):
        result = {k: v for k, v in self.__dict__.items() if k[0] != '_' }
        return result
    
    def keys(self):
        return self.to_dict().keys()
    
    def values(self):
        return self._to_dict().values()
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def setdefault(self, key, default=None):
        try:
            return getattr(self, key)
        except:
            setattr(self, key, default)
        return default
    
    def items(self):
        return self.to_dict().items()
    
    def copy(self):
        args = self.to_dict()
        return type(self)(**args)
    
    def integrate(self,         scenario, 
                  states_array, params_array, 
                  overlap=True, include_events=True, 
                  tspan=None,   modify=None,    
                  events=None,  **int_args):
    
        t, y = ivp.integrate(self._rhs, 
                             tspan          = self.tspan.get(scenario, self._tspan) if tspan is None else tspan, 
                             y0             = states_array, 
                             p              = params_array, 
                             events         = self._events if events is None else modify, 
                             modify         = self._modify if modify is None else modify,
                             overlap        = overlap, 
                             include_events = include_events,
                             scenario       = scenario,
                             **{**self.int_args, **int_args}
                             )
        
        return t, y
    
   
        
