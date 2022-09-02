import pandas   as pd
import warnings
from .function  import FunctionDict
from typing import Union

import dunlin.comp as cmp
from .reaction   import ReactionDict
from .variable   import VariableDict
from .rate       import RateDict
from .extra      import ExtraDict
from .event      import EventDict 
from .unit       import UnitsDict
from .stateparam import StateDict, ParameterDict
from .modeldata  import ModelData

class ODEModelData(ModelData):
    @classmethod
    def from_all_data(cls, all_data, ref, **kwargs):
        keys = ['ref',
                'states',
                'parameters',
                'functions',
                'variables',
                'reactions',
                'rates',
                'events',
                'extra',
                'units'
                ]
        args = {}
        
        flattened  = cmp.flatten_ode(all_data, ref)
        
        temp = flattened
        for key in keys:
            if key in temp:
                args[key] = temp[key]
        
        model_data = cls(**args, **kwargs)
        
        return model_data
    
    def __init__(self, 
                 ref: str, 
                 states: Union[dict, pd.DataFrame], 
                 parameters: Union[dict, pd.DataFrame], 
                 functions: dict = None, 
                 variables: dict = None, 
                 reactions: dict = None, 
                 rates: dict = None, 
                 events: dict = None, 
                 extra: Union[dict, callable] = None,
                 units: dict = None,
                 **kwargs
                 ) -> None:
        
        #Set up the data structures
        namespace = set()
        
        model_data = {'ref'       : ref,
                      'states'    : StateDict(namespace, states), 
                      'parameters': ParameterDict(namespace, parameters), 
                      
                      }
        
        #Process optionals
        model_data['functions'] = FunctionDict(namespace, functions)
        model_data['variables'] = VariableDict(namespace, variables)
        model_data['reactions'] = ReactionDict(namespace, reactions)
        model_data['rates']     = RateDict(namespace, rates)
        model_data['events']    = EventDict(namespace, events)
        
        if units:
            model_data['units'] = UnitsDict(namespace, units)
        
        #Extra needs to be processed differently because it can be a 
        #user supplied function
        if extra is None:
            pass
        elif callable(extra):
            if hasattr(extra, 'names'):
                model_data['extra'] = extra
                repeated = namespace.intersection(extra.names)
                if repeated:
                    msg = f'Redefinition of {repeated}.'
                else:
                    namespace.update(extra.names)
            else:
                msg = '''Custom extra function missing the "names" attribute. This 
                attribute should be the names of the variables 
                returned by the function and allows them to accessed after 
                simulation.
                '''
                msg = msg.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
                raise ValueError(msg)
        else:
            model_data['extra'] = ExtraDict(namespace, extra)
        
        #Check no overlap between rates and reactions
        rate_xs  = set(model_data['rates'].states)
        rxn_xs   = model_data['reactions'].states
        repeated = rate_xs.intersection(rxn_xs)
        
        if repeated:
            msg = f'The following states appear in both a reaction and rate: {repeated}.'
            raise ValueError(msg)
        
        #Freeze the namespace and add it into model_data
        model_data['namespace'] = frozenset(namespace)
        
        #Add in the remaining arguments as-is
        model_data.update(kwargs)
        
        super().__init__(model_data)
    
    def to_data(self, recurse=True, _extend=None) -> dict:
        keys = ['states', 
                'parameters', 
                'functions', 
                'variables',
                'reactions',
                'rates',
                'events',
                'units'
                ]
        if _extend:
            keys = keys + list(_extend)
            
        return super()._to_data(keys, recurse)
   
    