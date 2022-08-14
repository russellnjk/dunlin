import pandas   as pd
import warnings
from .function  import FunctionDict
from typing import Union

import dunlin.standardfile.dunl as sfd
from .reaction   import ReactionDict
from .variable   import VariableDict
from .rate       import RateDict
from .extra      import ExtraDict
from .event      import EventDict 
from .stateparam import StateDict, ParameterDict
from .modeldata  import ModelData
from dunlin.utils.typing import Dflike

class ODEModelData(ModelData):
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
                      'functions' : FunctionDict(functions, namespace), 
                      'variables' : VariableDict(variables, namespace),
                      'reactions' : ReactionDict(reactions, namespace), 
                      'rates'     : RateDict(rates, namespace),                      
                      'events'    : EventDict(events, namespace), 
                      }
        
        #Extra needs to be processed separately because it can be a 
        #user supplied function
        if callable(extra):
            if hasattr(extra, 'names'):
                model_data['extra'] = extra
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
            model_data['extra'] = ExtraDict(extra, namespace)
        
        if units:
            model_data['units'] = units
        
        #Freeze the namespace and add it into model_data
        model_data['namespace'] = frozenset(namespace)
        
        #Add in the remaining arguments as-is
        model_data.update(kwargs)
        
        super().__init__(model_data)
    
    def to_data(self, flattened=True) -> str:
        keys = ['ref', 
                'states', 
                'parameters', 
                'functions', 
                'variables',
                'reactions',
                'rates',
                'events',
                'units'
                ]
        return super().to_data(keys)
   
    