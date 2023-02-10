import pandas   as pd
import warnings
from typing import Union

import dunlin.comp as cmp
from .reaction   import ReactionDict
from .variable   import VariableDict
from .function   import FunctionDict
from .rate       import RateDict
from .extra      import ExtraDict
from .event      import EventDict 
from .unit       import UnitsDict
from .stateparam import StateDict, ParameterDict
from .modeldata  import ModelData

class ODEModelData(ModelData):
    @classmethod
    def from_all_data(cls, all_data, ref):
        flattened  = cmp.flatten_ode(all_data, ref)
        
        return cls(**flattened)
    
    def __init__(self, 
                 ref: str, 
                 states: Union[dict, pd.DataFrame], 
                 parameters: Union[dict, pd.DataFrame], 
                 functions: dict = None, 
                 variables: dict = None, 
                 reactions: dict = None, 
                 rates: dict = None, 
                 events: dict = None, 
                 extra: dict = None,
                 units: dict = None,
                 **kwargs
                 ) -> None:
        
    
        #kwargs are ignored
        #Set up the data structures
        namespace = set()
        
        self.ref        = ref
        self.states     = StateDict(namespace, states)
        self.parameters = ParameterDict(namespace, parameters)
        self.functions  = FunctionDict(namespace, functions)
        self.variables  = VariableDict(namespace, variables)
        self.reactions  = ReactionDict(namespace, reactions)
        self.rates      = RateDict(namespace, rates)
        self.events     = EventDict(namespace, events)
        self.extra      = ExtraDict(namespace, extra)
        self.units      = UnitsDict(namespace, units)
         
        #Check no overlap between rates and reactions
        if self.rates and self.reactions:
            rate_xs  = set(self.rates.states)
            rxn_xs   = self.reactions.states
            repeated = rate_xs.intersection(rxn_xs)
        
            if repeated:
                msg = f'The following states appear in both a reaction and rate: {repeated}.'
                raise ValueError(msg)
        
        #Freeze the namespace and save it
        self.namespace = frozenset(namespace)
        
        #Freeze the attributes
        self.freeze()
        
    def to_data(self, recurse=True) -> dict:
        keys = ['states', 
                'parameters', 
                'functions', 
                'variables',
                'reactions',
                'rates',
                'events',
                'extra',
                'units'
                ]
        
        return self._to_data(keys, recurse)
   
