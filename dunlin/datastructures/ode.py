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
    _attrs = {'ref'        : (str,),
              'states'     : (StateDict, None),
              'parameters' : (ParameterDict, None),
              'functions'  : (FunctionDict, None),
              'variables'  : (VariableDict, None),
              'reactions'  : (ReactionDict, None),
              'rates'      : (RateDict, None),
              'events'     : (EventDict, None),
              'extra'      : (ExtraDict, callable, None),
              'units'      : (UnitsDict, None),
              'namespace'  : (frozenset,)
              }
    
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
                 units: dict = None
                 ) -> None:
        
        #Set up the data structures
        namespace = set()
        
        self.ref        = ref
        self.states     = StateDict(namespace, states)
        self.parameters = ParameterDict(namespace, parameters)
        
        if functions:
            self.functions = FunctionDict(namespace, functions)
        
        if variables:
            self.variables = VariableDict(namespace, variables)
        
        if reactions:
            self.reactions = ReactionDict(namespace, reactions)
            
        if rates:
            self.rates = RateDict(namespace, rates)
        
        if events:
            self.events = EventDict(namespace, events)
        
        if units:
            self.units = UnitsDict(namespace, units)
        
        if extra:
            #Extra needs to be processed differently because it can be a 
            #user supplied function
            if callable(extra):
                if hasattr(extra, 'names'):
                    repeated = namespace.intersection(extra.names)
                    if repeated:
                        msg = f'Redefinition of {repeated}.'
                        raise NameError(msg)
                    else:
                        namespace.update(extra.names)
                        self.extra = extra
                else:
                    msg = '''Custom extra function missing the "names" attribute. This 
                    attribute should be the names of the variables 
                    returned by the function and allows them to accessed after 
                    simulation.
                    '''
                    msg = msg.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
                    raise ValueError(msg)
            else:
                self.extra = ExtraDict(namespace, extra)
        
        #Check no overlap between rates and reactions
        rate_xs  = set(self.rates.states)
        rxn_xs   = self.reactions.states
        repeated = rate_xs.intersection(rxn_xs)
        
        if repeated:
            msg = f'The following states appear in both a reaction and rate: {repeated}.'
            raise ValueError(msg)
        
        #Freeze the namespace and add it into model_data
        self.namespace = frozenset(namespace)
        
    def to_data(self, recurse=True) -> dict:
        keys = ['states', 
                'parameters', 
                'functions', 
                'variables',
                'reactions',
                'rates',
                'events',
                'units'
                ]
        
        data = {}
        for key in keys:
            value = getattr(self, key, None)
            
            if value is None:
                continue
            elif recurse and hasattr(value, 'to_data'):
                data[key] = value.to_data()
            else:
                data[key] = value
                
        return data
   
    def to_dunl_dict(self) -> dict:
        data = self.to_data(recurse=False)
        data = {self.ref: data}
        
        return data
        