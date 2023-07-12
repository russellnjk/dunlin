import pandas   as pd
import warnings
from datetime import datetime
from numbers  import Number
from typing   import Any, Union

import dunlin.comp as cmp
from .reaction   import ReactionDict
from .variable   import VariableDict
from .function   import FunctionDict
from .rate       import RateDict
from .event      import EventDict 
from .unit       import UnitsDict
from .stateparam import StateDict, ParameterDict
from .modeldata  import ModelData

class ODEModelData(ModelData):
    @classmethod
    def from_all_data(cls, all_data, ref):
        required_fields = {'states'     : [True, False, False],
                           'parameters' : [True, False, False],
                           'functions'  : [True, False],
                           'variables'  : [True, True],
                           'reactions'  : [True, True, True],
                           'rates'      : [True, True],
                           'events'     : [True, False, True]
                           }
        
        flattened  = cmp.flatten_model(all_data, ref, required_fields)
        
        return cls(**flattened)
    
    @classmethod
    def deep_copy(cls, 
                  name   : str, 
                  data   : Union[dict, list, str, Number, datetime, None],
                  _first : bool = True
                  ) -> Any:
        if data is None and _first:
            return data

        elif type(data) == dict:
            result = {}
            for k, v in data.items():
                k_ = cls.deep_copy(name, k, False)
                v_ = cls.deep_copy(name, v, False)
                
                result[k_] = v_
            return result
        
        elif type(data) == list or type(data) == tuple:
            return [cls._deep_copy(name, x, False) for x in data]
        elif isinstance(data, (Number, str, datetime)):
            return data
        else:
            msg  = 'Error when parsing {name}. '
            msg += 'Expected a dict, list, str, number or datetime. '
            msg += f'Received {type(data)}.'
            raise TypeError(msg)
            
    def __init__(self, 
                 ref          : str, 
                 states       : Union[dict, pd.DataFrame], 
                 parameters   : Union[dict, pd.DataFrame], 
                 functions    : dict = None, 
                 variables    : dict = None, 
                 reactions    : dict = None, 
                 rates        : dict = None, 
                 events       : dict = None, 
                 units        : dict = None,
                 compartments : dict = None,
                 meta         : dict = None,
                 int_args     : dict = None,
                 sim_args     : dict = None,
                 opt_args     : dict = None
                 ) -> None:
        
        #Call the parent constructor to specify which attributes are exportable
        #This allows the input to be reconstructed assuming no modification
        #after instantiation
        super().__init__(['ref', 
                          'states', 
                          'parameters', 
                          'functions', 
                          'variables',
                          'reactions',
                          'rates',
                          'events',
                          'units',
                          'compartments',
                          'meta',
                          'int_args',
                          'sim_args',
                          'opt_args'
                          ]
                         )
        
        #Set up the core data structures
        all_names = self._init_core(ref,
                                    states,
                                    parameters,
                                    functions,
                                    variables,
                                    reactions,
                                    rates,
                                    events,
                                    units
                                    )
        
        #Set up the remainder
        #We expect compartments to be processed differently in spatial
        self.compartments = compartments
        self.meta         = self.deep_copy('meta', meta)
        self.int_args     = self.deep_copy('int_args', int_args)
        self.sim_args     = self.deep_copy('sim_args', sim_args)
        self.opt_args     = self.deep_copy('opt_args', opt_args)
         
        
        #Freeze all_names and save it
        self.all_names = frozenset(all_names)
    
    def _init_core(self, 
                   ref          : str, 
                   states       : Union[dict, pd.DataFrame], 
                   parameters   : Union[dict, pd.DataFrame], 
                   functions    : dict = None, 
                   variables    : dict = None, 
                   reactions    : dict = None, 
                   rates        : dict = None, 
                   events       : dict = None, 
                   units        : dict = None,
                   ) -> set:
        
        #Set up the data structures
        all_names = set()
        
        self.ref          = ref
        self.states       = StateDict(all_names, states)
        self.parameters   = ParameterDict(all_names, parameters)
        self.functions    = FunctionDict(all_names, functions)
        self.variables    = VariableDict(all_names, variables)
        self.reactions    = ReactionDict(all_names, self.states, reactions)
        self.rates        = RateDict(all_names, self.states, rates)
        self.events       = EventDict(all_names, events)
        self.units        = None if units is None else UnitsDict(self.states, parameters, units)
        
        #Check no overlap between rates and reactions
        if self.rates and self.reactions:
            rate_xs  = set(self.rates.states)
            rxn_xs   = self.reactions.states
            repeated = rate_xs.intersection(rxn_xs)
        
            if repeated:
                msg = f'The following states appear in both a reaction and rate: {repeated}.'
                raise ValueError(msg)
        
        return all_names