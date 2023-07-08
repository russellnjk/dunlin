import pandas   as pd
import warnings
from typing import Union

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
        
        #Set up the data structures
        namespace = set()
        
        self.ref          = ref
        self.states       = StateDict(namespace, states)
        self.parameters   = ParameterDict(namespace, parameters)
        self.functions    = FunctionDict(namespace, functions)
        self.variables    = VariableDict(namespace, variables)
        self.reactions    = ReactionDict(namespace, reactions)
        self.rates        = RateDict(namespace, rates)
        self.events       = EventDict(namespace, events)
        self.units        = UnitsDict(namespace, units)
        self.compartments = compartments
        self.meta         = meta
        self.int_args     = int_args
        self.sim_args     = sim_args
        self.opt_args     = opt_args
         
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
        
        