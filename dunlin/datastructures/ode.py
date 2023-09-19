import pandas   as pd

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
    required_fields = {'states'     : [True, False, False],
                       'parameters' : [True, False, False],
                       'functions'  : [True, False],
                       'variables'  : [True, True],
                       'reactions'  : [True, False, True, False],
                       'rates'      : [True, True],
                       'events'     : [True, False, True, True]
                       }
    
    @classmethod
    def from_all_data(cls, all_data, ref):
        flattened  = cmp.flatten_model(all_data, ref, cls.required_fields)
        return cls(**flattened)
    
    
    def __init__(self, 
                 ref          : str, 
                 states       : dict|pd.DataFrame, 
                 parameters   : dict|pd.DataFrame, 
                 functions    : dict = None, 
                 variables    : dict = None, 
                 reactions    : dict = None, 
                 rates        : dict = None, 
                 events       : dict = None, 
                 units        : dict = None,
                 domain_types : dict = None,
                 tspans       : dict = None,
                 meta         : dict = None,
                 int_args     : dict = None,
                 sim_args     : dict = None,
                 opt_args     : dict = None,
                 trace_args   : dict = None,
                 data_args    : dict = None
                 ) -> None:
        
        #Call the parent constructor to specify which attributes are exportable
        #This allows the input to be reconstructed assuming no modification
        #after instantiation
        super()._set_exportable_attributes(['ref', 
                                            'states', 
                                            'parameters', 
                                            'functions', 
                                            'variables',
                                            'reactions',
                                            'rates',
                                            'events',
                                            'units',
                                            'domain_types',
                                            'meta',
                                            'int_args',
                                            'sim_args',
                                            'opt_args',
                                            'trace_args',
                                            'data_args'
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
        #We expect compartments/domain_types to be processed differently in spatial
        self.domain_types = self.deep_copy('domain_types', domain_types)
        self.meta         = self.deep_copy('meta', meta)
        self.int_args     = self.deep_copy('int_args', int_args)
        self.sim_args     = self.deep_copy('sim_args', sim_args)
        self.opt_args     = self.deep_copy('opt_args', opt_args)
        self.trace_args   = self.deep_copy('trace_args', trace_args)
        self.data_args    = self.deep_copy('data_args', data_args)
        
        #Freeze all_names and save it
        self.all_names = frozenset(all_names)
    
    def _init_core(self, 
                   ref        : str, 
                   states     : dict|pd.DataFrame, 
                   parameters : dict|pd.DataFrame, 
                   functions  : dict = None, 
                   variables  : dict = None, 
                   reactions  : dict = None, 
                   rates      : dict = None, 
                   events     : dict = None, 
                   units      : dict = None,
                   tspans     : dict = None
                   ) -> set:
        
        #Set up the data structures
        all_names = set()
        
        self.ref        = ref
        self.states     = StateDict(all_names, states)
        self.parameters = ParameterDict(all_names, parameters)
        self.functions  = FunctionDict(all_names, functions)              
        self.variables  = VariableDict(all_names, variables)              
        self.reactions  = ReactionDict(all_names, self.states, reactions) 
        self.rates      = RateDict(all_names, self.states, rates)         
        self.events     = EventDict(all_names, states, parameters, events)                    
        self.units      = UnitsDict(self.states, parameters, units) if units     else None
        self.tspans     = self.deep_copy('tspans', tspans)
        
        #Check no overlap between rates and reactions
        if self.rates and self.reactions:
            rate_xs  = set(self.rates.states)
            rxn_xs   = self.reactions.states
            repeated = rate_xs.intersection(rxn_xs)
        
            if repeated:
                msg = f'The following states appear in both a reaction and rate: {repeated}.'
                raise ValueError(msg)
        
        return all_names