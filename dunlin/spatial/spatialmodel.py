import numpy as np
import pandas as pd
from numba   import njit  
from numbers import Number
from scipy   import spatial
from typing  import Union

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.utils          as ut
import dunlin.comp           as cmp
import dunlin.datastructures as dst
import dunlin.ode.ode_coder  as odc
import dunlin.ode.event      as oev
import dunlin.ode.ivp        as ivp
import dunlin.utils_plot     as upp
from .grid.grid            import RegularGrid, NestedGrid
from .grid.bidict          import One2One, One2Many
from .ratestack            import RateStack
from dunlin.datastructures import SpatialModelData

class SpatialModel:
    default_tspan = np.linspace(0, 1000, 21)
    _df           = ['states', 'parameters']
    _locked       = ['state_names', 'parameter_names', 'functions', 'reactions', 
                     'variables']
    
    @classmethod
    def from_data(cls, all_data: dict, ref: str) -> 'SpatialModel':
        flattened = cmp.flatten_ode(all_data, ref)
        
        return cls(**flattened)
    
    def __init__(self, 
                 ref                 : str, 
                 geometry            : dict,
                 states              : pd.DataFrame, 
                 parameters          : pd.DataFrame, 
                 functions           : dict = None, 
                 variables           : dict = None, 
                 reactions           : dict = None, 
                 rates               : dict = None, 
                 events              : dict = None, 
                 extra               : dict = None, 
                 tspan               : dict = None,
                 compartments        : dict = None, 
                 advection           : dict = None,
                 diffusion           : dict = None,
                 boundary_conditions : dict = None,
                 int_args            : dict = None, 
                 sim_args            : dict = None, 
                 optim_args          : dict = None, 
                 data_args           : dict = None, 
                 meta                : dict = None,
                 dtype               : str = 'spatial',
                 **kwargs
                 ):
        
        if dtype != 'spatial':
            msg = f'Attempted to instantiate {type(self).__name__} with {dtype} data.'
            raise TypeError(msg)
        
        spatial_data = dst.ODEModelData(ref, 
                                        states, 
                                        parameters, 
                                        functions, 
                                        variables, 
                                        reactions, 
                                        rates, 
                                        events, 
                                        extra, 
                                        )
        
        
        stk = RateStack(spatial_data)
        
        #Assign ode attributes
        self._rhs       = stk.rhs
        self._events    = oev.make_events(stk, spatial_data)
        self._extra     = stk.rhsextra
        self._dct       = stk.rhsdct 
        self.namespace  = spatial_data.namespace
        self.variables  = tuple(variables)
        self.functions  = tuple([ut.split_functionlike(f)[0] for f in functions])
        self.reactions  = tuple(reactions)
        self.extra      = () if extra is None else tuple(spatial_data['extra'].keys())
        
        #Assign data-related attributes
        self._state_dict     = spatial_data['states']
        self._parameter_dict = spatial_data['parameters']
        self.ref             = ref
        self.state_names     = tuple(spatial_data['states'].keys())
        self.parameter_names = tuple(spatial_data['parameters'].keys())
        self.states          = states
        self.parameters      = parameters
        self.int_args        = int_args
        self.sim_args        = {} if sim_args   is None else sim_args
        self.optim_args      = {} if optim_args is None else optim_args
        self.data_args       = {} if data_args  is None else data_args
        self.tspan           = tspan
        self.meta            = meta
        
        