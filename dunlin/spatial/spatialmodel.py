import matplotlib.axes as axes
import numpy           as np
import pandas          as pd
from numba   import njit  
from numbers import Number
from scipy   import spatial
from typing  import Union

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.utils          as ut
import dunlin.comp           as cmp
import dunlin.utils_plot     as upp
from .stack.eventstack import (EventStack,
                               Domain_type, Domain, Voxel, 
                               State, Parameter,
                               Surface_type,
                               )
from ..datastructures  import SpatialModelData
from ..ode.basemodel   import BaseModel
from ..ode.odemodel    import ODEResult

###############################################################################
#Typing
###############################################################################
Scenario = str|Number

###############################################################################
#Model
###############################################################################
class SpatialModel(BaseModel):
    default_tspan = np.linspace(0, 1000, 21)
    _df           = ['states', 'parameters']
    _locked       = BaseModel._locked + ['advection', 
                                         'diffusion', 
                                         'boundary_conditions',
                                         'coordinate_components',
                                         'grid_config',
                                         'domain_types',
                                         'surfaces',
                                         'geometry_definitions',
                                         ]
    _dtype        = 'spatial'
    
    @classmethod
    def from_data(cls, all_data: dict, ref: str) -> 'SpatialModel':
        flattened = cmp.flatten_model(all_data, ref, SpatialModelData.required_fields)
        
        return cls(**flattened)
    
    def __init__(self, 
                 ref                   : str, 
                 states                : dict|pd.DataFrame, 
                 parameters            : dict|pd.DataFrame, 
                 coordinate_components : dict,
                 grid_config           : dict,
                 domain_types          : dict,
                 geometry_definitions  : dict,
                 surfaces              : dict = None,
                 functions             : dict = None, 
                 variables             : dict = None, 
                 reactions             : dict = None, 
                 rates                 : dict = None, 
                 events                : dict = None, 
                 tspans                : dict = None,
                 units                 : dict = None,
                 advection             : dict = None,
                 diffusion             : dict = None,
                 boundary_conditions   : dict = None,
                 meta                  : dict = None,
                 int_args              : dict = None,
                 sim_args              : dict = None,
                 opt_args              : dict = None,
                 trace_args            : dict = None,
                 dtype                 : str  = 'spatial'
                 ):
        
        #Parse the data using the datastructures submodule
        spatial_data = SpatialModelData(ref, 
                                        states, 
                                        parameters, 
                                        coordinate_components,
                                        grid_config,
                                        domain_types,
                                        geometry_definitions,
                                        surfaces,
                                        functions, 
                                        variables, 
                                        reactions, 
                                        rates, 
                                        events, 
                                        units,
                                        advection,
                                        diffusion,
                                        boundary_conditions,
                                        meta,
                                        int_args,
                                        sim_args,
                                        opt_args,
                                        trace_args
                                        )
        
        stk      = EventStack(spatial_data)
        self.stk = stk
        
        #Call the parent constructors to save key information
        BaseModel.__init__(self,
                           model_data = spatial_data, 
                           ref        = ref, 
                           tspans     = tspans, 
                           int_args   = int_args, 
                           dtype      = dtype, 
                           events     = stk._events
                           )
        
        #Assign rhs 
        self._rhs_functions    = stk._rhs_funcs
        self._rhsdct_functions = stk._rhsdct_funcs
        
        #Viewable but not editable by front-end users
        #To be set once during instantiation and subsequently locked
        self.states     = tuple(spatial_data.states)
        self.parameters = tuple(spatial_data.parameters)
        self.functions  = tuple(spatial_data.functions)
        self.variables  = tuple(spatial_data.variables)
        self.reactions  = tuple(spatial_data.reactions)
        self.events     = tuple(spatial_data.events)
        self.meta       = spatial_data.meta
        
        #For back-end only
        #For storing state and parameters
        self.state_df     = states
        self.parameter_df = parameters
        
        #Specific to this class
        self.int_args   = {} if spatial_data.int_args   is None else spatial_data.int_args
        self.sim_args   = {} if spatial_data.sim_args   is None else spatial_data.sim_args
        self.opt_args   = {} if spatial_data.opt_args   is None else spatial_data.opt_args
        self.trace_args = {} if spatial_data.trace_args is None else spatial_data.trace_args 
        
    ###########################################################################
    #Integration
    ###########################################################################
    def __call__(self, 
                 y0             : np.ndarray, 
                 p0             : np.ndarray, 
                 tspan          : np.ndarray, 
                 raw            : bool = False, 
                 include_events : bool = True,
                 scenario       : str  = ''
                 ) -> np.array:
        
        y0_expanded = self.stk.expand_init(y0)
        
        return super().__call__(y0_expanded,
                                p0,
                                tspan,
                                raw,
                                include_events,
                                scenario
                                )
    
    def _convert_call(self, 
                      t : np.ndarray, 
                      y : np.ndarray, 
                      p : np.ndarray,
                      c : Scenario
                      ) -> 'SpatialResult':
        return SpatialResult(t, y, p, c, self)
    
    # def integrate(self, 
    #               scenarios      : list = None,
    #               raw            : bool = False, 
    #               include_events : bool = True,
    #               _y0            : dict[Scenario, np.ndarray] = None,
    #               _p0            : dict[Scenario, np.ndarray] = None,
    #               _tspans        : dict[Scenario, np.ndarray] = None
    #               ) -> Union[dict, 'SpatialResultDict']:
    #     pass
    
    def _convert_integrate(self, 
                           scenario2intresult: dict[Scenario, 'SpatialResult']
                           ) -> 'SpatialResultDict':
        return SpatialResultDict(scenario2intresult, self)

###############################################################################
#Integration Results
###############################################################################
class SpatialResult(ODEResult):
    def __init__(self, 
                 t     : np.ndarray, 
                 y     : np.ndarray, 
                 p     : np.ndarray, 
                 c     : Scenario,
                 model : SpatialModel
                 ):
        super().__init__(t, y, p, c, model)

class SpatialResultDict:
    def __init__(self, 
                 scenario2intresult : dict[Scenario, SpatialResult],
                 model              : SpatialModel
                 ):
        self.scenario2intresult = scenario2intresult
    
    def __getitem__(self, key: Scenario) -> SpatialResult:
        return self.scenario2intresult[key]
    
    def plot_line(self,
                  ax        : axes.Axes,
                  var       : str|tuple[str, str],
                  scenarios : Scenario|list[Scenario] = None,
                  **kwargs
                  ) -> axes.Axes:
        
        match scenarios:
            case None:
                scenarios = set(self.scenario2intresult)
            case [*scenarios]:
                scenarios = set(scenarios)
            case c if isinstance(c, Scenario):
                scenarios = {c}
            case _:
                msg = ''
                raise ValueError(msg)
        
        result = {}
        for scenario, intresult in self.scenario2intresult.items():
            if scenario not in scenarios:
                continue
            
            result[scenario] = intresult.plot_line(ax, var, **kwargs)
        
        return result
     

        