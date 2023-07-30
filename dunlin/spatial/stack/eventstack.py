import numpy as np
from numbers import Number
from typing  import Callable

import dunlin.utils          as ut
import dunlin.datastructures as dst
from ..grid.grid import RegularGrid, NestedGrid
from .bidict     import One2One, Many2One
from .ratestack  import (RateStack,
                         Domain_type, Domain, Voxel, 
                         State, Parameter,
                         Surface_type
                         )

from dunlin.datastructures import SpatialModelData
from dunlin.ode.ode_coder  import make_events

###############################################################################
#Code Generation for Events
###############################################################################
def trigger2code(event: dst.Event) -> str:
    
    trigger_code = f'\treturn __mean({event.trigger})\n'
    trigger_code = ut.undot(trigger_code)
    return trigger_code

def assignment2code(event: dst.Event) -> str:
    assignment_code = '\t#Assignment\n'
    
    for lhs, rhs in event.assignments.items():
        lhs = ut.undot(lhs)
        rhs = ut.undot(rhs)
        
        assignment_code += f'\tif type({lhs}) == __ndarray:\n'
        assignment_code += f'\t\t{lhs}.fill({rhs})\n'
    
        assignment_code += '\telse:\n'
        assignment_code += f'\t\tprint(type({lhs}))\n'
        
        assignment_code += f'\t\t{lhs} = {rhs}\n'
    
    return assignment_code
    
def make_new_y_code(spatial_data: SpatialModelData) -> str:
    states = ut.undot(spatial_data.states)
    states = ', '.join(states)
    return f'\tnew_y = __concatenate([{states}])'
    
###############################################################################
#Class
###############################################################################
class EventStack(RateStack):
    #Expected mappings and attributes
    grid                  : RegularGrid|NestedGrid
    ndims                 : int
    shifts                : list
    sizes                 : dict[Voxel, Number]
    voxels                : dict[Voxel, dict]
    shape_dict            : One2One[str, object]
    shape2domain_type     : Many2One[str, Domain_type]
    voxel2domain_type     : Many2One[Voxel, Domain_type]
    voxel2domain_type_idx : One2One[Voxel, tuple[int, Domain_type]]
    voxel2shape           : Many2One[Voxel, str]
    shape2domain          : Many2One[str, Domain]
    
    grids              : dict[str, RegularGrid|NestedGrid]
    adjacent_shapes    : set[tuple[str, str]]
    voxel2domain       : Many2One[Voxel, Domain]
    domain2domain_type : Many2One[Domain, Domain_type]
    shape2domain       : Many2One[str, Domain]
    surface2domain     : One2One[str|tuple[Domain, Domain], tuple[Domain, Domain]]
    
    spatial_data      : SpatialModelData
    element2idx       : One2One[tuple[Voxel, State], int]
    state2dxidx       : One2One[State, tuple[int, int]]
    state2domain_type : Many2One[State, Domain_type]
    state_code        : str
    parameter_code    : str
    function_code     : str
    diff_code         : str
    signature         : tuple[str]
    rhs_functions     : dict[str, callable]
    formatter         : str
    
    surface_data         : dict[Surface_type, dict]
    global_variables     : set
    bulk_variables       : dict[str, Domain_type]
    surface_variables    : dict[str, Surface_type]
    variable_code        : str
    bulk_reactions       : dict[str, Domain_type]
    surface_reactions    : dict[str, Surface_type]
    reaction_code_rhsdct : str
    
    domain_type2volume : dict[Domain_type, dict[int, float]]
    advection_terms    : dict[Domain_type, dict[int, dict]]
    diffusion_terms    : dict[Domain_type, dict[int, dict]]
    boundary_terms     : dict[Domain_type, dict[int, dict]]
    advection_code     : str
    diffusion_code     : str
    boundary_code      : str
    
    rate_code     : str
    d_states_code : str
    rhs_name      : str
    rhs_code      : str
    _rhs_funcs    : tuple[Callable, Callable]
    _rhs          : Callable
    rhsdct_name   : str
    rhsdct_code   : str
    _rhsdct_funcs : tuple[Callable, Callable]
    _rhsdct       : Callable
    
    def __init__(self, spatial_data: SpatialModelData):
        super().__init__(spatial_data)
        self.event_functions = {'__mean' : np.mean}
        
        self.average = np.mean
        self._events = []
        self._add_events()
    
    ###########################################################################
    #Events
    ###########################################################################
    def _add_events(self) -> None:
        spatial_data = self.spatial_data
        body_code    = '\n'.join([self.state_code,
                                  self.parameter_code,
                                  self.diff_code, 
                                  self.function_code,
                                  self.variable_code,
                                  self.d_states_code
                                  ])
        
        new_y_code    = make_new_y_code(spatial_data)
        event_objects = make_events(spatial_data, 
                                    trigger2code, 
                                    assignment2code, 
                                    body_code,
                                    new_y_code
                                    )
        
        self._events.extend(event_objects)
    
    
    
      