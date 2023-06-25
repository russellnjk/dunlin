import numpy as np
import re
from numba   import njit  
from numbers import Number
from scipy   import spatial
from typing  import Union

import dunlin.utils        as ut
from .grid.grid            import RegularGrid, NestedGrid
from .grid.bidict          import One2One, One2Many
from .ratestack            import (RateStack,
                                   Domain_type, Domain, Voxel, 
                                   AdjacentShapes, AdjacentDomains,
                                   State, Parameter,
                                   Surface
                                   )
from dunlin.datastructures import SpatialModelData
from dunlin.ode.ode_coder  import make_rhsevents
from dunlin.ode.event      import Event

class EventStack(RateStack):
    #Expected mappings and attributes
    grid                  : Union[RegularGrid, NestedGrid]
    ndims                 : int
    shifts                : list
    sizes                 : dict[Voxel, Number]
    voxels                : dict[Voxel, dict]
    shape_dict            : One2One[str, object]
    shape2domain_type     : One2Many[str, Domain_type]
    voxel2domain_type     : One2Many[Voxel, Domain_type]
    voxel2domain_type_idx : One2Many[Voxel, int]
    voxel2shape           : One2Many[Voxel, str]
    shape2domain          : One2Many[str, Domain]
    
    grids              : dict[str, Union[RegularGrid, NestedGrid]]
    adjacent_shapes    : dict[tuple, AdjacentShapes]
    adjacent_domains   : dict[tuple, AdjacentDomains]
    voxel2domain       : One2Many[Voxel, Domain]
    domain2domain_type : One2Many[Domain, Domain_type]
    
    spatial_data      : SpatialModelData
    element2idx       : One2One[tuple[Voxel, State], int]
    state2dxidx       : One2One[State, tuple[int, int]]
    state2domain_type : One2Many[State, Domain_type]
    state_code        : str
    parameter_code    : str
    function_code     : str
    diff_code         : str
    signature         : tuple[str]
    rhs_functions     : dict[str, callable]
    rhsdct_functions  : dict[str, callable]
    formatter         : str
    
    surface2domain_type_idx  : dict[Surface, One2One[int, int]]
    surfacepoint2surface_idx : One2One[tuple[Number], tuple[int, Surface]]
    surfacepoint_lst         : list[tuple[Number]]
    surface2tree             : spatial.KDTree
    global_variables         : set
    bulk_variables           : dict[str, Domain_type]
    surface_variables        : dict[str, Surface]
    variable_code            : str
    bulk_reactions           : dict[str, Domain_type]
    surface_reactions        : dict[str, Surface]
    reaction_code            : str
    surface_linewidth        : float
    
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
    _rhs_funcs    : tuple[callable, callable]
    _rhs          : callable
    rhsdct_name   : str
    rhsdct_code   : str
    _rhsdct_funcs : tuple[callable, callable]
    _rhsdct       : callable
    
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
        
        #Shared templates
        ref       = spatial_data.ref
        signature = ', '.join(self.signature)
        body_code = '\n'.join([self.state_code,
                               self.parameter_code,
                               self.diff_code, 
                               self.function_code,
                               self.variable_code,
                               self.d_states_code
                               ])
        
        #Templates for trigger
        trigger_def       = f'def {{trigger_name}}({signature}):'
        trigger_return    = '\treturn __triggered'
        trigger_expr      = '\t__triggered = {trigger_expr}'
        
        #Templates for assignment
        assign_def        = f'def {{assign_name}}({signature}):'
        assign_return     = '\treturn new_y, new_p'
        assign_expr       = '\t__triggered = __ones{assign_expr}'
        xs                = [ut.undot(x) for x in spatial_data.states    ]
        ps                = [ut.undot(p) for p in spatial_data.parameters]
        new_y             = f'\tnew_y = __concatenate(({", ".join(xs)}))'
        new_p             = f'\tnew_p = __array([{", ".join(ps)}])'
        assign_collate    = '\n'.join(['\t#Collate', new_y, new_p])
        
        
        #Iterate through events
        for event in spatial_data.events.values(): 
            #Create trigger function
            trigger_name = f'trigger_{ref}_{event.name}'
            
            code = [trigger_def.format(trigger_name=trigger_name), 
                    body_code, 
                    '\t#Trigger\n', 
                    trigger_expr.format(trigger_expr=event.trigger_expr), 
                    trigger_return
                    ]
            code = '\n\n'.join(code)
            
            #Execute code
            scope = {}
            exec(code, self.rhs_functions, scope)
            trigger_func      = scope[trigger_name]
            trigger_func      = self._wrap_trigger(trigger_func)
            trigger_func.code = code
            
            #Create assign function=
            assign_name = f'assign_{ref}_{event.name}'
            
            exprs = []
            for assign_expr in event.assign:
                pattern = '(^\W*)(\w*)(\W*=)'
                
                def repl(match):
                    lhs = match[2]
                    if lhs in xs:
                        add = f'__ones({lhs}.shape)*'
                        result = match[0] + add
                    else:
                        result = match[0]
                    
                    return f'\t{result}'
                
                new_expr = re.sub(pattern, repl, assign_expr)
                exprs.append(new_expr)
            
            
            code = [assign_def.format(assign_name=assign_name), 
                    body_code, 
                    '\t#Assign\n', 
                    '\n'.join(exprs), 
                    assign_collate,
                    assign_return
                    ]
            code = '\n\n'.join(code)
            
            #Execute code
            scope = {}
            exec(code, self.rhs_functions, scope)
            assign_func      = scope[assign_name]
            assign_func.code = code 
            
            #Make event
            event_object = Event(event.name, 
                                 trigger_func, 
                                 assign_func, 
                                 delay      = event.delay, 
                                 persistent = event.persistent, 
                                 priority   = event.priority, 
                                 ref        = ref
                                 )
            self._events.append(event_object)
            
    def _wrap_trigger(self, trigger_func: callable) -> callable:
        def helper(time, states, parameters):
            raw     = trigger_func(time, states, parameters)
            average = self.average(raw)
            return average
        return helper
    
    
      