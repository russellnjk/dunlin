import numpy as np
from numba   import njit  
from numbers import Number
from scipy   import spatial
from typing  import Callable

import dunlin.utils        as ut
from ..grid.grid           import RegularGrid, NestedGrid
from .bidict               import One2One, Many2One
from .masstransferstack    import (MassTransferStack,
                                   Domain_type, Domain, Voxel, 
                                   State, Parameter,
                                   Surface_type
                                   )
from dunlin.datastructures import SpatialModelData

class RateStack(MassTransferStack):
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
        
        #Add required functions
        self.rhs_functions['__concatenate'] = np.concatenate
        self.rhs_functions['__njit'       ] = njit
        self.rhs_functions['__newaxis'    ] = np.newaxis
        
        self.rhsdct_functions['__concatenate'] = np.concatenate
        self.rhsdct_functions['__njit'       ] = njit
        self.rhsdct_functions['__newaxis'    ] = np.newaxis
        
        #Parse rates
        self.rate_code = ''
        self._add_rate_code()
        
        
        #Make the d_states code
        self.d_states_code = ''
        self._make_d_states()
        
        #Combine all code and execute
        self.rhs_name   = ''
        self.rhs_code   = ''
        self._rhs_funcs = tuple()
        self._rhs       = None
        
        self.rhsdct_name   = ''
        self.rhsdct_code   = ''
        self._rhsdct_funcs = tuple()
        self._rhsdct       = None
        
        self._make_rhs()
        
    ###########################################################################
    #Controlled Access to RHS
    ###########################################################################
    @property
    def rhs(self):
        return self._rhs
    
    @property
    def rhsdct(self):
        return self._rhsdct
    
    @property
    def numba(self) -> bool:
        return self.rhs is self._rhs_funcs[1]
    
    @numba.setter
    def numba(self, value: bool) -> None:
        if value:
            self._rhs    = self._rhs_funcs[1]
            self._rhsdct = self._rhsdct_funcs[1]
        else:
            self._rhs    = self._rhs_funcs[0]
            self._rhsdct = self._rhsdct_funcs[0]
    
    ###########################################################################
    #Code Generation
    ###########################################################################
    def _add_rate_code(self) -> None:
        rates = self.spatial_data.rates
        
        code = ''
        for state, rate in rates.items():
            state_ = ut.undot(state)
            lhs    = f'{ut.diff(state_)}'
            rhs    = ut.undot(rate.expr)
            
            code += f'\t{lhs} += {rhs}\n'
        
        self.rate_code += code
    
    def _make_d_states(self):
        code = f'\t{ut.diff("states")} = __concatenate(('
        
        for state in self.spatial_data.states:
            state_  = ut.undot(state)
            code   += ut.diff(state_) + ', '
        
        code += '))'
        
        self.d_states_code = code
    
    ###########################################################################
    #Code Execution
    ###########################################################################
    def _make_rhs(self) -> None:
        spatial_data = self.spatial_data
        
        #Extract and preprocess
        model_ref = spatial_data.ref
        signature = ', '.join(self.signature)
    
        #Make rhs function
        body_code = '\n'.join([self.state_code,
                               self.parameter_code,
                               self.diff_code, 
                               self.function_code,
                               self.variable_code,
                               self.reaction_code,
                               self.advection_code,
                               self.diffusion_code,
                               self.boundary_condition_code,
                               self.rate_code, 
                               self.d_states_code
                               ])
        
        self.rhs_name = f'model_{model_ref}'
        function_def  = f'def {self.rhs_name}({signature}):\n'
        return_val    = f'\treturn {ut.diff("states")}'
        
        code = '\n'.join([function_def, body_code, return_val])
        
        self.rhs_code = code
        
        scope = {}
        exec(code, self.rhs_functions, scope)
        self._rhs_funcs = scope[self.rhs_name], njit(scope[self.rhs_name])
        self._rhs       = self._rhs_funcs[1]
        
        #Make rhsdct function
        body_code = '\n'.join([self.state_code,
                               self.parameter_code,
                               self.diff_code, 
                               self.function_code,
                               self.variable_code,
                               self.reaction_code_rhsdct,
                               self.advection_code,
                               self.diffusion_code,
                               self.boundary_condition_code,
                               self.rate_code, 
                               ])
        
        self.rhsdct_name = f'model_{model_ref}_dct'
        function_def     = f'def {self.rhsdct_name}({signature}):\n'
        
        lst  = ['time']
        lst  += list(spatial_data.states)
        lst  += list(spatial_data.parameters)
        lst  += list(spatial_data.variables)
        lst  += list(spatial_data.reactions)
        lst  += [ut.adv(x)  for x in spatial_data.advection]
        lst  += [ut.dfn(x)  for x in spatial_data.diffusion]
        lst  += [ut.bc(x)   for x in spatial_data.boundary_conditions]
        lst  += [ut.diff(x) for x in spatial_data.states]
        lst_  = ut.undot(lst)
        
        temp       = ', '.join(lst_)
        return_val = f'\treturn ({temp})'
        
        code = '\n'.join([function_def, body_code, return_val])
        
        self.rhsdct_code = code
        
        #Execute code
        exec(code, self.rhsdct_functions, scope)
        tuple_func0 = scope[self.rhsdct_name]
        tuple_func1 = njit(tuple_func0)
        
        dct_func0   = self._wrap(lst, tuple_func0)
        dct_func1   = self._wrap(lst, tuple_func1)
        
        self._rhsdct_funcs = dct_func0, dct_func1
        self._rhsdct       = self._rhsdct_funcs[1]
    
    @staticmethod
    def _wrap(keys: list[str], func: Callable) -> Callable:
        def helper(t, y, p):
            tup_values = func(t, y, p)
            dct        = dict(zip(keys, tup_values))
            
            return dct
        
        helper.func = func
        return helper
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def plot_rate(self, 
                  ax, 
                  state_name  : str,
                  rate_values : np.ndarray,
                  **kwargs
                  ) -> dict:
        
        domain_type    = self.state2domain_type[state_name]
        
        return self._plot_bulk(ax, 
                               state_name, 
                               rate_values, 
                               domain_type, 
                               **kwargs
                               )
    
        
        
        