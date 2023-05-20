import numpy     as np
import numpy.ma  as ma
import re
import textwrap  as tw
import warnings
from collections        import namedtuple
from matplotlib.patches import Rectangle
from numba              import njit  
from numbers            import Number
from scipy              import spatial
from typing             import Union

import dunlin.utils      as ut
import dunlin.utils_plot as upp
from .grid.grid            import RegularGrid, NestedGrid
from .grid.bidict          import One2One, One2Many
from .reactionstack        import (ReactionStack,
                                   Domain_type, Domain, Voxel, 
                                   AdjacentShapes, AdjacentDomains,
                                   State, Parameter,
                                   Surface
                                   )
from dunlin.datastructures import SpatialModelData, AdvectionDict, DiffusionDict

#Mass Transfer Functions
@njit
def calculate_neumann_boundary0(X               : np.array, 
                               flux             : Number, 
                               domain_type_idxs : np.array,
                               scale            : np.array
                               ) -> np.array:
    dX = np.zeros(len(X))
    
    #Calculate the change in concentration
    change = flux * scale
    
    dX[domain_type_idxs] += change
    
    return dX

@njit
def calculate_neumann_boundary1(X                : np.array, 
                                flux             : np.array, 
                                domain_type_idxs : np.array,
                                scale            : np.array
                                ) -> np.array:
    dX = np.zeros(len(X))
    
    #Apply the indices to extract the quantities
    flux_boundary = flux[domain_type_idxs]
    
    #Calculate the change in concentration
    change = flux_boundary * scale
    
    dX[domain_type_idxs] += change
    
    return dX
    
@njit
def calculate_advection0(X              : np.array, 
                         coeff          : Number, 
                         left_mappings  : tuple[np.array], 
                         right_mappings : tuple[np.array],
                         ) -> np.array:
    
    dX        = np.zeros(len(X))
    abs_coeff = np.abs(coeff)
    
    #Case 1: coeff < 0
    for left_mapping in left_mappings:
        #Determine which elements are negative
        #Find the indices of the sources and destinations
        left_src_idxs = np.where((coeff < 0) & left_mapping[2])
        left_src_idxs = left_src_idxs[0].astype(np.int64)
        left_dst_idxs = left_mapping[0][left_src_idxs]
        scale_left    = left_mapping[1][left_src_idxs]
        
        #Apply the indices to extract the quantities for leftward calculation
        X_src      = X[left_src_idxs]
        
        #Calculate change in concentration
        change = abs_coeff * X_src * scale_left
    
        dX[left_src_idxs] -= change
        dX[left_dst_idxs] += change
    
    #Case 2: coeff > 0
    for right_mapping in right_mappings:
        #Determine which elements are negative
        #Find the indices of the sources and destinations
        right_src_idxs = np.where((coeff > 0) & right_mapping[2])
        right_src_idxs = right_src_idxs[0].astype(np.int64)
        right_dst_idxs = right_mapping[0][right_src_idxs]
        scale_right    = right_mapping[1][right_src_idxs]
        
        #Apply the indices to extract the quantities for rightward calculation
        X_src       = X[right_src_idxs]
        
        #Calculate change in concentration
        change = abs_coeff * X_src * scale_right
    
        dX[right_src_idxs] -= change
        dX[right_dst_idxs] += change
    
    return dX

@njit
def calculate_advection1(X              : np.array, 
                         coeff          : np.array, 
                         left_mappings  : tuple[np.array], 
                         right_mappings : tuple[np.array],
                         ) -> np.array:
    
    dX        = np.zeros(len(X))
    abs_coeff = np.abs(coeff)
    
    #Case 1: coeff < 0
    for left_mapping in left_mappings:
        #Determine which elements are negative
        #Find the indices of the sources and destinations
        left_src_idxs = np.where((coeff < 0) & left_mapping[2])
        left_src_idxs = left_src_idxs[0].astype(np.int64)
        left_dst_idxs = left_mapping[0][left_src_idxs]
        scale_left    = left_mapping[1][left_src_idxs]
        
        #Apply the indices to extract the quantities for leftward calculation
        coeff_left = abs_coeff[left_src_idxs] 
        X_src      = X[left_src_idxs]
        
        #Calculate change in concentration
        change = coeff_left * X_src * scale_left
    
        dX[left_src_idxs] -= change
        dX[left_dst_idxs] += change
    
    #Case 2: coeff > 0
    for right_mapping in right_mappings:
        #Determine which elements are negative
        #Find the indices of the sources and destinations
        right_src_idxs = np.where((coeff > 0) & right_mapping[2])
        right_src_idxs = right_src_idxs[0].astype(np.int64)
        right_dst_idxs = right_mapping[0][right_src_idxs]
        scale_right    = right_mapping[1][right_src_idxs]
        
        #Apply the indices to extract the quantities for rightward calculation
        coeff_right = abs_coeff[right_src_idxs]
        X_src       = X[right_src_idxs]
        
        #Calculate change in concentration
        change = coeff_right * X_src * scale_right
    
        dX[right_src_idxs] -= change
        dX[right_dst_idxs] += change
        
    return dX

@njit
def calculate_diffusion(X              : np.array, 
                        coeff          : Union[Number, np.array], 
                        left_mappings  : tuple[np.array], 
                        ) -> np.array:
    #Overhead
    dX        = np.zeros(len(X))
    
    for left_mapping in left_mappings:
        #Find the center and the left voxels
        left_src_idxs = np.where(left_mapping[2])[0].astype(np.int64)
        left_dst_idxs = left_mapping[0][left_src_idxs]
        scale_left    = left_mapping[1][left_src_idxs]
        
        #Find concentration gradient
        gradient = X[left_src_idxs] - X[left_dst_idxs]
        
        #Calculate change in concentration
        change = coeff * gradient * scale_left
        
        dX[left_src_idxs] -= change
        dX[left_dst_idxs] += change
    
    return dX

#ReactionStack
class MassTransferStack(ReactionStack):
    #For plotting
    default_mass_transfer_args      = {'edgecolor': 'None'
                                       }
    default_boundary_condition_args = {'linewidth' : 5
                                       }
    
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
    
    def __init__(self, spatial_data: SpatialModelData):
        #Data structures for self._add_bulk
        self.advection_terms = {}
        self.diffusion_terms = {}
        
        #Data structures for self._add_boundary
        
        #Call the parent constructor
        super().__init__(spatial_data)
        
        #Data structures for retrieval by location
        
        #Parse the advection
        self.advection_calculators = {}
        self._reformat_mass_transfer_terms(self.advection_terms)
        self.advection_code = ''
        self._add_advection_code()
        
        #Parse the diffusion
        self._reformat_mass_transfer_terms(self.diffusion_terms)
        self.diffusion_code = ''
        self._add_diffusion_code()
        
        # # #Parse the boundary conditions
        # # self.boundary_condition_code = ''
        
        self.signature += tuple(self.advection_calculators)
        
    ###########################################################################
    #Preprocessing
    ###########################################################################
    def _add_bulk(self, voxel0, voxel1, shift) -> None:
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes 
        nd                    = self.ndims-1
        advection_terms       = self.advection_terms
        diffusion_terms       = self.diffusion_terms
        domain_type           = self.voxel2domain_type[voxel0]
        axis                  = abs(shift)
        
        #Template functions for adding new terms
        voxels  = self.voxel2domain_type.inverse[domain_type]
        n       = len(voxels)
        axes    = list(range(1, self.ndims+1))
        helper0 = lambda: np.array([[-1]*n, [0]*n, [0]*n]) 
        helper1 = lambda *shifts: {shift : [helper0()] for shift in shifts}
        helper2 = lambda *shifts: {x: helper1(*shifts) for x in axes}
        
        #Set defaults in the nested dictionary
        if domain_type not in self.advection_terms:       
            self.advection_terms[domain_type] = helper2('left', 'right')
            self.diffusion_terms[domain_type] = helper2('left')
            
        #Get the advection and diffusion terms
        if shift < 0:
            advection_term = advection_terms[domain_type][axis]['left']
            diffusion_term = diffusion_terms[domain_type][axis]['left']
        else:
            advection_term = advection_terms[domain_type][axis]['right']
            diffusion_term = None
            
        #Calculate the values to store
        c0    = voxel2domain_type_idx[voxel0][0]
        c1    = voxel2domain_type_idx[voxel1][0]
        size0 = sizes[voxel0]
        size1 = sizes[voxel1]
        D     = (size0 + size1)/2
        A     = min(size0, size1)**nd
        
        #Find a vacant element and store the information
        m = 0
        while True:
            #0 : dst, 1: scale, 2: mask
            if advection_term[m][0][c0] == -1:
                advection_term[m][0][c0]   = c1
                advection_term[m][1][c0] = A/D
                advection_term[m][2][c0]  = 1
                
                if diffusion_term is not None:
                    diffusion_term[m][0][c0]   = c1
                    diffusion_term[m][1][c0] = A/D**2
                    diffusion_term[m][2][c0]  = 1
                
                break
            
            else:
                advection_term.append(helper0())
                
                if diffusion_term is not None:
                    diffusion_term.append(helper0())
                    
                m += 1
           
    def _add_boundary(self, voxel, shift) -> None:
        axis = abs(shift)
        
        pass
    
    def _reformat_mass_transfer_terms(self, terms: dict) -> None:
        helper = lambda x: repr(tuple(x)).replace('array', '__array').replace('\n', ' ').replace('\t', '') 
        
        for domain_type, domain_type_data in terms.items():
            for axis, axis_data in domain_type_data.items():
                for shift in axis_data:
                    axis_data[shift]  = helper(axis_data[shift])
                
    ###########################################################################
    #Mass Transfer
    ###########################################################################
    def _add_advection_code(self) -> None:
        advection_terms = self.advection_terms
        advection       = self.spatial_data.advection
        datastructure   = advection
        
        state2domain_type     = self.state2domain_type
        domain_type2voxel     = self.voxel2domain_type.inverse
        code                  = ''
        name                  = '#Advection'
            
        for state in datastructure:
            domain_type     = state2domain_type[state]
            n_voxels        = len(domain_type2voxel[domain_type])
            code += f'\t#{name} {state}\n'
            code += f'\t{ut.adv(state)} = __zeros({n_voxels})\n\n'
            
            for axis, axis_data in advection_terms[domain_type].items():
                coeff = datastructure.find(state, axis)
                left_mappings  = axis_data['left']
                right_mappings = axis_data['right']
                
                code += f'\t_left = {left_mappings}\n'
                code += f'\t_right = {right_mappings}\n'
                
                lhs = f'{ut.adv(state)}'
                rhs = f'_advfunc_{state}_{axis}({state}, {coeff}, _left, _right)\n'
                
                code += f'\t{lhs} += {rhs}\n'
                
                self._set_advection_calculator(state, coeff, axis)
                
                
            code += f'\t{ut.diff(state)} += {ut.adv(state)}\n\n'
                
        self.advection_code = code + '\n'
    
    
    def _set_advection_calculator(self,
                                  state : State, 
                                  coeff : Union[str, Number],
                                  axis  : int
                                  ) -> None:
        
        key = f'_advfunc_{state}_{axis}'
        
        if isinstance(ut.try2num(coeff), Number):
            self.advection_calculators[key] = calculate_advection0
            return
        
        namespace = list(ut.get_namespace(coeff))
        name      = namespace[0]

        if name in self.global_variables or name in self.spatial_data.parameters:
            self.advection_calculators[key] = calculate_advection0
            
        elif name in self.bulk_variables:
            #Check the domain types match
            domain_type0 = self.bulk_variables[name]
            domain_type1 = self.state2domain_type[state]
            if domain_type0 == domain_type1:
                self.advection_calculators[key] = calculate_advection1
            else:
                msg  = f'Advection coefficient {coeff} is a variable defined in domain type {domain_type0}. '
                msg += f'However, it is being for a state that exists in domain type {domain_type1}.'
                raise ValueError(msg)
        else:
            msg  = 'The advection coefficient must be a parameter, global variable or bulk variable.'
            msg += f'Coefficient "{coeff}" for state "{state}" is neither.'
            raise ValueError(msg)
    
    def _add_diffusion_code(self) -> None:
        diffusion_terms = self.diffusion_terms
        diffusion       = self.spatial_data.diffusion
        datastructure   = diffusion
        
        state2domain_type     = self.state2domain_type
        domain_type2voxel     = self.voxel2domain_type.inverse
        code                  = ''
        name                  = '#diffusion'
            
        for state in datastructure:
            domain_type     = state2domain_type[state]
            n_voxels        = len(domain_type2voxel[domain_type])
            code += f'\t#{name} {state}\n'
            code += f'\t{ut.dfn(state)} = __zeros({n_voxels})\n\n'
            
            for axis, axis_data in diffusion_terms[domain_type].items():
                coeff = datastructure.find(state, axis)
                left_mappings  = axis_data['left']
                
                code += f'\t_left = {left_mappings}\n'
                
                lhs = f'{ut.dfn(state)}'
                rhs = f'__dfnfunc({state}, {coeff}, _left)\n'
                
                code += f'\t{lhs} += {rhs}\n'
                
                
            code += f'\t{ut.diff(state)} += {ut.dfn(state)}\n\n'
                
        self.diffusion_code = code + '\n'
        
    ###########################################################################
    #Boundary Conditions
    ###########################################################################

    ###########################################################################
    #Retrieval
    ###########################################################################
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def plot_advection(self, 
                       ax, 
                       state_name       : str,
                       advection_values : np.array,
                       advection_args   : dict = None
                       ) -> None:
        
        domain_type    = self.state2domain_type[state_name]
        default_kwargs = self.default_mass_transfer_args
        
        return self._plot_bulk(ax, 
                               state_name, 
                               advection_values, 
                               domain_type, 
                               default_kwargs,
                               advection_args
                               )
    
    def plot_diffusion(self, 
                       ax, 
                       state_name       : str,
                       diffusion_values : np.array,
                       diffusion_args   : dict = None
                       ) -> None:
        
        domain_type    = self.state2domain_type[state_name]
        default_kwargs = self.default_mass_transfer_args
        
        return self._plot_bulk(ax, 
                               state_name, 
                               diffusion_values, 
                               domain_type, 
                               default_kwargs,
                               diffusion_args
                               )
    