import numpy             as np
import re
import warnings
from collections        import namedtuple
from matplotlib.patches import Rectangle
from numbers import Number
from scipy   import spatial
from typing  import Union

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

#Typing

#Containers
class MassTransfer:
    def __init__(self):
        self.left0  = '0'
        self.left1  = '0'
        self.right0 = '0'
        self.right1 = '0'
        
    @property
    def lst(self):
        return [self.left0, self.left1, self.right0, self.right1]
    
    def __iter__(self):
        return iter(self.lst)
    
    def __repr__(self):
        return str(self.lst)
    
    def __str__(self):
        return repr(self.lst)
    
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
        self._reformat_mass_transfer_terms(self.advection_terms)
        self.advection_code = ''
        self._add_advection_code()
        
        #Parse the diffusion
        self._reformat_mass_transfer_terms(self.diffusion_terms)
        self.diffusion_code = ''
        self._add_diffusion_code()
        
        # # #Parse the boundary conditions
        # # self.boundary_condition_code = ''
        
    ###########################################################################
    #Preprocessing
    ###########################################################################
    def _add_bulk(self, voxel0, voxel1, shift) -> None:
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes 
        n                     = self.ndims-1
        advection_terms       = self.advection_terms
        diffusion_terms       = self.diffusion_terms
        domain_type           = self.voxel2domain_type[voxel0]
        domain_type_idx       = voxel2domain_type_idx[voxel0][0]
        axis                  = abs(shift)
        
        advection_terms.setdefault(domain_type, {}).setdefault(axis, {}).setdefault(domain_type_idx, MassTransfer())
        diffusion_terms.setdefault(domain_type, {}).setdefault(axis, {}).setdefault(domain_type_idx, MassTransfer())
        
        advection_term = advection_terms[domain_type][axis][domain_type_idx]
        diffusion_term = diffusion_terms[domain_type][axis][domain_type_idx]
        
        advect  = lambda sign, c1, D, A    : f'{sign}{{state}}[{c1}]*{A}/{D}' 
        diffuse = lambda sign, c0, c1, D, A: f'({{state}}[{c1}]-{{state}}[{c0}])*{A}/{D}**2'
        
        if shift < 0:
            sign  = '+'
            c0    = voxel2domain_type_idx[voxel0][0]
            c1    = voxel2domain_type_idx[voxel1][0]
            size0 = sizes[voxel0]
            size1 = sizes[voxel1]
            D     = (size0 + size1)/2
            A     = min(size0, size1)**n
            
            term  = advect(sign, c1, D, A)
            
            if advection_term.left0 == '0':
                advection_term.left0 = term
            else:
                advection_term.left1 = term
            
            term = diffuse(sign, c0, c1, D, A)
            
            if diffusion_term.left0 == '0':
                diffusion_term.left0 = term
            else:
                diffusion_term.left1 = term
                
        else:
            sign  = '-'
            c0    = voxel2domain_type_idx[voxel0][0]
            c1    = voxel2domain_type_idx[voxel1][0]
            size0 = sizes[voxel0]
            size1 = sizes[voxel1]
            D     = (size0 + size1)/2
            A     = min(size0, size1)**2
            
            term  = advect(sign, c1, D, A)
            
            if advection_term.right0 == '0':
                advection_term.right0 = term
            else:
                advection_term.right1 = term
            
            term = diffuse(sign, c0, c1, D, A)
            
            if diffusion_term.right0 == '0':
                diffusion_term.right0 = term
            else:
                diffusion_term.right1 = term
        
    def _add_boundary(self, voxel, shift) -> None:
        pass
    
    def _reformat_mass_transfer_terms(self, terms: dict) -> None:
        
        for domain_type, domain_type_data in terms.items():
            for axis, axis_data in list(domain_type_data.items()):
                transposed    = zip(*axis_data.values())
                driving_force = ['__array([' + ', '.join(x)  + '])' for x in transposed]
                state_idxs    = list(axis_data.keys())
                
                domain_type_data[axis] = {'state_idxs'    : state_idxs,
                                          'driving_force' : driving_force
                                          }
                
    ###########################################################################
    #Mass Transfer
    ###########################################################################
    def _add_advection_code(self) -> None:
        advection_terms = self.advection_terms
        advection       = self.spatial_data.advection
        
        self.advection_code = self._add_mass_transfer_code('Advection',
                                                           self.advect,
                                                           advection_terms,
                                                           advection
                                                           )
    
    def _add_diffusion_code(self) -> None:
        diffusion_terms = self.diffusion_terms
        diffusion       = self.spatial_data.diffusion
        
        self.diffusion_code = self._add_mass_transfer_code('Diffusion',
                                                           self.diffuse,
                                                           diffusion_terms,
                                                           diffusion
                                                           )
        
    def _add_mass_transfer_code(self, 
                                name          : str, 
                                formatter     : callable,
                                terms         : dict, 
                                datastructure : Union[AdvectionDict, DiffusionDict]
                                ) -> None:
        
        state2domain_type     = self.state2domain_type
        domain_type2voxel     = self.voxel2domain_type.inverse
        domain_type_idx2voxel = self.voxel2domain_type_idx.inverse
        sizes                 = self.sizes
        code        = ''
        
        for state in datastructure:
            domain_type     = state2domain_type[state]
            n_voxels        = len(domain_type2voxel[domain_type])
            code += f'\t#{name} {state}\n'
            code += f'\t{formatter(state)} = __zeros({n_voxels})\n\n'
            
            get_size = lambda domain_type_idx: sizes[domain_type_idx2voxel[domain_type_idx, domain_type]]
            
            for axis, term in terms[domain_type].items():
                rhs_idxs = term['driving_force']
                lhs      = f'{formatter(state)}_{axis}'
                rhs      = [x.format(state=state) for x in rhs_idxs]
                rhs      = ' +'.join(rhs)
                
                code += f'\t{lhs}  = {rhs}\n'
                
                coeff      = datastructure.find(state, axis)
                rhs_sizes  = [get_size(x) for x in term['state_idxs']]
                rhs_sizes  = f'__array({rhs_sizes})'
                code      += f'\t{lhs} *= {coeff}/{rhs_sizes}\n'
                
                lhs_idxs = term['state_idxs']
                lhs      = f'{formatter(state)}[__array({lhs_idxs})]'
                rhs      = f'{formatter(state)}_{axis}'
                
                code += f'\t{lhs} += {rhs}\n\n'
            
            code += f'\t{ut.diff(state)} += {formatter(state)}\n\n'
                
        return code + '\n'
    
    ###########################################################################
    #Boundary Conditions
    ###########################################################################
    
    
    ###########################################################################
    #Namespace
    ###########################################################################
    def advect(self, state: str) -> str:
        return f'_adv_{state}'
    
    def diffuse(self, state: str) -> str:
        return f'_dfn_{state}'

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
    