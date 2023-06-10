import numpy as np
from numba   import njit  
from numbers import Number
from scipy   import spatial
from typing  import Union

import dunlin.utils      as ut
from .grid.grid            import RegularGrid, NestedGrid
from .grid.bidict          import One2One, One2Many
from .reactionstack        import (ReactionStack,
                                   Domain_type, Domain, Voxel, 
                                   AdjacentShapes, AdjacentDomains,
                                   State, Parameter,
                                   Surface
                                   )
from dunlin.datastructures import SpatialModelData

'''
Notes:
    I wasn't sure if/how diffusion, advection and boundary conditions were fixed 
    numbers of allowed to vary across space and time. Currently, the functions 
    for calculating mass transfer (and boundary conditions) allow for the 
    conditions and mass transfer coefficients to be arrays. However, for simplicity,
    the MassTransferStack only allows them to be scalars either explicitly 
    specified or given by a parameter.
    
    As of now, Dunlin's event-handling capabilities only allow changes in the 
    values of states and parameters. There is thus no possibility that an event 
    being triggered would require a change in the formula/method for calculating 
    the mass transfer.
'''

#Typing
Boundary = Domain_type

#Mass Transfer Functions
@njit
def calculate_neumann_boundary0(X       : np.array, 
                                flux    : Number, 
                                mapping : np.array
                                ) -> np.array:
    mass_tranferred = np.zeros(len(X))
    
    domain_type_idxs = mapping[0]
    scale            = mapping[1] 
    
    #Calculate the change in concentration
    change = flux * scale
    
    mass_tranferred[domain_type_idxs] += change
    
    return mass_tranferred

@njit
def calculate_neumann_boundary(X       : np.array, 
                               flux    : np.array, 
                               mapping : np.array
                               ) -> np.array:
    mass_tranferred = np.zeros(len(X))
    
    domain_type_idxs = mapping[0]
    scale            = mapping[1] 
    
    #Apply the indices to extract the quantities
    flux_boundary = flux[domain_type_idxs]
    
    #Calculate the change in concentration
    change = flux_boundary * scale
    
    mass_tranferred[domain_type_idxs] += change
    
    return mass_tranferred

@njit
def calculate_dirichlet_boundary(X             : np.array, 
                                 coeff         : np.array,
                                 concentration : np.array, 
                                 mapping       : np.array, 
                                 ) -> np.array:
    #Overhead
    mass_tranferred        = np.zeros(len(X))
    
    #Process the diffusion at the axis minimum
    #Find the center and the left voxels
    src_idxs = mapping[0]
    scale    = mapping[1]
    
    #Find concentration gradient
    gradient = X[src_idxs] - concentration[src_idxs]
    coeff_   = coeff[src_idxs]
    
    #Calculate change in concentration
    change = coeff_ * gradient * scale
    
    mass_tranferred[src_idxs] -= change
    
    return mass_tranferred

@njit
def calculate_advection(X              : np.array, 
                        coeff          : np.array, 
                        left_mappings  : tuple[np.array], 
                        right_mappings : tuple[np.array],
                        ) -> np.array:
    
    mass_tranferred        = np.zeros(len(X))
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
    
        mass_tranferred[left_src_idxs] -= change
        mass_tranferred[left_dst_idxs] += change
    
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
    
        mass_tranferred[right_src_idxs] -= change
        mass_tranferred[right_dst_idxs] += change
        
    return mass_tranferred

@njit
def calculate_diffusion(X              : np.array, 
                        coeff          : Number, 
                        left_mappings  : tuple[np.array], 
                        ) -> np.array:
    #Overhead
    mass_tranferred        = np.zeros(len(X))
    
    for left_mapping in left_mappings:
        #Find the center and the left voxels
        left_src_idxs = np.where(left_mapping[2])[0].astype(np.int64)
        left_dst_idxs = left_mapping[0][left_src_idxs]
        scale_left    = left_mapping[1][left_src_idxs]
        
        #Find concentration gradient
        gradient   = X[left_src_idxs] - X[left_dst_idxs]
        coeff_left = coeff[left_src_idxs]
        
        #Calculate change in concentration
        change = coeff_left * gradient * scale_left
        
        mass_tranferred[left_src_idxs] -= change
        mass_tranferred[left_dst_idxs] += change
    
    return mass_tranferred

#ReactionStack
class MassTransferStack(ReactionStack):
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
    functions         : dict[str, callable]
    
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
    formatter          : str
    
    def __init__(self, spatial_data: SpatialModelData):
        #Data structures for self._add_voxel
        self.domain_type2volume = {}
        
        #Data structures for self._add_bulk
        self.advection_terms = {}
        self.diffusion_terms = {}
        self.boundary_terms  = {}
        
        #Call the parent constructor
        super().__init__(spatial_data)
        
        #Data structures for retrieval by location
        
        #Update calculators
        self.functions['__advection'] = calculate_advection
        self.functions['__diffusion'] = calculate_diffusion
        self.functions['__Neumann'  ] = calculate_neumann_boundary
        self.functions['__Dirichlet'] = calculate_dirichlet_boundary
        
        #Parse the advection
        self._reformat_mass_transfer_terms(self.advection_terms)
        self.advection_code = ''
        self._add_advection_code()
        
        #Parse the diffusion
        self._reformat_mass_transfer_terms(self.diffusion_terms)
        self.diffusion_code = ''
        self._add_diffusion_code()
        
        #Parse the boundary conditions
        self._reformat_boundary_terms(self.boundary_terms)
        self.boundary_condition_code = ''
        self._add_boundary_code()
        
        
    ###########################################################################
    #Preprocessing
    ###########################################################################
    def _add_voxel(self, voxel) -> None:
        super()._add_voxel(voxel)
        
        nd                    = self.ndims
        domain_type2volume    = self.domain_type2volume
        sizes                 = self.sizes
        voxel2domain_type_idx = self.voxel2domain_type_idx
        domain_typ2voxel      = self.voxel2domain_type.inverse 
        
        size                         = sizes[voxel]
        domain_type_idx, domain_type = voxel2domain_type_idx[voxel]
        
        default = [0]*len(domain_typ2voxel[domain_type])
        domain_type2volume.setdefault(domain_type, default)
        
        domain_type2volume[domain_type][domain_type_idx] = size**nd
        
    def _add_bulk(self, voxel0, voxel1, shift) -> None:
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes 
        nd                    = self.ndims
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
        A     = min(size0, size1)**(nd-1)
        
        #Find a vacant element and store the information
        m = 0
        while True:
            #0 : dst, 1: scale, 2: mask
            if advection_term[m][0][c0] == -1:
                advection_term[m][0][c0] = c1
                advection_term[m][1][c0] = A/D
                advection_term[m][2][c0] = 1
                
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
        voxel2domain_type_idx    = self.voxel2domain_type_idx
        sizes                    = self.sizes 
        nd                       = self.ndims
        boundary_terms           = self.boundary_terms
        domain_type              = self.voxel2domain_type[voxel]
        axis                     = abs(shift)
        
        #Set defaults in the nested dictionary
        default = {'left': [[], []], 'right': [[], []]}
        boundary_terms.setdefault(domain_type, {}).setdefault(axis, default)
        
        #Get the boundary terms
        if shift < 0:
            boundary_term = boundary_terms[domain_type][axis]['left']
        else:
            boundary_term = boundary_terms[domain_type][axis]['right']
        
        #Calculate the values to store
        c    = voxel2domain_type_idx[voxel][0]
        size = sizes[voxel]
        A     = size**(nd-1)
        V     = size**nd
        
        #Find a vacant element and store the information
        boundary_term[0].append(c)
        boundary_term[1].append(A*V)
    
    def _reformat_mass_transfer_terms(self, terms: dict) -> None:
        helper0 = lambda x: repr(tuple([np.array(i) for i in x]))
        helper1 = lambda x: x.replace('array', '__array').replace('\n', ' ').replace('   ', '')
        helper  = lambda x: helper1(helper0(x))
        
        for domain_type, domain_type_data in terms.items():
            for axis, axis_data in domain_type_data.items():
                for shift in axis_data:
                    axis_data[shift]  = helper(axis_data[shift])
    
    def _reformat_boundary_terms(self, terms: dict) -> None:
        helper0 = lambda x: repr(np.array(x))
        helper1 = lambda x: x.replace('array', '__array').replace('\n', ' ').replace('   ', '') 
        helper  = lambda x: helper1(helper0(x))
        
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
        
        state2domain_type  = self.state2domain_type
        domain_type2voxel  = self.voxel2domain_type.inverse
        code               = ''
        name               = '#Advection'
        domain_type2volume = self.domain_type2volume
            
        for state in advection:
            domain_type  = state2domain_type[state]
            n_voxels     = len(domain_type2voxel[domain_type])
            code        += f'\t#{name} {state}\n'
            code        += f'\t{ut.adv(state)} = __zeros({n_voxels})\n\n'
            
            for axis, axis_data in advection_terms[domain_type].items():
                #Extract information
                coeff           = advection.find(state, axis)
                left_mappings   = axis_data['left']
                right_mappings  = axis_data['right']
                volumes         = domain_type2volume[domain_type]
                volumes         = f'__array({volumes})'
                code           += f'\t_left = {left_mappings}\n'
                code           += f'\t_right = {right_mappings}\n'
                
                
                #Make code for calling external function
                lhs   = f'{ut.adv(state)}'
                
                if coeff in self.bulk_variables:
                    #Check the domain types match
                    domain_type0 = self.bulk_variables[name]
                    domain_type1 = self.state2domain_type[state]
                    
                    if domain_type0 != domain_type1:
                        msg  = f'Advection coefficient {coeff} is a variable defined in domain type {domain_type0}. '
                        msg += f'However, it is being for a state that exists in domain type {domain_type1}.'
                        raise ValueError(msg)
                    
                coeff = f'__ones({n_voxels})*{coeff}'
                rhs   = f'__advection({state}, {coeff}, _left, _right)/{volumes}\n'
                code += f'\t{lhs} += {rhs}\n'
                
            code += f'\t{ut.diff(state)} += {ut.adv(state)}\n\n'
                
        self.advection_code = code + '\n'
            
    def _add_diffusion_code(self) -> None:
        diffusion_terms = self.diffusion_terms
        diffusion       = self.spatial_data.diffusion
        
        state2domain_type     = self.state2domain_type
        domain_type2voxel     = self.voxel2domain_type.inverse
        code                  = ''
        name                  = '#Diffusion'
        domain_type2volume    = self.domain_type2volume
        
        for state in diffusion:
            #Extract information
            domain_type = state2domain_type[state]
            n_voxels    = len(domain_type2voxel[domain_type])
            volumes     = domain_type2volume[domain_type]
            volumes     = f'__array({volumes})'
            
            #Update code
            code += f'\t#{name} {state}\n'
            code += f'\t{ut.dfn(state)} = __zeros({n_voxels})\n\n'
            
            for axis, axis_data in diffusion_terms[domain_type].items():
                #Extract information
                coeff          = diffusion.find(state, axis)
                left_mappings  = axis_data['left']
                code          += f'\t_left = {left_mappings}\n'
                
                #Make code for calling external function
                lhs   = f'{ut.dfn(state)}'
                
                if coeff in self.bulk_variables:
                    msg = 'Diffusion coefficient must be a number, parameter or global variable.'
                    raise NotImplementedError(msg)
                    
                    #Check the domain types match
                    domain_type0 = self.bulk_variables[name]
                    domain_type1 = self.state2domain_type[state]
                    
                    if domain_type0 != domain_type1:
                        msg  = f'Diffusion coefficient {coeff} is a variable defined in domain type {domain_type0}. '
                        msg += f'However, it is being for a state that exists in domain type {domain_type1}.'
                        raise ValueError(msg)
                
                coeff = f'__ones({n_voxels})*{coeff}'
                rhs   = f'__diffusion({state}, {coeff}, _left)/{volumes}\n'
                code += f'\t{lhs} += {rhs}\n'
                
            code += f'\t{ut.diff(state)} += {ut.dfn(state)}\n\n'
                
        self.diffusion_code = code + '\n'
        
    ###########################################################################
    #Boundary Conditions
    ###########################################################################
    def _add_boundary_code(self) -> None:
        boundary_terms      = self.boundary_terms
        boundary_conditions = self.spatial_data.boundary_conditions
        
        state2domain_type     = self.state2domain_type
        domain_type2voxel     = self.voxel2domain_type.inverse
        code                  = ''
        name                  = '#Boundary Conditions'
        
        def helper(state, axis, n_voxels, condition, mapping):
            if condition.condition_type == 'Neumann':
                flux = condition.condition
                
                if flux in self.bulk_variables:
                    msg = 'Neumann boundary flux must be a number, parameter or global variable.'
                    raise NotImplementedError(msg)
                    
                    #Check the domain types match
                    domain_type0 = self.bulk_variables[name]
                    domain_type1 = self.state2domain_type[state]
                    
                    if domain_type0 != domain_type1:
                        msg  = f'Neumann boundary flux {flux} is a variable defined in domain type {domain_type0}. '
                        msg += f'However, it is being for a state that exists in domain type {domain_type1}.'
                        raise ValueError(msg)
                        
                flux = f'__ones({n_voxels})*{flux}'
                rhs  = f'__Neumann({state}, {flux}, {mapping})'
                
            elif condition.condition_type == 'Dirichlet':
                concentration = condition.condition
                
                concentration = f'__ones({n_voxels})*{concentration}'
                if concentration in self.bulk_variables:
                    msg = 'Dirichlet boundary concentration must be a number, parameter or global variable.'
                    raise NotImplementedError(msg)
                    
                    #Check the domain types match
                    domain_type0 = self.bulk_variables[name]
                    domain_type1 = self.state2domain_type[state]
                    
                    if domain_type0 != domain_type1:
                        msg  = f'Dirichlet boundary concentration {condition} is a variable defined in domain type {domain_type0}. '
                        msg += f'However, it is being for a state that exists in domain type {domain_type1}.'
                        raise ValueError(msg)
                    
                coeff = self.spatial_data.diffusion.find(state, axis) 
                coeff = f'__ones({n_voxels})*{coeff}'
                rhs   = f'__Dirichlet({state}, {coeff}, {concentration}, {mapping})'
            else:
                msg = f'No implementation for {condition.condition_type} boundary.'
                raise NotImplementedError(msg)
            
            return rhs
        
        for state in boundary_conditions.states:
            #Extract information
            domain_type     = state2domain_type[state]
            n_voxels        = len(domain_type2voxel[domain_type])
            
            #Update code
            code += f'\t#{name} {state}\n'
            code += f'\t{ut.bc(state)} = __zeros({n_voxels})\n\n'
            
            for axis, axis_data in boundary_terms[domain_type].items():
                #Extract information
                condition  = boundary_conditions.find(state, axis=-axis)
                mappings   = axis_data['left']
                code      += f'\t_left = {mappings}\n'

                #Make code for calling external function
                lhs   = f'{ut.bc(state)}'
                rhs   = helper(state, axis, n_voxels, condition, '_left')
                code += f'\t{lhs} += {rhs}\n'
                
                #Extract information
                condition  = boundary_conditions.find(state, axis=axis)
                mappings   = axis_data['right']
                code      += f'\t_right = {mappings}\n'

                #Make code for calling external function
                lhs   = f'{ut.bc(state)}'
                rhs   = helper(state, axis, n_voxels, condition, '_right')
                code += f'\t{lhs} += {rhs}\n'
                
            code += f'\t{ut.diff(state)} += {ut.bc(state)}\n\n'
                
        self.boundary_condition_code = code + '\n'
        
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
                       **kwargs
                       ) -> dict:
        
        domain_type    = self.state2domain_type[state_name]
        
        return self._plot_bulk(ax, 
                               state_name, 
                               advection_values, 
                               domain_type, 
                               **kwargs
                               )
    
    def plot_diffusion(self, 
                       ax, 
                       state_name       : str,
                       diffusion_values : np.array,
                       **kwargs
                       ) -> dict:
        
        domain_type    = self.state2domain_type[state_name]
        
        return self._plot_bulk(ax, 
                               state_name, 
                               diffusion_values, 
                               domain_type, 
                               **kwargs
                               )
    
    def plot_boundary_condition(self,
                                ax, 
                                state_name                : str,
                                boundary_condition_values : np.array,
                                **kwargs
                                ) -> dict:
    
        domain_type    = self.state2domain_type[state_name]
        
        return self._plot_bulk(ax, 
                               state_name, 
                               boundary_condition_values, 
                               domain_type, 
                               **kwargs
                               )
    