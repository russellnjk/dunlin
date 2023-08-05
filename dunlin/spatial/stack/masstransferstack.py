import numpy as np
from numba   import njit  
from numbers import Number

import dunlin.utils as ut
from ..grid.grid    import RegularGrid, NestedGrid
from .bidict        import One2One, Many2One
from .reactionstack import (ReactionStack,
                            Domain_type, Domain, Voxel, 
                            State, Parameter,
                            Surface_type
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
def calculate_neumann_boundary(X       : np.ndarray, 
                               flux    : np.ndarray, 
                               mapping : np.ndarray,
                               volumes : np.ndarray
                               ) -> np.ndarray:
    mass_transferred = np.zeros(X.shape)
    
    domain_type_idxs = mapping[0].astype(np.int64)
    scale            = mapping[1] 
    
    #Apply the indices to extract the quantities
    flux_boundary = flux[domain_type_idxs]
    
    #Calculate the change in concentration
    change = flux_boundary * scale
    
    mass_transferred[domain_type_idxs] += change
    
    dX = mass_transferred/volumes
    return dX

@njit
def calculate_dirichlet_boundary(X             : np.ndarray, 
                                 coeff         : np.ndarray,
                                 concentration : np.ndarray, 
                                 mapping       : np.ndarray,
                                 volumes       : np.ndarray 
                                 ) -> np.ndarray:
    #Overhead
    mass_transferred        = np.zeros(X.shape)
    
    #Process the diffusion at the axis minimum
    #Find the center and the left voxels
    src_idxs = mapping[0].astype(np.int64)
    scale    = mapping[1]
    
    #Find concentration gradient
    gradient = X[src_idxs] - concentration[src_idxs]
    coeff_   = coeff[src_idxs]
    
    #Calculate change in concentration
    change = coeff_ * gradient * scale
    
    mass_transferred[src_idxs] -= change
    
    return mass_transferred

@njit
def calculate_advection(X              : np.ndarray, 
                        coeff          : np.ndarray, 
                        left_mappings  : tuple[np.ndarray], 
                        right_mappings : tuple[np.ndarray],
                        volumes        : np.ndarray
                         ) -> np.ndarray:
    '''
    The left and right mappings are a tuple of 2×2 arrays. In other words, in 
    each mapping, there are 2 rows. The first row denotes the domain index of 
    the neighbours, the second row denotes a scaling factor. The length of each 
    row equals that of X.
    
    Consider the 1-D system of voxels below. In this example, they all belong 
    to the same domain type.
    
        #############
        #0#1#2#3#4#5#
        #############
    
    Suppose now that we want to construct the left mappings for the system. Because 
    there are 6 voxels, the length of each row is 6.
    
    The left neighbour of the nth voxel is the nth-1 voxel. Voxel 0 has no left 
    neighbour and we will denote it with a dummy value of -1. Note that the index 
    must always be 0 or more. Any negative values will be interpreted as not 
    having a neighbour. This gives us the first row: `[-1, 0, 1, 2, 3, 4]`.
    
    The scaling factor depends on things like the distance between the voxels. 
    The exact formulation depends on whether advection or diffusion are being 
    calculated. In this example, all the voxels are equally spaced so the 
    second row is as follows: `[0, 1, 1, 1, 1, 1]`. Because voxel 0 has no 
    neighbour, the value is ignored.
    
    This gives us a left mapping of: `[[-1, 0, 1, 2, 3, 4], [0, 1, 1, 1, 1, 1]]`
    
    Suppose voxel 3 belongs a different domain type. The left mapping would 
    then be: `[[-1, 0, 1, -1, 3, 4], [0, 1, 1, 0, 1, 1]]`. Once again, the 
    values in the second row that correspond to voxels 0 and 3 do not actually 
    matter as they will be ignored.
    
    In the event a voxel has two neighbours, it is not possible to put the 
    information for both voxel-neighbour pairs in the same left mapping. A second 
    left mapping must be created. This is why left mappings is a tuple; it 
    accommodates multiple left mappings.
    
    '''
    
    mass_transferred = np.zeros(X.shape)
    abs_coeff        = np.abs(coeff)
    
    #Case 1: coeff < 0
    for left_mapping in left_mappings:
        #Determine which elements are negative
        #Find the indices of the sources and destinations
        left_src_idxs = np.where((coeff < 0) & (left_mapping[0] > -1))
        left_src_idxs = left_src_idxs[0].astype(np.int64)
        left_dst_idxs = left_mapping[0][left_src_idxs]
        scale_left    = left_mapping[1][left_src_idxs]
        
        #Apply the indices to extract the quantities for leftward calculation
        coeff_left = abs_coeff[left_src_idxs] 
        X_src      = X[left_src_idxs]
        
        #Calculate change in concentration
        change = coeff_left * X_src * scale_left
    
        mass_transferred[left_src_idxs] -= change
        mass_transferred[left_dst_idxs] += change
    
    #Case 2: coeff > 0
    for right_mapping in right_mappings:
        #Determine which elements are negative
        #Find the indices of the sources and destinations
        right_src_idxs = np.where((coeff > 0) & (right_mapping[0] > -1))
        right_src_idxs = right_src_idxs[0].astype(np.int64)
        right_dst_idxs = right_mapping[0][right_src_idxs]
        scale_right    = right_mapping[1][right_src_idxs]
        
        #Apply the indices to extract the quantities for rightward calculation
        coeff_right = abs_coeff[right_src_idxs]
        X_src       = X[right_src_idxs]
        
        #Calculate change in concentration
        change = coeff_right * X_src * scale_right
    
        mass_transferred[right_src_idxs] -= change
        mass_transferred[right_dst_idxs] += change
        
    dX = mass_transferred/volumes
    return dX

@njit
def calculate_diffusion(X              : np.ndarray, 
                        coeff          : Number, 
                        left_mappings  : tuple[np.ndarray], 
                        volumes        : np.ndarray
                        ) -> np.ndarray:
    #Overhead
    mass_transferred = np.zeros(X.shape)
    
    for left_mapping in left_mappings:
        #Find the center and the left voxels
        left_src_idxs = np.where(left_mapping[0] > -1)[0].astype(np.int64)
        left_dst_idxs = left_mapping[0][left_src_idxs]
        scale_left    = left_mapping[1][left_src_idxs]
        
        #Find concentration gradient
        gradient   = X[left_src_idxs] - X[left_dst_idxs]
        coeff_left = coeff[left_src_idxs]
        
        #Calculate change in concentration
        change = coeff_left * gradient * scale_left
        
        mass_transferred[left_src_idxs] -= change
        mass_transferred[left_dst_idxs] += change
    
    dX = mass_transferred/volumes
    return dX

@njit
def calculate_neumann_boundary2(X       : np.ndarray, 
                                flux    : np.ndarray, 
                                mapping : np.ndarray,
                                volumes : np.ndarray
                                ) -> np.ndarray:
    dX = np.zeros(X.shape)
    
    for col_num in range(flux.shape[1]):
        flux_ = flux[:, col_num]
        X_    = X[:, col_num]
        
        dX[:,col_num] = calculate_neumann_boundary(X_, 
                                                   flux_, 
                                                   mapping,
                                                   volumes
                                                   )
        
    return dX

@njit
def calculate_dirichlet_boundary2(X             : np.ndarray, 
                                  coeff         : np.ndarray,
                                  concentration : np.ndarray, 
                                  mapping       : np.ndarray, 
                                  volumes       : np.ndarray
                                  ) -> np.ndarray:
    #Overhead
    dX = np.zeros(X.shape)
    
    for col_num in range(concentration.shape[1]):
        coeff_         = coeff[:, col_num]
        concentration_ = concentration[:, col_num]
        X_             = X[:, col_num]
        
        dX[:,col_num] = calculate_dirichlet_boundary(X_, 
                                                     coeff_, 
                                                     concentration_, 
                                                     mapping,
                                                     volumes
                                                     )
    
    return dX
    
# @njit
def calculate_advection2(X              : np.ndarray, 
                         coeff          : np.ndarray, 
                         left_mappings  : tuple[np.ndarray], 
                         right_mappings : tuple[np.ndarray],
                         volumes         : np.ndarray
                         ) -> np.ndarray:
    
    dX = np.zeros(X.shape)
    
    for col_num in range(coeff.shape[1]):
        coeff_ = coeff[:, col_num]
        X_     = X[:, col_num]
        
        dX[:,col_num] = calculate_advection(X_, 
                                            coeff_, 
                                            left_mappings, 
                                            right_mappings,
                                            volumes
                                            )
    
    return dX

@njit
def calculate_diffusion2(X              : np.ndarray, 
                         coeff          : Number, 
                         left_mappings  : tuple[np.ndarray], 
                         volumes        : np.ndarray  
                         ) -> np.ndarray:
    #Overhead
    dX = np.zeros(X.shape)
    
    for col_num in range(coeff.shape[1]):
        coeff_ = coeff[:, col_num]
        X_     = X[:, col_num]
        
        dX[:,col_num] = calculate_diffusion(X_, 
                                            coeff_, 
                                            left_mappings, 
                                            volumes
                                            )
        
    return dX

#ReactionStack
class MassTransferStack(ReactionStack):
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
    reaction_code        : str
    reaction_code_rhsdct : str
    
    domain_type2volume : dict[Domain_type, dict[int, float]]
    advection_terms    : dict[Domain_type, dict[int, dict]]
    diffusion_terms    : dict[Domain_type, dict[int, dict]]
    boundary_terms     : dict[Domain_type, dict[int, dict]]
    advection_code     : str
    diffusion_code     : str
    boundary_code      : str
    
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
        self.rhs_functions['__advection'] = calculate_advection
        self.rhs_functions['__diffusion'] = calculate_diffusion
        self.rhs_functions['__Neumann'  ] = calculate_neumann_boundary
        self.rhs_functions['__Dirichlet'] = calculate_dirichlet_boundary
        
        self.rhsdct_functions['__advection'] = calculate_advection2
        self.rhsdct_functions['__diffusion'] = calculate_diffusion2
        self.rhsdct_functions['__Neumann'  ] = calculate_neumann_boundary2
        self.rhsdct_functions['__Dirichlet'] = calculate_dirichlet_boundary2
        
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
        helper0 = lambda: np.array([[-1]*n, [0]*n]) 
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
            #0 : dst, 1: scale
            if advection_term[m][0][c0] == -1:
                advection_term[m][0][c0] = c1
                advection_term[m][1][c0] = A
                
                if diffusion_term is not None:
                    diffusion_term[m][0][c0] = c1
                    diffusion_term[m][1][c0] = A/D
                    
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
        A    = size**(nd-1)
        D    = size
        
        #Find a vacant element and store the information
        boundary_term[0].append(c)
        boundary_term[1].append(A/D)
    
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
        code               = ''
        name               = '#Advection'
        domain_type2volume = self.domain_type2volume
            
        for state in advection:
            #Extract information
            state_       = ut.undot(state)
            domain_type  = state2domain_type[state]
            code        += f'\t#{name} {state}\n'
            code        += f'\t{ut.adv(state_)} = __zeros({state_}.shape)\n\n'
            
            for axis, axis_data in advection_terms[domain_type].items():
                #Extract information
                coeff           = advection.get(state, axis)
                left_mappings   = axis_data['left']
                right_mappings  = axis_data['right']
                volumes         = domain_type2volume[domain_type]
                volumes         = f'__array({volumes})'
                code           += f'\t_left = {left_mappings}\n'
                code           += f'\t_right = {right_mappings}\n'
                
                
                #Make code for calling external function
                lhs   = f'{ut.adv(state_)}'
                
                if coeff in self.bulk_variables:
                    #Check the domain types match
                    domain_type0 = self.bulk_variables[name]
                    domain_type1 = self.state2domain_type[state]
                    
                    if domain_type0 != domain_type1:
                        msg  = f'Advection coefficient {coeff} is a variable defined in domain type {domain_type0}. '
                        msg += f'However, it is being for a state that exists in domain type {domain_type1}.'
                        raise ValueError(msg)
                    
                coeff = f'__ones({state_}.shape)*{coeff}'
                rhs   = f'__advection({state_}, {coeff}, _left, _right, {volumes})\n'
                code += f'\t{lhs} += {rhs}\n'
                
            code += f'\t{ut.diff(state_)} += {ut.adv(state_)}\n\n'
                
        self.advection_code = code + '\n'
            
    def _add_diffusion_code(self) -> None:
        diffusion_terms = self.diffusion_terms
        diffusion       = self.spatial_data.diffusion
        
        state2domain_type     = self.state2domain_type
        code                  = ''
        name                  = '#Diffusion'
        domain_type2volume    = self.domain_type2volume
        
        for state in diffusion:
            #Extract information
            state_      = ut.undot(state)
            domain_type = state2domain_type[state]
            volumes     = domain_type2volume[domain_type]
            volumes     = f'__array({volumes})'
            
            #Update code
            code += f'\t#{name} {state}\n'
            code += f'\t{ut.dfn(state_)} = __zeros({state_}.shape)\n\n'
            
            for axis, axis_data in diffusion_terms[domain_type].items():
                #Extract information
                coeff          = diffusion.get(state, axis)
                left_mappings  = axis_data['left']
                code          += f'\t_left = {left_mappings}\n'
                
                #Make code for calling external function
                lhs   = f'{ut.dfn(state_)}'
                
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
                
                coeff = f'__ones({state_}.shape)*{coeff}'
                rhs   = f'__diffusion({state_}, {coeff}, _left, {volumes})\n'
                code += f'\t{lhs} += {rhs}\n'
                
            code += f'\t{ut.diff(state_)} += {ut.dfn(state_)}\n\n'
                
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
        domain_type2volume    = self.domain_type2volume
        
        def helper(state, axis, n_voxels, condition, mapping, volumes):
            state_         = ut.undot(state)
            value          = ut.undot(condition['value'])
            condition_type = condition['condition_type']
            
            if condition_type == 'Neumann':
                flux = value
                
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
                        
                flux = f'__ones({state_}.shape)*{flux}'
                rhs  = f'__Neumann({state_}, {flux}, {mapping}, {volumes})'
                
            elif condition_type == 'Dirichlet':
                concentration = value
                
                concentration = f'__ones({state_}.shape)*{concentration}'
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
                    
                coeff = self.spatial_data.diffusion.get(state, axis) 
                coeff = f'__ones({state_}.shape)*{coeff}'
                rhs   = f'__Dirichlet({state_}, {coeff}, {concentration}, {mapping}, {volumes})'
            else:
                msg = f'No implementation for {condition_type} boundary.'
                raise NotImplementedError(msg)
            
            return rhs
        
        for state in boundary_conditions:
            #Extract information
            state_          = ut.undot(state)
            domain_type     = state2domain_type[state]
            n_voxels        = len(domain_type2voxel[domain_type])
            volumes         = domain_type2volume[domain_type]
            volumes         = f'__array({volumes})'
            
            #Update code
            code += f'\t#{name} {state}\n'
            code += f'\t{ut.bc(state_)} = __zeros({state}.shape)\n\n'
            
            for axis, axis_data in boundary_terms[domain_type].items():
                #Parse left side/axis minimum
                #Extract information
                condition  = boundary_conditions.get(state, axis=axis, bound='min')
                mappings   = axis_data['left']
                code      += f'\t_left = {mappings}\n'

                #Make code for calling external function
                lhs   = f'{ut.bc(state_)}'
                rhs   = helper(state, axis, n_voxels, condition, '_left', volumes)
                code += f'\t{lhs} += {rhs}\n'
                
                #Parse right side/axis maximum
                #Extract information
                condition  = boundary_conditions.get(state, axis=axis, bound='max')
                mappings   = axis_data['right']
                code      += f'\t_right = {mappings}\n'

                #Make code for calling external function
                lhs   = f'{ut.bc(state_)}'
                rhs   = helper(state, axis, n_voxels, condition, '_right', volumes)
                code += f'\t{lhs} += {rhs}\n'
                
            code += f'\t{ut.diff(state_)} += {ut.bc(state_)}\n\n'
                
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
                       advection_values : np.ndarray,
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
                       diffusion_values : np.ndarray,
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
                                boundary_condition_values : np.ndarray,
                                **kwargs
                                ) -> dict:
    
        domain_type    = self.state2domain_type[state_name]
        
        return self._plot_bulk(ax, 
                               state_name, 
                               boundary_condition_values, 
                               domain_type, 
                               **kwargs
                               )
    