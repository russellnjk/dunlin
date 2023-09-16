import numpy as np
import sys
from numba   import njit  
from numbers import Number
from typing  import Callable, Literal

import dunlin.utils as ut
from ..grid.grid    import RegularGrid, NestedGrid
from .bidict        import One2One, Many2One
from .reactionstack import (ReactionStack,
                            Domain_type, Domain, Voxel, 
                            State, Parameter,
                            Surface_type
                            )
from dunlin.datastructures import SpatialModelData

#Mass Transfer Functions
@njit
def calculate_advection(X             : np.ndarray, 
                        coeff         : Number|np.ndarray, 
                        left_mapping  : np.ndarray, 
                        right_mapping : np.ndarray,
                        volumes       : np.ndarray
                        ) -> np.ndarray:
    '''
    Calculates the differential associated with advection for a given state. For 
    rhsdct calculations, use calculate_advection_rhsdct.

    Parameters
    ----------
    X : np.ndarray
        An array of values for the state variable. It will be a 1-D array 
        during rhs calculations and a 2-D array during rhsdct calculations.
    coeff : Number|np.ndarray
        The advection coefficient. It will be a number during rhs calculations 
        and a 1-D array during rhsdct calculations.
    left_mapping : np.ndarray
        An N by 3 array. Takes the following form: 
            `[source_idx, destination_idx, scaling_factor]`
        See notes for details.
    right_mapping : np.ndarray
        An N by 3 array. Takes the following form: 
            `[source_idx, destination_idx, scaling_factor]`
        See notes for details. In this case, the scaling factor is the interfacial 
        area between the source and destination voxels.
    volumes : np.ndarray
        A 1-D array of volumes for each voxel of a given domain type. Takes the 
        following form:
            `volume[voxel_idx] = volume_for_voxel_idx`
        
    Returns
    -------
    dX : np.ndarray
        The change in X. 

    Notes
    ------
    The left and right mappings are N by 3 arrays. In other words, each mapping 
    has N rows or 3 elements. Each row has the format 
    `[source_idx, destination_idx, scaling_factor]`
    
    Consider the 1-D system of voxels below. In this example, all except voxel 
    3 belong to the same domain type. Assume the scaling factor is always 1.5.
    
        ###########
        #0#1#2#3#4#
        ###########
    
    To encode a leftward mapping for voxel 1, we would use `[1, 0, 1]`. This 
    tells us that the source is voxel 1, that its leftward neighbour is voxel 0 
    and that the scaling factor is 1. There is no voxel on the left of voxel 0 
    so there is no corresponding mapping. Voxel 3 has voxel 2 on its left but 
    it belongs to a different domain type; there is no mapping for voxel 3 either.
    Voxel 4 has voxel 3 on its left but it belongs to a different domain type; 
    there is no mapping for voxel 4 either.
    
    The final leftward mapping should look like this: `[[1, 0, 1.5], [2, 1, 1.5]]`.
    The order of the rows does not matter.
    
    The rightward mapping should look like this: `[[0, 1, 1.5], [1, 2, 1.5]]`. The 
    order of the rows does not matter.
    
    '''
    
    dX = np.zeros(X.shape)
    
    #Case 1: coeff < 0
    if coeff < 0:
        for row in left_mapping:
            src    = row[0]
            dst    = row[1]
            scale  = row[2]
            X_i    = X[src]
            volume = volumes[src]
            dX_i   = -coeff * X_i * scale / volume
            
            dX[src] -= dX_i
            dX[dst] += dX_i
    
    #Case 2: coeff > 0
    elif coeff > 0:
        for row  in right_mapping:
            src      = row[0]
            dst      = row[1]
            scale    = row[2]
            X_i      = X[src]
            volume_i = volumes[src]
            volume_j = volumes[dst]
            change   = coeff * X_i * scale 
            
            dX[src] -= change / volume_i
            dX[dst] += change / volume_j
            
    return dX

@njit
def calculate_advection_rhsdct(X             : np.ndarray, 
                               coeffs        : Number|np.ndarray, 
                               left_mapping  : np.ndarray, 
                               right_mapping : np.ndarray,
                               volumes       : np.ndarray
                               ) -> np.ndarray:
    '''
    Calculates the differential associated with advection for a given state.
    Meant only rhsdct calculations.
    '''
    
    dX     = np.zeros(X.shape)
    coeffs = np.ones(X.shape[1]) * coeffs
    
    for i in range(len(coeffs)):
        coeff = coeffs[i]
        
        #Case 1: coeff < 0
        if coeff < 0:
            for row in left_mapping:
                src    = row[0]
                dst    = row[1]
                scale  = row[2]
                X_i    = X[src, i]
                volume = volumes[src]
                dX_i   = -coeff * X_i * scale / volume
                
                dX[src, i] -= dX_i
                dX[dst, i] += dX_i
        
        #Case 2: coeff > 0
        elif coeff > 0:
            for row  in right_mapping:
                src      = row[0]
                dst      = row[1]
                scale    = row[2]
                X_i      = X[src, i]
                volume_i = volumes[src]
                volume_j = volumes[dst]
                change   = coeff * X_i * scale 
                
                dX[src, i] -= change / volume_i
                dX[dst, i] += change / volume_j
            
    return dX

@njit
def calculate_diffusion(X            : np.ndarray, 
                        coeff        : Number|np.ndarray, 
                        left_mapping : np.ndarray, 
                        volumes      : np.ndarray
                        ) -> np.ndarray:
    '''
    Calculates the differential associated with diffusion for a given state.    

    Parameters
    ----------
    X : np.ndarray
        See the explanation for calculate_advection. It will be a 1-D array 
        during rhs calculations and a 2-D array during rhsdct calculations.
    coeff : Number|np.ndarray
        The diffusion coefficient. It will be a number during rhs calculations 
        and a 1-D array during rhsdct calculations.
    left_mapping : np.ndarray
        See the explanation for calculate_advection. It is similar but the 
        columns are source, destination, area/distance.
    volumes : np.ndarray
        See the explanation for calculate_advection.

    Returns
    -------
    dX : np.ndarray
        See the explanation for calculate_advection.
        
    '''
    #Overhead
    dX = np.zeros(X.shape)
    
    for row in left_mapping:
        src      = row[0]
        dst      = row[1]
        scale    = row[2]
        X_i      = X[src]
        X_j      = X[dst]
        volume_i = volumes[src]
        volume_j = volumes[dst]
        
        #Find concentration difference
        #If src > dst, then change is a positive number
        d_concentration   = X_i - X_j
        
        #Calculate mass transferred
        change = coeff * d_concentration * scale
        
        dX[src] -= change / volume_i
        dX[dst] += change / volume_j
    
    return dX

@njit
def calculate_neumann(X       : np.ndarray, 
                      fluxes  : np.ndarray, 
                      mapping : np.ndarray,
                      volumes : np.ndarray
                      ) -> np.ndarray:
    '''
    Calculates the differential associated with Dirichlet boundary conditions 
    for a given state variable.

    Parameters
    ----------
    X : np.ndarray
        An array of values for the state variable. It will be a 1-D array 
        during rhs calculations and a 2-D array during rhsdct calculations.
    fluxes : np.ndarray
        The fluxes at the boundary. It will be a 1-D array with the same length 
        as X during rhs calculation. During rhsdct calculation, it will be a 
        2-D array with the same shape as X.
    mapping : np.ndarray
        See explanation for calculate_advection. It is similar but the columns 
        are source and area.
    volumes : np.ndarray
        See explanation for calculate_advection.

    Returns
    -------
    dX : np.ndarray
        The change in X. 

    '''
    
    dX = np.zeros(X.shape)
    
    for row in mapping:
        src    = row[0]
        scale  = row[1]
        flux   = fluxes[src]
        volume = volumes[src]
        
        dX[src] += flux * scale / volume
        
    return dX

@njit
def calculate_dirichlet(X              : np.ndarray, 
                        coeff          : Number|np.ndarray,
                        concentrations : np.ndarray, 
                        mapping        : np.ndarray,
                        volumes        : np.ndarray 
                        ) -> np.ndarray:
    '''
    Calculates the differential associated with Dirichlet boundary conditions 
    for a given state variable.

    Parameters
    ----------
    X : np.ndarray
        An array of values for the state variable. It will be a 1-D array 
        during rhs calculations and a 2-D array during rhsdct calculations.
    coeff : Number|np.ndarray
        The diffusion coefficient. It will be a number during rhs calculations 
        and a 1-D array during rhsdct calculations.
    concentrations : np.ndarray
        The concentration at the boundary. It will be 1-D array during rhs 
        calculation and a 2-D array during rhsdct calculation.
    mapping : np.ndarray
        See explanation for calculate_advection. It is similar but the columns 
        are source and area/distance.
    volumes : np.ndarray
        See explanation for calculate_advection.

    Returns
    -------
    dX : np.ndarray
        The change in X. 

    '''
    
    dX = np.zeros(X.shape)
    
    for row in mapping:
        src           = row[0]
        scale         = row[1]
        concentration = concentrations[src]
        volume        = volumes[src]
        
        #Find the difference in concentration
        d_concentration = X[src] - concentration
        
        #Calculate mass transferred
        change = coeff * d_concentration * scale
        
        dX[src] -= change / volume
    
    return dX

#MassTransferStack
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
    signature         : list[str]
    args              : dict
    rhs_functions     : dict[str, Callable]
    rhsdct_functions  : dict[str, Callable]
    formatter         : str
    
    surface_data         : dict[Surface_type, dict]
    global_variables     : set
    bulk_variables       : dict[str, Domain_type]
    surface_variables    : dict[str, Surface_type]
    variable_code        : str
    bulk_reactions       : dict[str, Domain_type]
    surface_reactions    : dict[str, Surface_type]
    reaction_code        : str
    
    domain_type2volume      : dict[Domain_type, dict[int, float]]
    bulk_data               : dict[Domain_type, dict[int, dict]]
    boundary_data           : dict[Domain_type, dict[int, dict]]
    advection_code          : str
    diffusion_code          : str
    boundary_condition_code : str
    
    def __init__(self, spatial_data: SpatialModelData):
        #Data structures for self._add_voxel
        self.domain_type2volume = {}
        
        #Data structures for self._add_bulk
        self.bulk_data = {}
        self.advection_data = {}
        self.diffusion_data = {}
        self.boundary_data  = {}
        
        #Call the parent constructor
        super().__init__(spatial_data)
        self._reformat_bulk_data()
        
        #Update calculators
        self.rhs_functions['__advection'] = calculate_advection
        self.rhs_functions['__diffusion'] = calculate_diffusion
        self.rhs_functions['__Neumann'  ] = calculate_neumann
        self.rhs_functions['__Dirichlet'] = calculate_dirichlet
        
        self.rhsdct_functions['__advection'] = calculate_advection_rhsdct
        self.rhsdct_functions['__diffusion'] = calculate_diffusion
        self.rhsdct_functions['__Neumann'  ] = calculate_neumann
        self.rhsdct_functions['__Dirichlet'] = calculate_dirichlet
        
        #Parse the advection
        self.advection_code = ''
        self._add_advection_code()
        
        #Parse the diffusion
        self.diffusion_code = ''
        self._add_diffusion_code()
        
        #Parse the boundary conditions
        self._reformat_boundary_data()
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
        domain_type2voxel     = self.voxel2domain_type.inverse 
        
        size                         = sizes[voxel]
        domain_type_idx, domain_type = voxel2domain_type_idx[voxel]
        
        if domain_type not in domain_type2volume:
            length                          = len(domain_type2voxel[domain_type]) 
            domain_type2volume[domain_type] = np.zeros(length)
        
        domain_type2volume[domain_type][domain_type_idx] = size**nd
        
    def _add_bulk(self, voxel0, voxel1, shift) -> None:
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes 
        nd                    = self.ndims
        bulk_data             = self.bulk_data
        domain_type           = self.voxel2domain_type[voxel0]
        axis                  = abs(shift)
        
        voxel0_idx = voxel2domain_type_idx[voxel0][0]
        voxel1_idx = voxel2domain_type_idx[voxel1][0]
        size0      = sizes[voxel0]
        size1      = sizes[voxel1]
        distance   = (size0 + size1)/2
        area       = min(size0, size1)**(nd-1)
        
        #Update mass transfer data
        default = {'mappings': {'left': [], 'right': []}}
        datum   = (bulk_data
                   .setdefault(domain_type, {})
                   .setdefault(axis, default)
                   )
        
        mapping = (voxel0_idx, voxel1_idx, area, area/distance)
        
        if shift < 0:
            datum['mappings']['left'].append(mapping)
        else:
            datum['mappings']['right'].append(mapping)
        
    def _add_boundary(self, voxel, shift) -> None:
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes 
        nd                    = self.ndims
        boundary_data         = self.boundary_data
        domain_type           = self.voxel2domain_type[voxel]
        axis                  = abs(shift)
        
        voxel_idx = voxel2domain_type_idx[voxel][0]
        size      = sizes[voxel]
        distance  = size
        area      = size**(nd-1)
        
        #Set defaults in the nested dictionary
        default = {'mappings': {'left': [], 'right': []}}
        datum   = (boundary_data
                   .setdefault(domain_type, {})
                   .setdefault(axis, default)
                   )
        
        mapping = (voxel_idx, area, area/distance)
        
        if shift < 0:
            datum['mappings']['left'].append(mapping)
        else:
            datum['mappings']['right'].append(mapping)
            
    def _reformat_bulk_data(self) -> None:
        data = self.bulk_data
        
        for domain_type, domain_type_data in data.items():
            for axis, axis_data in domain_type_data.items():
                mappings = axis_data['mappings']
                
                dtype = [('source',        np.int32  ),
                         ('destination',   np.int32  ),
                         ('area',          np.float64),
                         ('area/distance', np.float64)
                         ]
                
                #Reformat left mappings
                if 'left' in mappings:
                    array = np.array(mappings['left'],
                                      dtype = dtype
                                      )
                    mappings['left'] = array
                
                #Reformat right mappings
                if 'right' in mappings:
                    array = np.array(mappings['right'],
                                      dtype = dtype
                                      )
                    mappings['right'] = array
                
    def _reformat_boundary_data(self) -> None:
        data = self.boundary_data
        
        for domain_type, domain_type_data in data.items():
            for axis, axis_data in domain_type_data.items():
                mappings = axis_data['mappings']
                
                dtype = [('source',        np.int32  ),
                         ('area',          np.float64),
                         ('area/distance', np.float64)
                         ]
                
                #Reformat left mappings
                array = np.array(mappings['left'],
                                  dtype = dtype
                                  )
                mappings['left'] = array
                
                #Reformat right mappings
                array = np.array(mappings['right'],
                                  dtype = dtype
                                  )
                mappings['right'] = array
                    
    ###########################################################################
    #Bulk Transfer
    ###########################################################################
    def _add_advection_code(self) -> None:
        bulk_data          = self.bulk_data
        advection          = self.spatial_data.advection
        domain_type2volume = self.domain_type2volume
        state2domain_type  = self.state2domain_type  
        signature          = self.signature
        args               = self.args
        code               = ''
        keys               = ['source', 'destination', 'area']
        
        for state in advection:
            domain_type  = state2domain_type[state]
            state_       = ut.undot(state)
            code        += f'\t#Advection {state}\n'
            lhs          = f'{ut.adv(state_)}'
            code        += f'\t{lhs} = __zeros({state_}.shape)\n'
            
            for axis, axis_data in bulk_data[domain_type].items():
                #Coeffs
                coeffs = f'{advection.get(state, axis)}'
                
                #Mappings
                left_mapping = f'_adv_{domain_type}_{axis}_left'
                right_mapping = f'_adv_{domain_type}_{axis}_right'

                #Volumes
                volumes = f'_vol_{domain_type}'
                
                #Code for calling calculator
                rhs = f'__advection({state_}, {coeffs}, {left_mapping}, {right_mapping}, {volumes})'
                code += f'\t{lhs} += {rhs}\n'
            
                #Update the signature and args
                if left_mapping not in args:
                    signature.append(left_mapping)
                    mappings            = axis_data['mappings']
                    args[left_mapping]  = mappings['left'][keys]
                    
                if right_mapping not in args:
                    signature.append(right_mapping)
                    mappings            = axis_data['mappings']
                    args[right_mapping] = mappings['right'][keys]
                
                    
                if volumes not in args:
                    #Volumes
                    signature.append(volumes)
                    args[volumes] = domain_type2volume[domain_type]
                    
            code += f'\t{ut.diff(state_)} += {lhs}\n\n'
                
        self.advection_code = code + '\n'
            
    def _add_diffusion_code(self) -> None:
        bulk_data          = self.bulk_data
        diffusion          = self.spatial_data.diffusion
        domain_type2volume = self.domain_type2volume
        state2domain_type  = self.state2domain_type  
        signature          = self.signature
        args               = self.args
        code               = ''
        keys               = ['source', 'destination', 'area/distance']
        
        for state in diffusion:
            domain_type  = state2domain_type[state]
            state_       = ut.undot(state)
            code        += f'\t#Diffusion {state}\n'
            lhs          = f'{ut.dfn(state_)}'
            code        += f'\t{lhs} = __zeros({state_}.shape)\n'
            
            for axis, axis_data in bulk_data[domain_type].items():
                #Coeffs
                coeffs = f'{diffusion.get(state, axis)}'
                
                #Mappings
                left_mapping = f'_dfn_{domain_type}_{axis}_left'
                
                #Volumes
                volumes = f'_vol_{domain_type}'
                
                #Code for calling calculator
                rhs = f'__diffusion({state_}, {coeffs}, {left_mapping}, {volumes})'
                code += f'\t{lhs} += {rhs}\n'
            
                #Update the signature and args
                if left_mapping not in args:
                    #Update the signature
                    signature.append(left_mapping)
                    
                    #Extract the arrays required for calculation
                    mappings            = axis_data['mappings']
                    args[left_mapping]  = mappings['left'][keys]
                
                if volumes not in args:
                    #Volumes
                    signature.append(volumes)
                    args[volumes] = domain_type2volume[domain_type]
                    
            code += f'\t{ut.diff(state_)} += {lhs}\n\n'
                
        self.diffusion_code = code + '\n'
        
    ###########################################################################
    #Boundary Conditions
    ###########################################################################
    def _add_boundary_code(self) -> None:
        boundary_data       = self.boundary_data
        boundary_conditions = self.spatial_data.boundary_conditions
        state2domain_type   = self.state2domain_type
        code                = '#Boundary Conditions\n'
        
        for state in boundary_conditions:
            #Extract information
            state_          = ut.undot(state)
            domain_type     = state2domain_type[state]
            # n_voxels        = len(domain_type2voxel[domain_type])
            # volumes         = f'_vol_{domain_type}'
            
            #Update code
            code  = f'\t#Boundary conditions {state}\n'
            code += f'\t{ut.bc(state_)} = __zeros({state}.shape)\n\n'
            
            for axis, axis_data in boundary_data[domain_type].items():
                #Parse left side/ axis minimum
                bound      = 'min'
                condition  = boundary_conditions.get(state, 
                                                     axis=axis, 
                                                     bound=bound
                                                     )
                if condition:
                    value          = condition['value']
                    condition_type = condition['condition_type']
                    
                    if condition_type == 'Neumann':
                        code += self._make_neumann_code(state, axis, bound, value)
                    elif condition_type == 'Dirichlet':
                        code += self._add_dirichlet_code(state, axis, bound, value)
                    else:
                        msg = f'Could not add code for {condition_type} boundary.'
                        raise NotImplementedError(msg)
                        
                #Parse right side/axis maximum
                bound      = 'max'
                condition  = boundary_conditions.get(state, 
                                                     axis=axis, 
                                                     bound=bound
                                                     )
                if condition:
                    value          = condition['value']
                    condition_type = condition['condition_type']
                    
                    if condition_type == 'Neumann':
                        code += self._make_neumann_code(state, axis, bound, value)
                    elif condition_type == 'Dirichlet':
                        code += self._add_dirichlet_code(state, axis, bound, value)
                    else:
                        msg = f'Could not add code for {condition_type} boundary.'
                        raise NotImplementedError(msg)
        
            code += f'\t{ut.diff(state_)} += _bc_{state}\n\n'
            
        self.boundary_condition_code += code
    
    def _make_neumann_code(self, 
                           state : State, 
                           axis  : int, 
                           bound : Literal['min', 'max'], 
                           value : str|Number
                           ) -> str:
        
        code = f'\t#Neumann boundary, {state}, {axis}, {bound}\n'
        
        if ut.isnum(value, include_strings=True):
            fluxes = f'__ones({state}.shape) * {value}'
        elif value in self.spatial_data.parameters:
            fluxes = f'__ones({state}.shape) * {value}'
        elif value in self.global_variables:
            fluxes = f'__ones({state}.shape) * {value}'
        else:
            msg  = 'Could not make code boundary condition {state}, {axis}, {bound}. '
            msg += 'Value for Neumannn boundary must be a '
            msg += 'number, parameter or global variable. '
            msg += 'Received : {value}.'
            raise ValueError(msg)
        
        lhs          = f'_bc_{state}'
        domain_type  = self.state2domain_type[state]
        volumes      = f'_vol_{domain_type}'
        mapping      = f'_bc_{domain_type}_{axis}_{bound}'
        rhs          = f'__Neumann({state}, {fluxes}, {mapping}, {volumes})'
        code        += f'\t{lhs} += {rhs}\n\n'
        
        #Update signature and args
        if mapping not in self.args:
            self.signature.append(mapping)
            
            mappings = self.boundary_data[domain_type][axis]['mappings']
            keys     = ['source', 'area']
            
            if bound == 'min':
                self.args[mapping] = mappings['left'][keys]
            else:
                self.args[mapping] = mappings['right'][keys]
        
        if volumes not in self.args:
            self.signature.append(volumes)
            self.args[volumes] = self.domain_type2volume[domain_type]
        
        return code     
    
    def _add_dirichlet_code(self, 
                            state : State, 
                            axis  : int, 
                            bound : str, 
                            value : str|Number
                            ) -> str:
        
        coeff = self.spatial_data.diffusion.get(state, axis)
        
        #Coefficient is zero or None
        #Return immediately
        if not coeff:
            return ''
        
        #Else proceed with code generation
        code = f'\t#Dirichlet boundary, {state}, {axis}, {bound}\n'
        
        if ut.isnum(value, include_strings=True):
            concentrations = f'__ones({state}.shape) * {value}'
        elif value in self.spatial_data.parameters:
            concentrations = f'__ones({state}.shape) * {value}'
        elif value in self.global_variables:
            concentrations = f'__ones({state}.shape) * {value}'
        elif value in self.bulk_variables:
            concentrations = value
        else:
            msg  = 'Could not make code boundary condition {state}, {axis}, {bound}. '
            msg += 'Value for Neumannn boundary must be a '
            msg += 'number, parameter, global or bulk variable. '
            msg += 'Received : {value}.'
            raise ValueError(msg)
        
        lhs          = f'_bc_{state}'
        domain_type  = self.state2domain_type[state]
        volumes      = f'_vol_{domain_type}'
        mapping      = f'_bc_{domain_type}_{axis}_{bound}'
        rhs          = f'__Dirichlet({state}, {coeff}, {concentrations}, {mapping}, {volumes})'
        code        += f'\t{lhs} += {rhs}\n\n'
        
        #Update signature and args
        if mapping not in self.args:
            self.signature.append(mapping)
            
            mappings = self.boundary_data[domain_type][axis]['mappings']
            keys     = ['source', 'area/distance']
            
            if bound == 'min':
                self.args[mapping] = mappings['left'][keys]
            else:
                self.args[mapping] = mappings['right'][keys]
        
        if volumes not in self.args:
            self.signature.append(volumes)
            self.args[volumes] = self.domain_type2volume[domain_type]
            
        return code     
    
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
    