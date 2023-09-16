import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
from matplotlib.patches import Rectangle
from numbers            import Number
from typing             import Callable, Literal, Union

import dunlin.utils      as ut
from ..grid.grid   import RegularGrid, NestedGrid
from .bidict       import One2One, Many2One
from .domainstack  import (DomainStack,
                           Domain_type, Domain, Voxel,
                           )
from ..geometrydefinition   import make_shapes
from dunlin.datastructures import SpatialModelData

#Typing
State     = str
Parameter = str

#StateStack
class StateStack(DomainStack):
    '''
    This class makes use of SpatialModelData. All tests should involve the use 
    of SpatialModelData instead of raw Python data.
    '''
    
    #Expected mappings and attributes
    grid                  : Union[RegularGrid, NestedGrid]
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
    
    def __init__(self, spatial_data: SpatialModelData):
        #Arguments for calling parent constructor
        grid_config        = spatial_data['grid_config']
        shapes             = make_shapes(spatial_data['geometry_definitions'])
        domain_type2domain = {k: {kk: vv for kk, vv in v.domain2internal_point.items()} for k, v in spatial_data.domain_types.items()}
        
        if spatial_data.surfaces.domain2surface:
            domain2surface = One2One('surface', 'domain', spatial_data.surfaces.domain2surface)
            surface2domain = domain2surface.inverse
            
        else:
            domain2surface = One2One('surface', 'domain', {})
            surface2domain = domain2surface.inverse
        
        #Template the mappings for _add_voxel 
        state2domain_type      = spatial_data.domain_types.state2domain_type
        state2domain_type      = spatial_data.domain_types.state2domain_type 
        
        self.spatial_data      = spatial_data
        self.state2domain_type = One2One('state', 'domain_type', state2domain_type) 
        self.element2idx       = One2One('voxel_state', 'idx')
        self.state2dxidx       = One2One('state', 'start_stop')
        self.state2domain_type = Many2One('state', 'domain_type', state2domain_type)
        
        #Call the parent constructor
        super().__init__(grid_config, shapes, domain_type2domain, surface2domain)
        
        self.state_code     = '\t#States\n'
        self.parameter_code = '\t#Parameters\n'
        self.function_code  = '\t#Functions\n'
        self.diff_code      = '\t#Diffs\n'
        
        self._add_state_code()
        self._add_parameter_code()
        self._add_function_code()
        
        #For code excution
        self.signature        = ['time', 'states', 'parameters']
        self.args             = {}
        self.rhs_functions    = {'__array'   : np.array,
                                 '__ndarray' : np.ndarray,
                                 '__ones'    : np.ones,
                                 '__zeros'   : np.zeros,
                                 '__float64' : np.float64
                                 }
        self.rhsdct_functions = self.rhs_functions.copy()
        
        #For plotting
        self.formatter = '{:.2f}'
    
    @property
    def rhsdef(self) -> str:
        signature = ', '.join(self.signature)
        ref       = self.spatial_data.ref
        
        return f'def model_{ref}({signature}):'
    
    @property
    def rhsdctdef(self) -> str:
        signature = ', '.join(self.signature)
        ref       = self.spatial_data.ref
        
        return f'def modeldct_{ref}({signature}):'
        
    def _add_state_code(self) -> None:
        spatial_data      = self.spatial_data
        voxel2domain_type = self.voxel2domain_type
        domain_type2voxel = voxel2domain_type.inverse
        state2dxidx       = self.state2dxidx
        state2domain_type = self.state2domain_type
        idx               = 0
        
        for state in spatial_data.states:
            domain_type = state2domain_type[state]
            voxels      = domain_type2voxel[domain_type]
            n_voxels    = len(voxels)
            
            #Get indices
            stop               = idx + n_voxels 
            state2dxidx[state] = idx, stop
            
            #Update code
            state_           = ut.undot(state)
            self.state_code += f'\t{state_} = states[{idx}:{stop}]\n'
            self.diff_code  += f'\t{ut.diff(state_)} = __zeros({state}.shape, __float64)\n'
            
            #Update idx
            idx = stop
    
    def _add_parameter_code(self) -> None:
        spatial_data = self.spatial_data
        parameters   = spatial_data.parameters
        
        for i, parameter in enumerate(parameters):
            parameter_ = ut.undot(parameter)
            self.parameter_code += f'\t{parameter_} = parameters[{i}]\n'
    
    def _add_function_code(self) -> None:
        spatial_data = self.spatial_data
        functions    = spatial_data.functions
        
        for function in functions.values():
            signature = ', '.join(function.signature)
            name      = function.name
            expr      = function.expr
            chunk     = f'\tdef {name}({signature}):\n\t\treturn {expr}\n'
            
            self.function_code += chunk
    
    def _add_voxel(self, voxel) -> None:
        super()._add_voxel(voxel)
        domain_type       = self.voxel2domain_type[voxel]
        element2idx       = self.element2idx
        datum             = self.voxels[voxel]
        datum['states']   = One2One('state', 'idx')
        domain_type2state = self.state2domain_type.inverse
        states            = domain_type2state[domain_type]
        
        for state in states:
            idx                       = len(element2idx)
            element2idx[voxel, state] = idx
            datum['states'][state]    = idx    
    
    ###########################################################################
    #Retrieval
    ###########################################################################
    def get_bulk_idx(self, point: tuple[Number]) -> int:
        point = tuple(point)
        try:
            idx = self.voxel2domain_type_idx[point][0]
        except KeyError:
            point_ = self.voxelize(point)
            idx    = self.voxel2domain_type_idx[point_][0]
        except Exception as e:
            raise e
        
        return idx
    
    def get_state_from_array(self, 
                             state_name : str, 
                             array      : np.ndarray
                             ) -> np.ndarray:
        
        start, stop = self.state2dxidx[state_name]
        
        return array[start: stop]
            
    ###########################################################################
    #Conversion of State Input
    ###########################################################################
    def expand_init(self, init_array: np.ndarray) -> np.ndarray:
        '''Reshapes row from DataFrame into array for rhs.
        '''
        
        state2dxidx = self.state2dxidx
        
        lst = []
        for state, value in zip(self.spatial_data.states, init_array):
            start, stop  = state2dxidx[state]
            lst         += [value]* (stop-start)
        
        array = np.array(lst)
        
        return array
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def _make_cmap_and_norm(self, values: np.ndarray, cmap='coolwarm') -> tuple:
        #Find the bounds and normalize
        lb   = min(values)
        ub   = max(values)
        norm = mpl.colors.Normalize(lb, ub)
        
        #Extract the cmap from matplotlib if a string was provided
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        
        return cmap, norm
    
    def _parse_cmap(self, values, cmap, norm) -> tuple:
        lb    = np.min(values)
        ub    = np.max(values)
        if callable(cmap):
            cmap_ = cmap
            norm_ = norm
            
            if not callable(norm):
                msg = 'If cmap is callable, norm must also be callable.'
                raise ValueError(msg)
        else:
            cmap_ = plt.get_cmap(cmap)
            
            if norm == 'linear':
                norm_  = mpl.colors.Normalize(lb, ub)
            elif norm == 'lognorm':
                norm_  = mpl.colors.LogNorm(lb, ub)
            else:
                msg = f'norm argument must be "linear" or "log". Received {norm}.'
                raise ValueError(msg)
            
            return cmap_, norm_, lb, ub
    
    def _plot_bulk(self,
                   ax             : plt.Axes,
                   name           : str,
                   values         : np.ndarray,
                   domain_type    : Domain_type,
                   cmap           : str='coolwarm',
                   norm           : Literal['linear', 'log']='linear',
                   label_voxels   : bool=True,
                   colorbar_ax    : plt.Axes=None
                   ) -> None:
        #Create the cache for the results
        results = {}
        
        #Parse the cmap and norm
        cmap_, norm_, *_ = self._parse_cmap(values, cmap, norm)
        
        #Set up the mappings
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes
        voxels                = self.voxel2domain_type.inverse[domain_type]
        formatter             = self.formatter
        
        #Iterate and plot
        for voxel in voxels:
            size            = sizes[voxel]
            domain_type_idx = voxel2domain_type_idx[voxel][0]
            value           = values[domain_type_idx]
            
            #Determine the arguments for the shape
            facecolor = cmap_(norm_(value))
            
            #Create the patch
            s      = size/2
            anchor = [i-s for i in voxel]
            patch  = Rectangle(anchor, size, size, facecolor=facecolor)
                
            #Plot the patch
            temp = ax.add_patch(patch)
            results[voxel] = temp
            
            #Add text
            if label_voxels:
                ax.text(*voxel, 
                        formatter.format(value), 
                        horizontalalignment='center'
                        )
                
        #Plot the color bar if applicable
        if colorbar_ax:
            mpl.colorbar.Colorbar(ax   = colorbar_ax, 
                                  cmap = cmap, 
                                  norm = norm_
                                  )   
        return results
    
    def plot_state(self, 
                   ax, 
                   state_name   : str, 
                   state_values : np.ndarray, 
                   **kwargs
                   )-> dict:
        
        if state_name not in self.state2domain_type:
            raise ValueError(f'No state named {state_name}.')
        
        domain_type    = self.state2domain_type[state_name]
        
        return self._plot_bulk(ax, 
                               state_name, 
                               state_values, 
                               domain_type, 
                               **kwargs
                               )
    
    def plot_diff(self, 
                   ax, 
                   state_name   : str, 
                   diff_values  : np.ndarray, 
                   **kwargs
                   )-> dict:
        
        return self.plot_state(ax, 
                               state_name, 
                               diff_values, 
                               **kwargs
                               )
    
        