import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
from matplotlib.patches import Rectangle
from numbers            import Number
from typing             import Literal, Union

import dunlin.utils      as ut
import dunlin.utils_plot as upp
from .grid.grid            import RegularGrid, NestedGrid
from .grid.bidict          import One2One, One2Many
from .grid.domainstack     import (DomainStack,
                                   Domain_type, Domain, Voxel,
                                   AdjacentShapes, AdjacentDomains
                                   )
from .geometrydefinition   import make_shapes
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
    #For plotting
    functions = {'__array'   : np.array,
                 '__ones'    : np.ones,
                 '__zeros'   : np.zeros,
                 '__float64' : np.float64
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
    voxel2domain_type_idx : One2One[Voxel, tuple[int, Domain_type]]
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
    formatter         : str
    
    def __init__(self, spatial_data: SpatialModelData) -> None:
        self.spatial_data = spatial_data
        
        #Generate shapes and grids
        grid_config      = spatial_data['grid_config']
        shapes           = make_shapes(spatial_data['geometry_definitions'])
        domain_types     = spatial_data['domain_types']
        adjacent_domains = spatial_data['adjacent_domains']
        
       
        self.element2idx       = One2One('voxel_state', 'idx')
        self.state2dxidx       = One2One('state', 'start_stop')
        self.state2domain_type = One2Many('state', 
                                          'domain_type',
                                          spatial_data.compartments.state2domain_type
                                          )
        
        #Call the parent constructor
        super().__init__(grid_config, shapes, domain_types, adjacent_domains)
        
        self.state_code     = '\t#States\n'
        self.parameter_code = '\t#Parameters\n'
        self.function_code  = '\t#Functions\n'
        self.diff_code      = '\t#Diffs\n'
        
        self._add_state_code()
        self._add_parameter_code()
        self._add_function_code()
        
        self.signature = 'time', 'states', 'parameters'
        self.formatter = '{:.2f}'
    
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
            
            
            stop               = idx + n_voxels 
            state2dxidx[state] = idx, stop
            
            self.state_code += f'\t{state} = states[{idx}:{stop}]\n'
            self.diff_code  += f'\t{ut.diff(state)} = __zeros({state}.shape, __float64)\n'
            
            #Update idx
            idx = stop
    
    def _add_parameter_code(self) -> None:
        spatial_data = self.spatial_data
        parameters   = spatial_data.parameters
        
        for i, parameter in enumerate(parameters):
            self.parameter_code += f'\t{parameter} = parameters[{i}]\n'
    
    def _add_function_code(self) -> None:
        spatial_data = self.spatial_data
        functions    = spatial_data.functions
        
        for function in functions.values():
            signature = function.signature
            name      = function.name
            expr      = function.expr
            chunk     = f'\tdef {name}({signature}):\n\t\treturn {expr}\n'
            
            self.function_code += chunk
    
    def _add_voxel(self, voxel) -> None:
        super()._add_voxel(voxel)
        spatial_data    = self.spatial_data
        domain_type     = self.voxel2domain_type[voxel]
        element2idx     = self.element2idx
        datum           = self.voxels[voxel]
        datum['states'] = One2One('state', 'idx')
        states          = spatial_data.compartments.domain_type2state[domain_type]
        
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
                             state_name: str, 
                             array: np.array
                             ) -> np.array:
        
        start, stop = self.state2dxidx[state_name]
        
        return array[start: stop]
            
    ###########################################################################
    #Conversion of State Input
    ###########################################################################
    def reshape(self, init_array: np.array) -> np.array:
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
    def _make_cmap_and_norm(self, values: np.array, cmap='coolwarm') -> tuple:
        #Find the bounds and normalize
        lb   = min(values)
        ub   = max(values)
        norm = mpl.colors.Normalize(lb, ub)
        
        #Extract the cmap from matplotlib if a string was provided
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        
        return cmap, norm
    
    def _plot_bulk(self,
                   ax             : plt.Axes,
                   name           : str,
                   values         : np.array,
                   domain_type    : Domain_type,
                   cmap           : str='coolwarm',
                   norm           : Literal['linear', 'log']='linear',
                   label_voxels   : bool=True,
                   colorbar_ax    : plt.Axes=None
                   ) -> None:
        
        results = {}
        
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
        
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes
        voxels                = self.voxel2domain_type.inverse[domain_type]
        formatter             = self.formatter
        
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
        
        if colorbar_ax:
            mpl.colorbar.Colorbar(ax   = colorbar_ax, 
                                  cmap = cmap, 
                                  norm = norm_
                                  )   
        return results
    
    def plot_state(self, 
                   ax, 
                   state_name   : str, 
                   state_values : np.array, 
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
                   diff_values  : np.array, 
                   **kwargs
                   )-> dict:
        
        return self.plot_state(ax, 
                               state_name, 
                               diff_values, 
                               **kwargs
                               )
    
        