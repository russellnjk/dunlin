import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
from matplotlib.patches import Rectangle
from numbers            import Number
from typing             import Union

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
    default_state_args = {'edgecolor': 'None'
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
            self.diff_code  += f'\t{ut.diff(state)} = __zeros({n_voxels}).astype(__float64)\n'
            
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
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def make_scaled_cmap(self, **name2cmap_and_values) -> tuple:
        '''
        Each state, parameter, reaction etc. can have a unique range of values. 
        This means that each of them requires a color map scaled to their range 
        of values. This method returns a callable of the form `func(name, value)` 
        that returns a color given the name of the item to be plotted and its value.
        
        To create the scaled color maps, this method applies `matplotlib.colors.Normalize` 
        to (1) a Matplotlib `cmap` (2) the maximum and minimum values for each item.

        Parameters
        ----------
        **name2cmap_and_values : dict
            A dictionary where the keys are names of states, parameters, variable etc.
            The values are dictionaries with the following mappings:
                1. `cmap` : A string corresponding to a Matplotlib colormap OR an actual 
                colormap object.
                2. 'values' : An iterable of two or more numbers. This method 
                searches for the maximum and minimum values to determine the 
                lower and upper bounds for scaling the color map.
            
            An example would be:
                `{'state0: {'cmap': 'coolwarm', 'values': [0, 1, 2, 3]}}`
             
        Returns
        -------
        norms : dict
            A dictionary with the same keys as `name2cmap_and_values`. The values 
            are `matplotlib.colors.Normalize` objects which normalize raw values. 
            
        func : callable
            A function with signature `func(name, value)`. When called, it searches 
            for a scaled color map indexed under `name` and returns the color for 
            `value` according to that `cmap`. The scaled color map is in turn, a 
            callable with the signature `scaled_cmap(value)` and wraps the 
            `matplotlib.colors.Normalize` object in `norms`.
        
        '''
        #For caching the results 
        dct   = {}
        norms = {}
        cmaps = {}
        
        for name, cmap_and_values in name2cmap_and_values.items():
            #Extract the arguments
            cmap   = cmap_and_values['cmap']
            values = cmap_and_values['values']
            
            sub_func = self._make_scaled_cmap_helper(values, cmap)
            
            dct[name]   = sub_func
            norms[name] = sub_func.norm
            cmaps[name] = sub_func.cmap
            
        #Combine into a single callable
        def func(name, value):
            try:
                sub_func = dct[name]
            except KeyError as e:
                if '_default' in dct:
                    sub_func = dct['_default']
                else:
                    raise e
            except Exception as e:
                raise e
            
            return sub_func(value)
        
        
        func.norms = norms
        func.cmaps = cmaps
        
        return func
    
    def _make_scaled_cmap_helper(self, values: np.array, cmap='coolwarm') -> callable:
        #Find the bounds and normalize
        lb   = min(values)
        ub   = max(values)
        norm = mpl.colors.Normalize(lb, ub)
        
        #Extract the cmap from matplotlib if a string was provided
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        
        #Make into a function
        func      = lambda value: cmap(norm(value))
        func.norm = norm
        func.cmap = cmap
        
        return func
    
    def _plot_bulk(self,
                   ax,
                   name           : str,
                   values         : np.array,
                   domain_type    : Domain_type,
                   default_kwargs : dict,
                   user_kwargs    : dict=None,
                   label_voxels   : bool=True
                   ) -> None:
        results    = {}
        converters = {'facecolor'   : upp.get_color, 
                      'surfacecolor': upp.get_color,
                      'color'       : upp.get_color
                      }
        
        voxel2domain_type_idx = self.voxel2domain_type_idx
        sizes                 = self.sizes
        voxels                = self.voxel2domain_type.inverse[domain_type]
        
        for voxel in voxels:
            size            = sizes[voxel]
            domain_type_idx = voxel2domain_type_idx[voxel][0]
            value           = values[domain_type_idx]
            
            #Determine the arguments for the shape
            sub_args            = {'name': name, 'value': value}
            bulk_reaction_args_ = upp.process_kwargs(user_kwargs,
                                                     [],
                                                     default_kwargs,
                                                     sub_args,
                                                     converters
                                                     )
            
            #Create the patch
            s      = size/2
            anchor = [i-s for i in voxel]
            patch  = Rectangle(anchor, size, size, **bulk_reaction_args_)
                
            #Plot the patch
            temp = ax.add_patch(patch)
            results[voxel] = temp
            
            #Add text
            if label_voxels:
                ax.text(*voxel, 
                        value, 
                        horizontalalignment='center'
                        )
            
        return results
    
    def plot_state(self, 
                   ax, 
                   state_name   : str, 
                   state_values : np.array, 
                   state_args   : dict = None,
                   label_voxels : bool = True
                   )-> dict:
        
        if state_name not in self.state2domain_type:
            raise ValueError(f'No state named {state_name}.')
        
        domain_type    = self.state2domain_type[state_name]
        default_kwargs = self.default_state_args
        
        return self._plot_bulk(ax, 
                               state_name, 
                               state_values, 
                               domain_type, 
                               default_kwargs, 
                               state_args,
                               label_voxels
                               )
    
    def plot_diff(self, 
                   ax, 
                   state_name   : str, 
                   diff_values  : np.array, 
                   diff_args    : dict = None,
                   label_voxels : bool = True
                   )-> dict:
        
        return self.plot_state(ax, 
                               state_name, 
                               diff_values, 
                               diff_args,
                               label_voxels
                               )
    
        