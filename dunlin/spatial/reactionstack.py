import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import re
import warnings
from matplotlib.patches import Rectangle
from numbers import Number
from scipy   import spatial
from typing  import Literal, Union

import dunlin.utils      as ut
import dunlin.utils_plot as upp
from .grid.grid            import RegularGrid, NestedGrid
from .grid.bidict          import One2One, One2Many
from .statestack           import (StateStack,
                                   Domain_type, Domain, Voxel, 
                                   AdjacentShapes, AdjacentDomains,
                                   State, Parameter,
                                   )
from dunlin.datastructures import SpatialModelData

#Typing
Surface = tuple[Domain_type, Domain_type]

#ReactionStack
class ReactionStack(StateStack):
    #For plotting
    surface_linewidth = 5
    
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
    
    def __init__(self, spatial_data: SpatialModelData):
        #Data structures for self._add_surface
        self.surface2domain_type_idx  = {}
        self.surfacepoint2surface_idx = One2One('surfacepoint', 'surface_idx/surface')
        
        #Call the parent constructor
        super().__init__(spatial_data)
        
        #Data structures for retrieval by location
        self.surfacepoint_lst = list(self.surfacepoint2surface_idx)
        self.surface2tree     = spatial.KDTree(self.surfacepoint_lst)
        
        #Parse the variables
        self.global_variables  = set()
        self.bulk_variables    = {}
        self.surface_variables = {}
        self.variable_code     = '\t#Variables'
        self._add_variable_code()
        
        #Parse the reactions
        self.bulk_reactions    = {}
        self.surface_reactions = {}
        self.reaction_code     = '\t#Reactions\n'
        self._add_reaction_code()
        
    ###########################################################################
    #Preprocessing
    ###########################################################################
    def _add_surface(self, voxel0, voxel1, shift) -> None:
        voxel2domain_type        = self.voxel2domain_type
        voxel2domain_type_idx    = self.voxel2domain_type_idx
        surface2domain_type_idx  = self.surface2domain_type_idx
        surfacepoint2surface_idx = self.surfacepoint2surface_idx
        
        #Extract domain_type information
        domain_type0 = voxel2domain_type[voxel0]
        domain_type1 = voxel2domain_type[voxel1]
        domain_types = domain_type0, domain_type1
        surface      = tuple(sorted([domain_type0, domain_type1]))
        
        #Prevent double copmutation
        if surface != domain_types:
            return
    
        #Update self.surface2domain_type_idx
        #When assigning the value of a surface variable/reaction
        #Given the domain types of each state/bulk variable, 
        #we can find the relevant indices
        voxel0_idx = voxel2domain_type_idx[voxel0][0]
        voxel1_idx = voxel2domain_type_idx[voxel1][0]
        
        default = {domain_type0: One2Many('surface_idx', 'domain_type_idx'), 
                   domain_type1: One2Many('surface_idx', 'domain_type_idx')
                   }
        
        dct = surface2domain_type_idx.setdefault(surface,
                                                 default
                                                 )
        
        surface_idx                    = len(dct[domain_type0])
        dct[domain_type0][surface_idx] = voxel0_idx
        dct[domain_type1][surface_idx] = voxel1_idx
        
        #Update self.surfacepoint2surface_idx
        #Given a location, we can find the index of interest
        surfacepoint                           = np.mean([voxel0, voxel1], axis=0)
        surfacepoint                           = tuple(surfacepoint)
        surfacepoint2surface_idx[surfacepoint] = surface, surface_idx
    
    ###########################################################################
    #Utils
    ###########################################################################
    def sub(self, expr, **kwargs):
        if ut.isstrlike(expr):
            repl = lambda match: self.repl(match, **kwargs)
            return re.sub('[a-zA-z]\w*', repl, expr)  
        else:
            return expr
    
    def repl(self, match):
        state2domain_type = self.state2domain_type
        bulk_variables    = self.bulk_variables
        
        if match[0] in state2domain_type:
            state       = match[0]
            domain_type = state2domain_type[state]
            
            return state + f'[__array({{{domain_type}}})]'
        
        elif match[0] in bulk_variables:
            variable    = match[0]
            domain_type = bulk_variables[variable]
            
            return variable + f'[__array({{{domain_type}}})]'
        
        else:
            return match[0]
        
    ###########################################################################
    #Variables
    ###########################################################################
    def _add_variable_code(self) -> None:
        spatial_data         = self.spatial_data
        variables            = spatial_data.variables
        state2domain_type    = self.state2domain_type
        states               = set(spatial_data.states.keys())
        global_variables     = self.global_variables
        bulk_variables       = self.bulk_variables
        surface_variables    = self.surface_variables
        
        for variable in variables.values():
            self.variable_code += f'\t#{variable.name}\n'
            domain_types        = set()
            
            #Figure out which states/variables are involved
            for name in variable.namespace:
                #Case 1: name is a state
                if name in states:
                    domain_type           = state2domain_type[name]
                    domain_types.add(domain_type)
                    
                #Case 2: name is a bulk variable
                elif name in bulk_variables:
                    domain_type = bulk_variables[name]
                    domain_types.add(domain_type)
                    
                #Case 3: surface variable
                elif name in surface_variables:
                    surface = surface_variables[name]
                    domain_types.update(surface)
                    
            #Update variable mappings
            #Ensure that the number of domain types is at most 2
            key = variable.name
            if len(domain_types) == 0:
                global_variables.add(key)
                self._add_global_variable_code(variable)
            elif len(domain_types) == 1:
                domain_type         = tuple(domain_types)[0]
                bulk_variables[key] = domain_type
                self._add_bulk_variable_code(variable)
            elif len(domain_types) == 2:
                surface                = tuple(sorted(domain_types))
                surface_variables[key] = surface
                self._add_surface_variable_code(variable, surface)
            else:
                msg = f'Variable {variable.name} has more than 2 domain_types: {domain_types}.'
                raise ValueError(msg)
    
    def _add_global_variable_code(self, variable) -> None:
        #Update variable code
        self.variable_code += f'\t{variable.name} = {variable.expr}\n\n'
    
    def _add_bulk_variable_code(self, variable) -> None:
        return self._add_global_variable_code(variable)
    
    def _add_surface_variable_code(self, variable, surface) -> None:
        surface2domain_type_idx = self.surface2domain_type_idx
        
        expr          = variable.expr
        lhs           = variable.name
        rhs           = self.sub(expr)
        surface_data  = surface2domain_type_idx[surface]
        surface_data_ = {k: list(v.values()) for k, v in surface_data.items()}
        rhs           = rhs.format(**surface_data_)
        variable_code = f'\t{lhs} = {rhs}\n'
        
        self.variable_code += variable_code + '\n'
    
    ###########################################################################
    #Reactions
    ###########################################################################
    def _add_reaction_code(self) -> None:
        spatial_data      = self.spatial_data
        reactions         = spatial_data.reactions
        state2domain_type    = self.state2domain_type
        states               = set(spatial_data.states.keys())
        bulk_variables       = self.bulk_variables
        surface_variables    = self.surface_variables
        bulk_reactions       = self.bulk_reactions
        surface_reactions    = self.surface_reactions
        
        for reaction in reactions.values():
            self.reaction_code += f'\t#{reaction.name}\n'
            domain_types        = set()
            
            #Figure out which states/variables are involved
            for name in reaction.namespace:
                #Case 1: name is a state
                if name in states:
                    domain_type           = state2domain_type[name]
                    domain_types.add(domain_type)
                    
                #Case 2: name is a bulk variable
                elif name in bulk_variables:
                    domain_type = bulk_variables[name]
                    domain_types.add(domain_type)
                    
                #Case 3: surface variable
                elif name in surface_variables:
                    surface = surface_variables[name]
                    domain_types.update(surface)
                    
            #Update variable mappings
            #Ensure that the number of domain types is at most 2
            key = reaction.name
            if len(domain_types) == 1:
                domain_type         = tuple(domain_types)[0]
                bulk_reactions[key] = domain_type
                self._add_bulk_reaction_code(reaction)
            elif len(domain_types) == 2:
                surface                = tuple(sorted(domain_types))
                surface_reactions[key] = surface
                self._add_surface_reaction_code(reaction, surface)
            else:
                msg = f'Reaction {reaction.name} has 0 or more than 2 domain_types: {domain_types}.'
                raise ValueError(msg)
    
        
    def _add_bulk_reaction_code(self, reaction) -> None:
        #Update reaction code
        self.reaction_code += f'\t{reaction.name} = {reaction.rate}\n\n'
        
        #Add the diff code
        name      = reaction.name
        diff_code = ''
        for state, stoich in reaction.stoichiometry.items():
            lhs   = ut.diff(state)
            rhs   = f'{stoich}*{name}' 
            chunk = f'\t{lhs} += {rhs}\n'
            
            diff_code += chunk
        
        self.reaction_code += diff_code + '\n'
        
    def _add_surface_reaction_code(self, reaction, surface) -> None:
        surface2domain_type_idx = self.surface2domain_type_idx
        state2domain_type       = self.state2domain_type
        
        expr          = reaction.rate
        lhs           = reaction.name
        rhs           = self.sub(expr)
        surface_data  = surface2domain_type_idx[surface]
        surface_data_ = {k: list(v.values()) for k, v in surface_data.items()}
        rhs           = rhs.format(**surface_data_)
        reaction_code = f'\t{lhs} = {rhs}\n\n'
        
        self.reaction_code += reaction_code 
        
        #Update diff code
        for state, stoich in reaction.stoichiometry.items():
            domain_type           = state2domain_type[state]
            state_idx2surface_idx = surface_data[domain_type].inverse
            
            state_idxs  = list(state_idx2surface_idx)
            lhs         = ut.diff(state) + f'[__array({state_idxs})]'
            
            rhs = f'{stoich}*__array(['
            for surface_idxs in state_idx2surface_idx.values():
                rhs_element = [f'{reaction.name}[{idx}]' for idx in surface_idxs]
                rhs_element = '+'.join(rhs_element)
                rhs_element = f'{rhs_element}, '
                
                rhs += rhs_element
            
            rhs += '])'
            
            diff_code   = f'\t{lhs} += {rhs}\n' 
            self.reaction_code += diff_code + '\n'
         
    ###########################################################################
    #Retrieval
    ###########################################################################
    def get_surface_idx(self, 
                        point          : tuple[Number], 
                        return_surface : bool=False
                        ) -> int:
        point = tuple(point)
        try:
            surface, idx  = self.surfacepoint2surface_idx[point]
            dist          = 0
        except KeyError:
            dist, temp   = self.surface2tree.query(point)
            point_       = self.surfacepoint_lst[temp]
            surface, idx = self.surfacepoints[point_]
        except Exception as e:
            raise e
        
        if dist > self.grid.step:
            msg  = 'Attempted to find closest point to {point} for {name}. '
            msg += 'The resulting closest point is further than the step size of the largest grid.'
            warnings.warn(msg)
        
        if return_surface:
            return surface, idx
        else:
            return idx
    
    def get_surface(self, 
                    surface     : Surface, 
                    domain_type : Domain_type=None,
                    ):
        dct = self.surface2domain_type_idx[surface]
        
        if domain_type is None:
            return dct
        else:
            return dct[domain_type]
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def _plot_surface(self, 
                      ax, 
                      name           : str, 
                      values         : np.array,
                      surface        : Surface,
                      default_kwargs : dict,
                      user_kwargs    : dict = None,
                      label_surfaces : bool = True
                      ) -> dict:
        results    = {}
        converters = {'facecolor'   : upp.get_color, 
                      'surfacecolor': upp.get_color,
                      'color'       : upp.get_color
                      }
        
        domain_type_idx2voxel         = self.voxel2domain_type_idx.inverse
        sizes                         = self.sizes
        voxels                        = self.voxels
        
        surface_data = self.surface2domain_type_idx[surface]
        
        domain_type0, domain_type1 = surface
        
        d         = self.surfacepoint2surface_idx.inverse 
        plot_data = []
        
        for surface_idx, value in enumerate(values):
            surfacepoint = d[surface, surface_idx]
            
            plot_data.append([*surfacepoint, value])
            
            domain_type_idx0 = surface_data[domain_type0][surface_idx]
            domain_type_idx1 = surface_data[domain_type1][surface_idx]
            
            voxel0 = domain_type_idx2voxel[domain_type_idx0, domain_type0]
            voxel1 = domain_type_idx2voxel[domain_type_idx1, domain_type1]
            
            size0 = sizes[voxel0]
            size1 = sizes[voxel1]
            
            if size0 > size1:
                voxel, neighbour = voxel1, voxel0
                neighbour_domain_type = domain_type0
            else:
                voxel, neighbour = voxel0, voxel1
                neighbour_domain_type = domain_type1
                
            #Determine the arguments for the surface
            sub_args       = {'name': name, 'value': value}
            reaction_args_ = upp.process_kwargs(user_kwargs,
                                                [],
                                                default_kwargs,
                                                sub_args,
                                                converters
                                                )
            
            #Create the line
            shift       = voxels[voxel]['surface'][neighbour_domain_type][neighbour]
            delta       = np.sign(shift)*size0/2
            idx         = abs(shift)-1
            point       = np.array(voxel)
            point[idx] += delta
            start       = np.array(voxel)
            stop        = np.array(voxel)
            
            for i, n in enumerate(voxel):
                if i == idx:
                    start[i] += delta
                    stop[i]  += delta
                else:
                    start[i] -= size0/2
                    stop[i]  += size0/2
            
            x, y = np.stack([start, stop]).T
            line = ax.plot(x, y, **reaction_args_)
            
            results[surface_idx] = line
            
            #Add text
            if label_surfaces:
                ax.text(*np.mean([start, stop], axis=0), 
                        value, 
                        horizontalalignment='center'
                        )
    
        return results
    
    def _plot_surface(self, 
                      ax, 
                      name           : str, 
                      values         : np.array,
                      surface        : Surface,
                      cmap           : str='coolwarm',
                      norm           : Literal['linear', 'log']='linear',
                      label_surfaces : bool=True,
                      colorbar_ax    : plt.Axes=None
                      ) -> dict:
        results    = {}
        
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
        
        domain_type_idx2voxel         = self.voxel2domain_type_idx.inverse
        sizes                         = self.sizes
        voxels                        = self.voxels
        
        surface_data = self.surface2domain_type_idx[surface]
        
        domain_type0, domain_type1 = surface
        
        d         = self.surfacepoint2surface_idx.inverse 
        plot_data = []
        
        surface_linewidth = self.surface_linewidth
        
        for surface_idx, value in enumerate(values):
            surfacepoint = d[surface, surface_idx]
            
            plot_data.append([*surfacepoint, value])
            
            domain_type_idx0 = surface_data[domain_type0][surface_idx]
            domain_type_idx1 = surface_data[domain_type1][surface_idx]
            
            voxel0 = domain_type_idx2voxel[domain_type_idx0, domain_type0]
            voxel1 = domain_type_idx2voxel[domain_type_idx1, domain_type1]
            
            size0 = sizes[voxel0]
            size1 = sizes[voxel1]
            
            if size0 > size1:
                voxel, neighbour = voxel1, voxel0
                neighbour_domain_type = domain_type0
            else:
                voxel, neighbour = voxel0, voxel1
                neighbour_domain_type = domain_type1
            
            #Determine the arguments for the
            color = cmap_(norm_(value))
            
            #Create the line
            shift       = voxels[voxel]['surface'][neighbour_domain_type][neighbour]
            delta       = np.sign(shift)*size0/2
            idx         = abs(shift)-1
            point       = np.array(voxel)
            point[idx] += delta
            start       = np.array(voxel)
            stop        = np.array(voxel)
            
            for i, n in enumerate(voxel):
                if i == idx:
                    start[i] += delta
                    stop[i]  += delta
                else:
                    start[i] -= size0/2
                    stop[i]  += size0/2
            
            x, y = np.stack([start, stop]).T
            line = ax.plot(x, y, linewidth=surface_linewidth, color=color)
            
            results[surface_idx] = line
            
            #Add text
            if label_surfaces:
                ax.text(*np.mean([start, stop], axis=0), 
                        value, 
                        horizontalalignment='center'
                        )
        
        if colorbar_ax:
            mpl.colorbar.Colorbar(ax   = colorbar_ax, 
                                  cmap = cmap, 
                                  norm = norm_
                                  )   
            
        return results
    
    def _plot_global(self,
                     ax,
                     name           : str,
                     values         : np.array,
                     default_kwargs : dict,
                     user_kwargs    : dict=None,
                     ) -> None:
        results    = {}
        converters = {'facecolor'   : upp.get_color, 
                      'surfacecolor': upp.get_color,
                      'color'       : upp.get_color
                      }
        
        sizes  = self.sizes
        voxels = self.voxels
        
        for voxel, value in zip(voxels, values):
            size = sizes[voxel]
            
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
            
        return results
    
    def plot_reaction(self, 
                      ax, 
                      reaction_name   : str, 
                      reaction_values : np.array,
                      **kwargs
                      ) -> dict:
        
        if reaction_name in self.bulk_reactions:
            return self._plot_bulk_reaction(ax, reaction_name, reaction_values, **kwargs)
    
        elif reaction_name in self.surface_reactions:
            return self._plot_surface_reaction(ax, reaction_name, reaction_values, **kwargs)
        
        else:
            raise ValueError(f'No reaction named {reaction_name}.')
    
    def _plot_bulk_reaction(self,
                           ax,
                           reaction_name   : str, 
                           reaction_values : np.array,
                           **kwargs
                           ) -> dict:
        domain_type = self.bulk_reactions[reaction_name]
        
        return self._plot_bulk(ax, 
                               reaction_name, 
                               reaction_values, 
                               domain_type, 
                               **kwargs
                               )

    def _plot_surface_reaction(self, 
                              ax, 
                              reaction_name   : str, 
                              reaction_values : np.array,
                              **kwargs
                              ) -> dict:
        
        surface = self.surface_reactions[reaction_name]
        
        return self._plot_surface(ax, 
                                  reaction_name, 
                                  reaction_values, 
                                  surface, 
                                  **kwargs
                                  )
    
    def plot_variable(self, 
                      ax              : plt.Axes, 
                      variable_name   : str,
                      variable_values : np.array,
                      **kwargs
                      ) -> dict:
        if variable_name in self.bulk_variables:
            return self._plot_bulk_variable(ax, variable_name, variable_values, **kwargs)
        elif variable_name in self.surface_variables:
            return self._plot_surface_variable(ax, variable_name, variable_values, **kwargs)
        elif variable_name in self.global_variable:
            return self._plot_global_variable(ax, variable_name, variable_values, **kwargs)
        else:
            raise ValueError(f'No variable named {variable_name}.')
    
    def _plot_global_variable(self,
                              ax,
                              variable_name   : str, 
                              variable_values : np.array,
                              **kwargs
                              ) -> dict:
        
        return self._plot_global(ax, 
                                 variable_name, 
                                 variable_values, 
                                 **kwargs
                                 )
    
    def _plot_bulk_variable(self,
                            ax,
                            variable_name   : str, 
                            variable_values : np.array,
                            **kwargs
                            ) -> dict:
        domain_type = self.bulk_variables[variable_name]
        
        return self._plot_bulk(ax, 
                               variable_name, 
                               variable_values, 
                               domain_type, 
                               **kwargs
                               )
    
    def _plot_surface_variable(self, 
                              ax, 
                              variable_name   : str, 
                              variable_values : np.array,
                              **kwargs
                              ) -> dict:
        
        surface = self.surface_variables[variable_name]
        
        return self._plot_surface(ax, 
                                  variable_name, 
                                  variable_values, 
                                  surface, 
                                  **kwargs
                                  )
    
    