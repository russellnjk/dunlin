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
    rhs_functions     : dict[str, callable]
    rhsdct_functions  : dict[str, callable]
    formatter         : str
    
    surface_data      : dict[Surface, dict]
    global_variables  : set
    bulk_variables    : dict[str, Domain_type]
    surface_variables : dict[str, Surface]
    variable_code     : str
    bulk_reactions    : dict[str, Domain_type]
    surface_reactions : dict[str, Surface]
    reaction_code     : str
    surface_linewidth : float
    
    def __init__(self, spatial_data: SpatialModelData):
        #Data structures for self._add_surface
        self.surface_data  = {}
        
        #Call the parent constructor
        super().__init__(spatial_data)
        self._reformat_surface_data()
        
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
        surface_data  = self.surface_data
        sizes                    = self.sizes
        ndims                    = self.ndims
        
        #Extract domain_type information
        domain_type0 = voxel2domain_type[voxel0]
        domain_type1 = voxel2domain_type[voxel1]
        domain_types = domain_type0, domain_type1
        surface      = tuple(sorted([domain_type0, domain_type1]))
        size0        = sizes[voxel0]
        size1        = sizes[voxel1]
        n            = ndims - 1
        
        #Prevent double computation
        if surface != domain_types:
            return
        
        #Update self.surface_data
        #When assigning the value of a surface variable/reaction
        #Given the domain types of each state/bulk variable, 
        #we can find the relevant indices
        voxel0_idx = voxel2domain_type_idx[voxel0][0]
        voxel1_idx = voxel2domain_type_idx[voxel1][0]
        
        default       = {'batch_data' : []}
        surface_datum = surface_data.setdefault(surface, default)
        batch_data    = surface_datum['batch_data']
        
        i = 0
        while True:
            if i == len(batch_data):
                batch_data.append({'idxs'    : {}, 
                                   'scales'  : [],
                                   'shifts'  : [],
                                   'mids'    : [],
                                   'volumes' : {}
                                   }
                                  )
            
            batch   = batch_data[i]
            idxs    = batch['idxs'  ]
            scales  = batch['scales']
            mids    = batch['mids'  ] 
            volumes = batch['volumes']
            shifts  = batch['shifts']
            
            if voxel0_idx in idxs:
                i += 1
                
            else:
                #Update idxs
                idxs[voxel0_idx] = voxel1_idx
                
                #Update scales
                A  = min(size0, size1)**n
                scales.append(A)
                
                #Update shifts
                shifts.append(shift)
                
                #Update midpoints
                midpoint = tuple(np.mean([voxel0, voxel1], axis=0))
                mids.append(midpoint)
                
                #Update volumes
                volumes.setdefault(domain_type0, []).append(size0**ndims)
                volumes.setdefault(domain_type1, []).append(size1**ndims)
                
                break
        
    def _reformat_surface_data(self) -> None:
        surface_data = self.surface_data
        
        start = 0
        for surface in surface_data.keys():
            domain_type0, domain_type1 = surface
            surface_datum              = surface_data[surface]  
            batches                    = surface_datum['batch_data']
            template                   = '__array({})'
            collated0                  = []
            collated1                  = []
            split                      = []
            surface_idx2midpoint       = {}
            domain_type_idxs           = []
            shifts                     = []
            
            for batch in batches:
                #Create code chunks corresponding to indices
                domain_type_idxs0, domain_type_idxs1 = zip(*batch['idxs'].items()) 
        
                stop         = start + len(domain_type_idxs0) 
                split_       = {domain_type0 : template.format(list(domain_type_idxs0)),
                                domain_type1 : template.format(list(domain_type_idxs1)),
                                '_scale'     : template.format(batch['scales']),
                                '_idxs'      : (start, stop),
                                '_volumes'   : {}
                                }
                
                split.append(split_)
                collated0.extend(domain_type_idxs0)
                collated1.extend(domain_type_idxs1)
                
                #Update surface_idx2midpoint
                midpoints             = batch['mids'] 
                surface_idx2midpoint_ = dict(enumerate(midpoints, start=start))
                surface_idx2midpoint.update(surface_idx2midpoint_)
                
                #Update surface_idx2domain_type_idx
                pairs = list(zip(domain_type_idxs0, domain_type_idxs1))
                
                domain_type_idxs.extend(pairs)
                
                #Update shifts
                shifts.extend(batch['shifts'])
                
                #Update volumes
                volumes0 = list(batch['volumes'][domain_type0])
                volumes1 = list(batch['volumes'][domain_type1])
                
                volumes0 = template.format(volumes0)
                volumes1 = template.format(volumes1)
                
                split_['_volumes'][domain_type0] = volumes0
                split_['_volumes'][domain_type1] = volumes1
                
                #Update start
                start = stop
            
            #Convert collated from list to string
            collated0   = template.format('[' + ', '.join([str(i) for i in collated0]) + ']')
            collated1   = template.format('[' + ', '.join([str(i) for i in collated1]) + ']')
            
            #Convert surface_idx2midpoint into One2One to allow 2 way access
            surface_idx2midpoint = One2One('surface_idx', 'midpoint', surface_idx2midpoint)
            
            #Update surface_datum
            #Add code chunks to surface_datum
            surface_datum['code'] = {'collated': {domain_type0 : collated0,
                                                  domain_type1 : collated1,
                                                  }, 
                                     'split'   : split,
                                     }
            
            #Add spatial data to surface_datum
            surface_datum['spatial'] = {'surface_idx2midpoint' : surface_idx2midpoint,
                                        'domain_type_idxs'     : domain_type_idxs,
                                        'shifts'               : shifts 
                                        }
            
            #Add a tree for searching
            midpoints             = list(surface_idx2midpoint.values())
            surface_datum['tree'] = spatial.KDTree(midpoints)
            
    ###########################################################################
    #Utils
    ###########################################################################
    def sub(self, expr):
        if ut.isstrlike(expr):
            repl = self.repl
            return re.sub('[a-zA-z]\w*', repl, expr)  
        else:
            return expr
    
    def repl(self, match):
        state2domain_type = self.state2domain_type
        bulk_variables    = self.bulk_variables
        
        if match[0] in state2domain_type:
            state       = match[0]
            domain_type = state2domain_type[state]
            
            return state + f'[{{{domain_type}}}]'
        
        elif match[0] in bulk_variables:
            variable    = match[0]
            domain_type = bulk_variables[variable]
            
            return variable + f'[{{{domain_type}}}]'
        
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
                    domain_type = state2domain_type[name]
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
        rhs = ut.undot(variable.expr)
        lhs = ut.undot(variable.name)
        
        self.variable_code += f'\t{lhs} = {rhs}\n\n'
    
    def _add_bulk_variable_code(self, variable) -> None:
        return self._add_global_variable_code(variable)
    
    def _add_surface_variable_code(self, variable, surface) -> None:
        surface_data = self.surface_data
        
        expr          = variable.expr
        lhs           = ut.undot(variable.name)
        rhs           = self.sub(expr)
        surface_datum = surface_data[surface]
        
        #Get a mapping that maps the domain_type of a state/bulk variable 
        #to indices in code form
        collated      = surface_datum['code']['collated']
        rhs           = ut.undot(rhs.format(**collated))
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
        name      = ut.undot(reaction.name)
        diff_code = ''
        for state, stoich in reaction.stoichiometry.items():
            lhs   = ut.diff(ut.undot(state))
            rhs   = f'{stoich}*{name}' 
            chunk = f'\t{lhs} += {rhs}\n'
            
            diff_code += chunk
        
        self.reaction_code += diff_code + '\n'
        
    def _add_surface_reaction_code(self, reaction, surface) -> None:
        surface_data      = self.surface_data
        state2domain_type = self.state2domain_type
        
        #Update reaction code
        expr          = reaction.rate
        lhs           = ut.undot(reaction.name)
        rhs           = self.sub(expr)
        surface_datum = surface_data[surface]
        
        #Get a mapping that maps the domain_type of a state/bulk variable 
        #to indices in code form
        collated      = surface_datum['code']['collated']
        rhs           = ut.undot(rhs.format(**collated))
        reaction_code = f'\t{lhs} = {rhs}\n\n'
        
        self.reaction_code += reaction_code 
        
        #Update diff code
        for state, stoich in reaction.stoichiometry.items():
            domain_type = state2domain_type[state]
            split       = surface_datum['code']['split']
            
            for batch in split:
                #Parse the lhs
                state_idxs  = batch[domain_type]
                lhs         = ut.diff(ut.undot(state)) + f'[{state_idxs}]'
                
                #Parse the rhs
                state_idxs = ':'.join([str(i) for i in batch['_idxs']])
                scale      = batch['_scale']
                volumes    = batch['_volumes'][domain_type]
                rhs        = f'{stoich}*{ut.undot(reaction.name)}[{state_idxs}]'
                rhs        = f'{rhs}*{scale}/{volumes}'
                
                #Update diff_code
                diff_code   = f'\t{lhs} += {rhs}\n' 
                
            self.reaction_code += diff_code + '\n'
         
    ###########################################################################
    #Retrieval
    ###########################################################################
    def get_surface_midpoints(self,
                              surface : Surface
                              ) -> list[tuple]:
        surface_idx2midpoint = self.surface_data[surface]['spatial']['surface_idx2midpoint']
        surface_midpoints    = list(surface_idx2midpoint.values())
        return surface_midpoints 
    
    def get_surface_idx(self, 
                        surface : Surface,
                        *points : tuple[Number], 
                        ) -> int:
        #Set up variables
        surface_datum        = self.surface_data
        surface_idx2midpoint = surface_datum[surface]['spatial']['surface_idx2midpoint']
        midpoint2surface_idx = surface_idx2midpoint.inverse
        surface_idxs         = []
        surface_midpoints    = []
        
        #Iterate and perform search for each point
        for point in points:
            point = tuple(point)
            try:
                idx      = midpoint2surface_idx[point]
                midpoint = point
                
            except KeyError:
                dist, idx = surface_datum['tree'].query(point)
                midpoint  = surface_idx2midpoint[idx]
                
                if dist > self.grid.step:
                    msg  = 'Attempted to find closest point to {point} for {name}. '
                    msg += 'The resulting closest point is further than the step size of the largest grid.'
                    warnings.warn(msg)
            except Exception as e:
                raise e
            
            #Update results
            surface_idxs.append(idx)
            surface_midpoints.append(midpoint)

        return surface_idxs, surface_midpoints
    
    ###########################################################################
    #Plotting
    ###########################################################################
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
        #Create the cache for the results
        results = {}
        
        #Parse the cmap and norm
        cmap_, norm_, *_ = self._parse_cmap(values, cmap, norm)
        
        #Set up the mappings
        domain_type_idx2voxel = self.voxel2domain_type_idx.inverse
        sizes                 = self.sizes
        surface_datum         = self.surface_data[surface]
        surface_idx2midpoint  = surface_datum['spatial']['surface_idx2midpoint']
        domain_type_idxs      = surface_datum['spatial']['domain_type_idxs']
        shifts                = surface_datum['spatial']['shifts']
        
        #Other overhead
        domain_type0, domain_type1 = surface
        surface_linewidth          = self.surface_linewidth
        
        for surface_idx, value in enumerate(values):
            #Determine the arguments for the
            color = cmap_(norm_(value))
            
            #Get the arguments for drawing the line
            domain_type_idx0, domain_type_idx1 = domain_type_idxs[surface_idx]
            
            midpoint   = surface_idx2midpoint[surface_idx]
            shift      = shifts[surface_idx]
            idx        = abs(shift)-1
            voxel0     = domain_type_idx2voxel[domain_type_idx0, domain_type0]
            voxel1     = domain_type_idx2voxel[domain_type_idx1, domain_type1]
            size0      = sizes[voxel0]
            size1      = sizes[voxel1]
            delta      = np.ones(len(midpoint)) * min(size0, size1)/2
            delta[idx] = 0
            start      = np.array(midpoint) - delta
            stop       = np.array(midpoint) + delta
            x, y       = np.stack([start, stop]).T
            
            #Plot the line
            line = ax.plot(x, y, linewidth=surface_linewidth, color=color)
            
            #Update the results
            results[surface_idx] = line
            
            #Add text
            if label_surfaces:
                ax.text(*np.mean([start, stop], axis=0), 
                        value, 
                        horizontalalignment='center'
                        )
        
        #Plot the color bar if applicable
        if colorbar_ax:
            mpl.colorbar.Colorbar(ax   = colorbar_ax, 
                                  cmap = cmap, 
                                  norm = norm_
                                  )   
            
        return results
    
    def _plot_global(self,
                     ax,
                     name         : str,
                     values       : np.array,
                     cmap         : str='coolwarm',
                     norm         : Literal['linear', 'log']='linear',
                     label_voxels : bool=True,
                     colorbar_ax  : plt.Axes=None
                     ) -> dict:
        #Create the cache for the results
        results = {}
        
        #Parse the cmap and norm
        cmap_, norm_, *_ = self._parse_cmap(values, cmap, norm)
        
        #Set up the mappings
        sizes     = self.sizes
        formatter = self.formatter
        voxels    = self.voxels
        
        for voxel, value in zip(voxels, values):
            size = sizes[voxel]
            
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
    
    