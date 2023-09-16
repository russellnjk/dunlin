import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import re
import warnings
from matplotlib.patches import Rectangle
from numba   import njit 
from numbers import Number
from scipy   import spatial
from typing  import Literal, Union

import dunlin.utils      as ut
from ..grid.grid  import RegularGrid, NestedGrid
from .bidict      import One2One, Many2One
from .statestack  import (StateStack,
                          Domain_type, Domain, Voxel, 
                          State, Parameter,
                          )
from dunlin.datastructures import SpatialModelData

#Typing
Surface_type = tuple[Domain_type, Domain_type]

@njit
def update_d_state(d_state, indices, stoich, reaction_rate, area, volume):
    for i in range(len(indices)):
        d_state[indices[i]] += stoich*reaction_rate[i]*area[i]/volume[i]
    
    return d_state

#ReactionStack
class ReactionStack(StateStack):
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
    
    def __init__(self, spatial_data: SpatialModelData):
        #Data structures for self._add_surface
        self.surface_data  = {}
        
        #Call the parent constructor
        super().__init__(spatial_data)
        self._reformat_surface_data()
        
        self.rhs_functions['__surface_reaction']    = update_d_state
        self.rhsdct_functions['__surface_reaction'] = update_d_state
        
        #For keeping track of code
        self.seen_code = set()
        
        #Parse the variables
        self.global_variables  = set()
        self.bulk_variables    = {}
        self.surface_variables = {}
        self.variable_code     = '\t#Variables\n'
        self._add_variable_code()
        
        #Parse the reactions
        self.bulk_reactions    = {}
        self.surface_reactions = {}
        self.reaction_code     = '\t#Reactions\n'
        self._add_reaction_code()
        
        # pattern = '[^[]__array\([^)]*\)'
        # repl    = lambda match: match[0] + '[:,__newaxis]'
        # edit    = lambda x: re.sub(pattern, repl, x)
        
        # self.reaction_code_rhsdct = edit(self.reaction_code)
        
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
        surface_type = tuple(sorted([domain_type0, domain_type1]))
        size0        = sizes[voxel0]
        size1        = sizes[voxel1]
        
        #Prevent double computation
        if surface_type != domain_types:
            return
        
        #Update self.surface_data
        #When assigning the value of a surface variable/reaction
        #Given the domain types of each state/bulk variable, 
        #we can find the relevant indices
        voxel0_idx = voxel2domain_type_idx[voxel0][0]
        voxel1_idx = voxel2domain_type_idx[voxel1][0]
        
        default       = {'mappings' : []}
        surface_datum = surface_data.setdefault(surface_type, default)
        
        A        = min(size0, size1)**(ndims - 1)
        midpoint = tuple(np.mean([voxel0, voxel1], axis=0))
        volume0  = size0**ndims
        volume1  = size1**ndims
        datum    = (voxel0_idx, 
                    voxel1_idx, 
                    A, 
                    shift, 
                    volume0, 
                    volume1,
                    *midpoint, 
                    )
        #Note: Use a tuple, not a list. This is required for structured arrays.
        
        surface_datum['mappings'].append(datum)
        
    def _reformat_surface_data(self) -> None:
        surface_data = self.surface_data
        
        
        #Functions for code chunks
        array_template ='__array([{}])'
        make_arr       = lambda x: array_template.format( ', '.join([str(i) for i in x]) )
        
        for surface_type in surface_data.keys():
            domain_type0, domain_type1 = surface_type
            surface_datum              = surface_data[surface_type]  
            mappings                   = surface_datum['mappings']
            
            #Convert the list of tuples into a structured array
            #This allows tabular access
            xyz = [('x', np.float64), 
                   ('y', np.float64), 
                   ('z', np.float64)
                   ]
            
            dtype    = [('voxel0_idx', np.int64),
                        ('voxel1_idx', np.int64),
                        ('A',          np.float64),
                        ('shift',      np.int8),
                        ('volume0',    np.float64),
                        ('volume1',    np.float64),
                        *xyz[:self.ndims]
                        ]
            
            mappings                  = np.array(mappings,
                                                 dtype=dtype
                                                 )
            surface_datum['mappings'] = mappings
            
            #Make code chunks
            code = {}
            
            code[f'__{domain_type0}2surface_idx'] = make_arr(mappings['voxel0_idx']) 
            code[f'__{domain_type1}2surface_idx'] = make_arr(mappings['voxel1_idx']) 
            code['A']                             = make_arr(mappings['A'])
            code['volume0']                       = make_arr(mappings['volume0'])
            code['volume1']                       = make_arr(mappings['volume1'])
            
            surface_datum['code'] = code
            
            #Convert surface_idx2midpoint into One2One to allow 2 way access
            midpoints            = mappings[['x', 'y', 'z'][:self.ndims]]
            surface_idx2midpoint = {i: tuple(p) for i, p in enumerate(midpoints)}
            surface_idx2midpoint = One2One('surface_idx', 'midpoint', surface_idx2midpoint)
            
            surface_datum['surface_idx2midpoint'] = surface_idx2midpoint
            
        # start = 0
        # for surface_type in surface_data.keys():
        #     domain_type0, domain_type1 = surface_type
        #     surface_datum              = surface_data[surface_type]  
        #     batches                    = surface_datum['batch_data']
        #     template                   = '__array({})'
        #     collated0                  = []
        #     collated1                  = []
        #     split                      = []
        #     surface_idx2midpoint       = {}
        #     domain_type_idxs           = []
        #     shifts                     = []
            
        #     for batch in batches:
        #         #Create code chunks corresponding to indices
        #         domain_type_idxs0, domain_type_idxs1 = zip(*batch['idxs'].items()) 
        
        #         stop         = start + len(domain_type_idxs0) 
        #         split_       = {domain_type0 : template.format(list(domain_type_idxs0)),
        #                         domain_type1 : template.format(list(domain_type_idxs1)),
        #                         '_scale'     : template.format(batch['scales']),
        #                         '_idxs'      : (start, stop),
        #                         '_volumes'   : {}
        #                         }
                
        #         split.append(split_)
        #         collated0.extend(domain_type_idxs0)
        #         collated1.extend(domain_type_idxs1)
                
        #         #Update surface_idx2midpoint
        #         midpoints             = batch['mids'] 
        #         surface_idx2midpoint_ = dict(enumerate(midpoints, start=start))
        #         surface_idx2midpoint.update(surface_idx2midpoint_)
                
        #         #Update surface_idx2domain_type_idx
        #         pairs = list(zip(domain_type_idxs0, domain_type_idxs1))
                
        #         domain_type_idxs.extend(pairs)
                
        #         #Update shifts
        #         shifts.extend(batch['shifts'])
                
        #         #Update volumes
        #         volumes0 = list(batch['volumes'][domain_type0])
        #         volumes1 = list(batch['volumes'][domain_type1])
                
        #         volumes0 = template.format(volumes0)
        #         volumes1 = template.format(volumes1)
                
        #         split_['_volumes'][domain_type0] = volumes0
        #         split_['_volumes'][domain_type1] = volumes1
                
        #         #Update start
        #         start = stop
            
        #     #Convert collated from list to string
        #     collated0 = template.format('[' + ', '.join([str(i) for i in collated0]) + ']')
        #     collated1 = template.format('[' + ', '.join([str(i) for i in collated1]) + ']')
            
        #     #Convert surface_idx2midpoint into One2One to allow 2 way access
        #     surface_idx2midpoint = One2One('surface_idx', 'midpoint', surface_idx2midpoint)
            
        #     #Update surface_datum
        #     #Add code chunks to surface_datum
        #     surface_datum['code'] = {'collated': {domain_type0 : collated0,
        #                                           domain_type1 : collated1,
        #                                           }, 
        #                              'split'   : split,
        #                              }
            
        #     #Add spatial data to surface_datum
        #     surface_datum['spatial'] = {'surface_idx2midpoint' : surface_idx2midpoint,
        #                                 'domain_type_idxs'     : domain_type_idxs,
        #                                 'shifts'               : shifts 
        #                                 }
            
        #     #Add a tree for searching
        #     midpoints             = list(surface_idx2midpoint.values())
        #     surface_datum['tree'] = spatial.KDTree(midpoints)
            
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
        spatial_data      = self.spatial_data
        variables         = spatial_data.variables
        state2domain_type = self.state2domain_type
        states            = set(spatial_data.states)
        global_variables  = self.global_variables
        bulk_variables    = self.bulk_variables
        surface_variables = self.surface_variables
        
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
        
        #Define the surface indices if they have not already been defined
        surface_datum = surface_data[surface]
        
        if ('vrb', surface) not in self.seen_code:
            lhs_ = [f'__{surface[0]}2surface_idx', f'__{surface[1]}2surface_idx']
            
            for lhs in lhs_:
                rhs   = surface_datum['code'][lhs]    
                chunk = '\t' + lhs + ' = ' + rhs
                self.variable_code += chunk + '\n'
                self.seen_code.add( ('vrb', surface))
            
            self.variable_code += '\n'
            
        #Update variable code
        expr          = variable.expr
        lhs           = ut.undot(variable.name)
        rhs           = ut.undot(self.sub(expr))
        idxs          = {surface[0]: f'__{surface[0]}2surface_idx',
                         surface[1]: f'__{surface[1]}2surface_idx',
                         }
        rhs           = rhs.format(**idxs)
        variable_code = f'\t{lhs} = {rhs}\n'
        
        self.variable_code += variable_code + '\n'
    
    ###########################################################################
    #Reactions
    ###########################################################################
    def _add_reaction_code(self) -> None:
        spatial_data      = self.spatial_data
        reactions         = spatial_data.reactions
        state2domain_type    = self.state2domain_type
        states               = set(spatial_data.states)
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
        
        #Define the surface indices if they have not already been defined
        surface_datum = surface_data[surface]
        
        if ('vrb', surface) not in self.seen_code:
            lhs_ = [f'__{surface[0]}2surface_idx', f'__{surface[1]}2surface_idx']
            
            for lhs in lhs_:
                rhs   = surface_datum['code'][lhs]    
                chunk = '\t' + lhs + ' = ' + rhs
                self.reaction_code += chunk + '\n'
                self.seen_code.add( ('vrb', surface))
            
            self.reaction_code += '\n'
        
        A         = surface_datum['code']['A']
        volume0   = surface_datum['code']['volume0'] 
        volume1   = surface_datum['code']['volume1']
        
        if ('rxn', surface) not in self.seen_code:
            self.reaction_code += f'\t__A_{surface[0]}_{surface[1]}       = {A}\n'
            self.reaction_code += f'\t__volume0_{surface[0]}_{surface[1]} = {volume0}\n'
            self.reaction_code += f'\t__volume1_{surface[0]}_{surface[1]} = {volume1}\n'
            self.reaction_code += '\n'
            self.seen_code.add(('rxn', surface))
        
        #Update reaction code
        expr          = reaction.rate
        lhs           = ut.undot(reaction.name)
        rhs           = ut.undot(self.sub(expr))
        idxs          = {surface[0]: f'__{surface[0]}2surface_idx',
                         surface[1]: f'__{surface[1]}2surface_idx',
                         }
        rhs           = rhs.format(**idxs)
        reaction_code = f'\t{lhs} = {rhs}\n\n'
        
        self.reaction_code += reaction_code 
        
        loop = 'for __i in range(len({idxs})): '
        itr  = '{dstate}[{idxs}[__i]] += {stoich}*{A}[__i]*{rxn}[__i]/{volume}[__i]'
        
        make_loop = lambda **kw: loop.format(**kw) + itr.format(**kw)
        
        #Update diff code
        diff_code = ''
        
        
        for state, stoich in reaction.stoichiometry.items():
            domain_type = state2domain_type[state]
            
            if domain_type == surface[0]:
                volume_ = f'__volume0_{surface[0]}_{surface[1]}'
            else:
                volume_ = f'__volume1_{surface[0]}_{surface[1]}'
            
            # line = make_loop(idxs    = f'__{domain_type}2surface_idx',
            #                  dstate  = ut.diff(ut.undot(state)),
            #                  rxn     = ut.undot(reaction.name),
            #                  A       = f'__A_{surface[0]}_{surface[1]}',
            #                  volume  = volume_,
            #                  stoich  = stoich
            #                  )
            d_state  = ut.diff(ut.undot(state))
            indices  = f'__{domain_type}2surface_idx'
            A        = f'__A_{surface[0]}_{surface[1]}'
            line     = f'__surface_reaction({d_state}, {indices}, {stoich}, {reaction.name}, {A}, {volume_})'   
            diff_code   = f'\t{line}\n' 
            
            self.reaction_code += diff_code + '\n'
         
    ###########################################################################
    #Retrieval
    ###########################################################################
    def get_surface_midpoints(self,
                              surface : Surface_type
                              ) -> list[tuple]:
        surface_idx2midpoint = self.surface_data[surface]['spatial']['surface_idx2midpoint']
        surface_midpoints    = list(surface_idx2midpoint.values())
        return surface_midpoints 
    
    def get_surface_idx(self, 
                        surface : Surface_type,
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
                      surface        : Surface_type,
                      cmap           : str='coolwarm',
                      norm           : Literal['linear', 'log']='linear',
                      label_surfaces : bool=True,
                      linewidth      : int = 5,
                      colorbar_ax    : plt.Axes=None,
                      ) -> dict:
        #Create the cache for the results
        results = {}
        
        #Parse the cmap and norm
        cmap_, norm_, *_ = self._parse_cmap(values, cmap, norm)
        
        #Set up the mappings
        domain_type_idx2voxel = self.voxel2domain_type_idx.inverse
        sizes                 = self.sizes
        surface_datum         = self.surface_data[surface]
        mappings              = surface_datum['mappings']
        surface_idx2midpoint  = surface_datum['surface_idx2midpoint']
        domain_type_idxs      = mappings[['voxel0_idx', 'voxel1_idx']]
        shifts                = mappings['shift']
        
        #Other overhead
        domain_type0, domain_type1 = surface
        
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
            line = ax.plot(x, y, linewidth=linewidth, color=color)
            
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
    
    