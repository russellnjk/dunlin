import re
from typing import Literal, Union

import dunlin.utils as ut
from .grid.stack           import Stack
from .grid.grid            import make_grids_from_config
from .geometrydefinition   import make_shapes
from dunlin.datastructures import SpatialModelData


class ShapeStack(Stack):
    #Element id
    element2idx: dict
    idx2element: dict 
    
    #Mappings required before iteration
    shape2domain            : dict
    domain2shape            : dict
    shape2domain_type       : dict
    domain_type2shape       : dict
    variable2domain_type    : dict
    domain_type2variable    : dict
    reaction2domain_type    : dict
    domain_type2reaction    : dict
    mass_transfer_templates : dict
    
    #For caching the differentials during iteration
    diff_terms : dict
    
    '''
    This class bridges the Stack class with SpatialModelData and generates the 
    code required by the model's rhs function.
    
    The algorithm is as follows:
        1. Make shapes from spatial data
        2. Make grid/mesh from spatial data
        3. Identify the main grid
        4. Construct the parent Stack class from the grid and the shapes
        5. Create mappings between geometry and non-geometry information such as 
        between domain types and reactions
        6. Iterate through each voxel 
        7. In each iteration, parse the variables, reactions, mass transfer and 
        rates and create the relevant code chunks
        8. Join code chunks to create the rhs for integration and for creating 
        the post-processing dictionary
        
    '''
    
    def __init__(self, spatial_data: SpatialModelData):
        
        #Generate shapes and grids
        shapes    = make_shapes(spatial_data['geometry_definitions'])
        grids     = make_grids_from_config(spatial_data['grid_config'])
        main_grid = next(iter(grids.values()))
        
        #Save attributes
        self.spatial_data = spatial_data
        self.grids        = grids
        
        #Call the parent constructor
        super().__init__(main_grid, *shapes)
        
        #Create auxillary mappings
        #Element numbering
        self.element2idx = {}
        self.idx2element = {}
        
        #Shape, domain, domain_type
        temp                   = self._map_shape2domain(spatial_data, *shapes)
        self.shape2domain      = temp[0]
        self.domain2shape      = temp[1]
        self.shape2domain_type = temp[2]
        self.domain_type2shape = temp[3]
        
        #Variable, domain_type
        temp                      = self._map_variable2domain_type(spatial_data)
        self.domain_type2variable = temp[0]
        self.variable2domain_type = temp[1]
        
        #Reaction, domain_type
        temp = self._map_reaction2domain_type(self.variable2domain_type,
                                              spatial_data
                                              )
        self.domain_type2reaction = temp[0]
        self.reaction2domain_type = temp[1]
        
        #Mass transfer
        self.mass_transfer_templates = self._template_mass_transfer(self.variable2domain_type, 
                                                                    spatial_data
                                                                    )
        
        #Rate, domain_type
        temp = self._map_rate2domain_type(self.domain_type2variable, 
                                          self.variable2domain_type, 
                                          spatial_data
                                          )

        self.domain_type2rate = temp[0]
        self.rate2domain_type = temp[1]

        #Create auxillary containers
        self.diff_terms               = {}
        self.seen_variable_items      = set()
        self.seen_reaction_items      = set()
        self.seen_mass_transfer_items = set()
        
        #Create initial code chunks
        temp                = self._empty_dict_code()
        helper              = lambda i: f'\t{i} = {temp}' 
        states              = [helper(i) for i in spatial_data.states.keys()]
        self.state_code     = '\n'.join(states) + '\n\n'
        parameters          = enumerate(spatial_data.parameters.keys())
        parameters          = [f'\t{p} = parameters[{i}]' for i, p in parameters]
        self.parameter_code = '\n'.join(parameters) + '\n\n'
        self.d_state_code   = ''
        self.function_code  = self._parse_functions(spatial_data)
        self.variable_code  = self._make_variable_overhead(self.variable2domain_type, 
                                                                     spatial_data
                                                                     )
        self.reaction_code            = self._make_reaction_overhead(self.reaction2domain_type)
        self.mass_transfer_code       = self._make_mass_transfer_overhead(self.mass_transfer_templates)
        
        #Iterate
        for voxel in self.voxels:
            self._parse(voxel)
        
        #Make differential code
        temp                = self._empty_dict_code()
        helper          = lambda x: f'\t{ut.diff(x)} = {temp}' 
        self.diff_code  = '\n'.join([helper(x) for x in spatial_data.states.keys()]) + '\n'
        self.diff_code += '\n'.join(self.diff_terms.values()) + '\n\n'
        
        #Update d_states code
        n                  = len(self.element2idx)
        # self.d_state_code  = f'\t_d_states = _np.zeros({n})\n\n' + self.d_state_code  + '\n' 
        self.d_state_code  = '\t_d_states = _np.array([\n' + self.d_state_code + '\t\t])\n'
        
        #Make body code
        self.body_code = '\n'.join(['\t#States',
                                    self.state_code,
                                    '\t#Parameters',
                                    self.parameter_code,
                                    '\t#Functions',
                                    self.function_code,
                                    '\t#Variables',
                                    self.variable_code,
                                    '\t#Reactions',
                                    self.reaction_code,
                                    '\t#Mass Transfer',
                                    self.mass_transfer_code,
                                    '\t#Differentials',
                                    self.diff_code,
                                    '\t#Assign to _d_states',
                                    self.d_state_code ,
                                    ])
        
    ###########################################################################
    #Mappings as Part of Overhead
    ###########################################################################
    @staticmethod
    def _map_shape2domain(spatial_data, *shapes) -> tuple:
        domain_types       = spatial_data.domain_types
        shape2domain       = {}
        domain2shape       = {}
        shape2domain_type  = {}
        domain_type2shape  = {}
        
        for shape in shapes:
            shape_name  = shape.name
            domain_type = shape.domain_type
            domains     = domain_types[domain_type]
            
            found = False
            for domain, internal_point in domains.items():
                if internal_point in shape:
                    found = True
                    break
            
            if found:
                if domain in domain2shape:
                    msg = f'Domain {domain} appears to belong to more than one shape.'
                else:
                    shape2domain[shape_name]      = domain
                    domain2shape[domain]          = shape_name
                    shape2domain_type[shape_name] = domain_type
                    domain_type2shape.setdefault(domain_type, set()).add(shape_name)
            else:
                msg = f'Could not assign shape {shape.name} to a domain.'
                raise ValueError(msg)
        
        
        return shape2domain, domain2shape, shape2domain_type, domain_type2shape 
    
    @classmethod
    def _map_variable2domain_type(cls,
                                  spatial_data : SpatialModelData,
                                  ) -> tuple[dict]:
        
        variables         = spatial_data.variables
        compartments      = spatial_data.compartments
        state2domain_type = compartments.state2domain_type
        states            = set(spatial_data.states.keys())
        variable_names    = set(variables.keys())
        
        domain_type2variable = {}
        variable2domain_type = {}
        
        templater = lambda : {'states': set(), 'variables': set()}
        
        for variable in variables.values():
            temp         = {}
            domain_types = []
            
            #Determine which states and variables are involved
            for name in variable.namespace:
                if name in states:
                    state       = name
                    domain_type = state2domain_type[state]
                    
                    temp.setdefault(domain_type, templater())
                    temp[domain_type]['states'].add(state)
                    domain_types.append(domain_type)
                    
                elif name in variable_names:
                    variable_   = name
                    domain_type = variable2domain_type[variable_]
                    
                    temp.setdefault(domain_type, templater())
                    temp[domain_type]['variables'].add(variable_)
                    
                    if domain_type is None:
                        pass
                    elif type(domain_type) == frozenset:
                        domain_types.extend(domain_type)
                    else:
                        domain_types.append(domain_type)
            
            #Update domain_type2variable 
            #Index the datum for each variable under its associated domain
            edge = frozenset(domain_types)
            expr = cls._reformat(variable.expr, states, variable2domain_type)
            if not edge:
                domain_type2variable.setdefault(None, {})[variable.name] = expr
                variable2domain_type[variable.name] = None
                
            elif len(edge) == 1:
                domain_type = domain_types[0]
                domain_type2variable.setdefault(domain_type, {})[variable.name] = expr
                variable2domain_type[variable.name] = domain_type
                
            elif len(edge) == 2:
                temp = {'expr': expr, 'domain_types': temp}
                domain_type2variable.setdefault(edge, {})[variable.name] = temp
                variable2domain_type[variable.name] = edge
                
            else:
                msg = 'Encountered a variable with more than 2 domain_types.'
                msg = f'{msg}\n{variable}: {edge}'
                raise ValueError(msg)
            
            
        return domain_type2variable, variable2domain_type
    
    @classmethod
    def _map_reaction2domain_type(cls,
                                  variable2domain_type : dict[str, Union[str, frozenset]],
                                  spatial_data         : SpatialModelData
                                  ) -> tuple[dict]:
        
        variables         = spatial_data.variables
        reactions         = spatial_data.reactions
        compartments      = spatial_data.compartments
        state2domain_type = compartments.state2domain_type
        states            = set(spatial_data.states.keys())
        variable_names    = set(variables.keys())
        
        domain_type2reaction = {}
        reaction2domain_type = {}
        
        templater = lambda : {'states': set(), 'variables': set()}
        
        for reaction in reactions.values():
            temp         = {}
            domain_types = []
            
            #Determine which states and variables are involved
            for name in reaction.namespace:
                if name in states:
                    state       = name
                    domain_type = state2domain_type[state]
                    
                    temp.setdefault(domain_type, templater())
                    temp[domain_type]['states'].add(state)
                    domain_types.append(domain_type)
                    
                elif name in variable_names:
                    variable_   = name
                    domain_type = variable2domain_type[variable_]
                    
                    temp.setdefault(domain_type, templater())
                    temp[domain_type]['variables'].add(variable_)
                    
                    if domain_type is None:
                        pass
                    elif type(domain_type) == frozenset:
                        domain_types.extend(domain_type)
                    else:
                        domain_types.append(domain_type)
                
            #Update domain_type2reaction 
            #Index the datum for each reaction under its associated domain
            edge = frozenset(domain_types)
            expr = cls._reformat(reaction.rate, states, variable2domain_type)
            if not edge:
                domain_type2reaction.setdefault(None, {})[reaction.name] = expr
                reaction2domain_type[reaction.name] = None
                
            elif len(edge) == 1:
                domain_type = domain_types[0]
                domain_type2reaction.setdefault(domain_type, {})[reaction.name] = expr
                reaction2domain_type[reaction.name] = domain_type
                
            elif len(edge) == 2:
                temp = {'expr': expr, 'domain_types': temp}
                domain_type2reaction.setdefault(edge, {})[reaction.name] = temp
                reaction2domain_type[reaction.name] = edge
                
            else:
                msg = 'Encountered a reaction with more than 2 domain_types.'
                msg = f'{msg}\n{reaction}: {edge}'
                raise ValueError(msg)
            
        return domain_type2reaction, reaction2domain_type
    
    @classmethod
    def _template_mass_transfer(cls,
                                variable2domain_type,
                                spatial_data: SpatialModelData
                                ) -> tuple:
        advection           = spatial_data.advection
        diffusion           = spatial_data.diffusion
        boundary_conditions = spatial_data.boundary_conditions
        state2domain_type   = spatial_data.compartments.state2domain_type
        states              = state2domain_type.keys()
        ndims               = len(spatial_data.coordinate_components)
        reformat            = lambda s: cls._reformat(s, states, variable2domain_type)
        
        mass_transfer_templates = {}
       
        for state, domain_type in state2domain_type.items():
            if state in spatial_data.rates:
                continue
            
            mass_transfer_templates[state] = {'advection'           : {},
                                              'diffusion'           : {},
                                              'boundary_conditions' : {}
                                              }
            temp =  {'advection'           : {},
                     'diffusion'           : {},
                     'boundary_conditions' : {}
                     }
            
            for axis in range(1, ndims+1):
                #Extract
                adv = advection.find(state, axis)
                dfn = diffusion.find(state, axis)
                bc0 = boundary_conditions.find(state,  axis)
                bc1 = boundary_conditions.find(state, -axis)
                
                #Reformat the string
                adv = reformat(adv)
                dfn = reformat(dfn)
                
                if bc0 is not None:
                    bc0 = {'type'      : bc0.condition_type,
                           'condition' : reformat(bc0.condition)
                           }
                if bc1 is not None:
                    bc1 = {'type'      : bc1.condition_type,
                           'condition' : reformat(bc1.condition)
                           }
                    
                #Update the mappings
                temp['advection'          ][ axis] = adv
                temp['diffusion'          ][ axis] = dfn
                temp['boundary_conditions'][ axis] = bc0
                temp['boundary_conditions'][-axis] = bc1
            
            mass_transfer_templates[state] = temp
            
        return mass_transfer_templates
            
    @classmethod
    def _map_rate2domain_type(cls, 
                              domain_type2variable, 
                              variable2domain_type,
                              spatial_data: SpatialModelData
                              ) -> tuple:
        rates             = spatial_data.rates
        states            = list(spatial_data.states.keys())
        parameters        = list(spatial_data.parameters.keys())
        functions         = list(spatial_data.functions.keys())
        names             = set(states + parameters + functions)
        domain_type2state = spatial_data.compartments.domain_type2state
        
        rate2domain_type = {}
        domain_type2rate = {}
        
        for domain_type, states in domain_type2state.items():
            for state in states:
                rate = rates.get(state)
                
                if not rate:
                    continue
                
                #Ensure the rate contains only variables from the same domain_type
                local = set(domain_type2variable[domain_type].keys())
                diff  = set(rate.namespace) - names - local
                
                if diff:
                    msg = f'Rate {rate.name} contains namespaces from domain_types other than {domain_type}.'
                    raise ValueError(msg)
                
                expr = cls._reformat(rate.expr, states, variable2domain_type)
                
                domain_type2rate.setdefault(domain_type, {})[state] = expr
                rate2domain_type[state] = domain_type
        
        return domain_type2rate, rate2domain_type
        
    ###########################################################################
    #Name Generation
    ###########################################################################
    @staticmethod
    def _combine_voxels(*voxels):
        lst    = sorted(voxels)
        # result = tuple([i for tup in lst for i in tup])
        result = tuple(lst)
        return result
    
    @classmethod
    def _make_term(cls, prefix: str, *voxels):
        voxels = cls._combine_voxels(*voxels)
        # helper = lambda i: str(int(i)) if ut.isint(i) else str(i) 
        # to_str = lambda voxel: '(' + ', '.join([str(i) for i in voxel]) + ')'
        # chunks = ', '.join([to_str(voxel) for voxel in voxels])
        # return f'{prefix}[{chunks}]'
        
        fmt    = lambda i: str(int(i)) if ut.isint(i) else str(i) 
        helper = lambda i: fmt(i).replace('.', 'p') 
        to_str = lambda voxel: '_'.join([helper(i) for i in voxel])
        chunks = '__'.join([to_str(voxel) for voxel in voxels])
        return f'{prefix}_{chunks}'
    
    @classmethod
    def _make_boundary(cls, state: str, shift: Literal[-3, -2, -1, 1, 2, 3]) -> str:
        sign  = f'{shift}' if shift > 0 else f'{abs(shift)}_'
        return f'_bc_{state}_{sign}'
    
    @staticmethod
    def _reformat(expr, states, variable2domain_type):
        if not ut.isstrlike(expr):
            return expr
        
        def repl(match):
            if match[0] in states:
                return '{' + match[0] + '}'
            elif  variable2domain_type.get(match[0]) is not None:
                return '{' + match[0] + '}'
            else:
                return match[0]
            
        return re.sub('[a-zA-z]\w*', repl, expr)
    
    @staticmethod
    def _empty_dict_code(n=1):
        return '{}'
    
    ###########################################################################
    #Overhead Code Chunks
    ###########################################################################
    @classmethod
    def _make_variable_overhead(cls,
                                variable2domain_type : dict[str, Union[str, frozenset]],
                                spatial_data         : SpatialModelData
                                ) -> str:
        code      = ''
        variables = spatial_data.variables
        
        for variable_name, domain_type in variable2domain_type.items():
            if domain_type is None:
                rhs = variables[variable_name].expr 
            elif type(domain_type) == str:
                rhs = cls._empty_dict_code(1)
            else:
                rhs = cls._empty_dict_code(2)
                
            code += f'\t{variable_name} = {rhs}\n'
        return code + '\n'
    
    @classmethod
    def _make_reaction_overhead(cls,
                                reaction2domain_type : dict[str, Union[str, frozenset]],
                                ) -> str:
        code      = ''
        
        for reaction_name, domain_type in reaction2domain_type.items():
            if type(domain_type) == str:
                line  = cls._empty_dict_code(1)
            else:
                line  = cls._empty_dict_code(2)
                
            code += f'\t{reaction_name} = {line}\n'
        return code + '\n'
    
    @classmethod 
    def _make_mass_transfer_overhead(cls, 
                                     mass_transfer_templates : dict,
                                     ) -> str:
        code   = ''
        
        for state, datum in mass_transfer_templates.items():
            #Advection and diffusion
            rhs   = cls._empty_dict_code(2)
            lhs   = '_adv_' + state
            code += f'\t{lhs} = {rhs}\n'
            
            lhs   = '_dfn_' + state
            code += f'\t{lhs} = {rhs}\n'
            
            #Boundary conditions
            for shift in datum.get('boundary_conditions', {}):
                sign  = f'{shift}' if shift > 0 else f'{abs(shift)}_'
                rhs   = cls._empty_dict_code(1)
                lhs   = f'_bc_{state}_{sign}'
                code += f'\t{lhs} = {rhs}\n'
            
        return code + '\n'
    
    ###########################################################################
    #Parsers for _add_voxel
    ###########################################################################
    @classmethod
    def _parse_functions(cls, spatial_data: SpatialModelData) -> str:
        code = ''
        
        for function in spatial_data.functions.values():
            signature = function.signature
            name      = function.name
            expr      = function.expr
            chunk     = f'\tdef {name}({signature}):\n\t\treturn{expr}\n'
            
            code += chunk
        
        return code + '\n'
        
    @classmethod
    def _parse_variables(cls,
                         voxel                : tuple,
                         voxels               : dict[tuple, dict],
                         shape2domain_type    : dict[str, Union[str, frozenset]],
                         domain_type2variable : dict[str, tuple],
                         seen_items           : set[frozenset],
                         spatial_data         : SpatialModelData
                         ) -> dict:
        
        variables = spatial_data.variables
        datum     = voxels[voxel]
        code      = ''
        
        #Parse the bulk and global variables
        shape_name  = datum['shape']
        domain_type = shape2domain_type[shape_name]
        
        bulk_variables = domain_type2variable.get(domain_type, {})
        to_sub         = list(spatial_data.states.keys()) + list(variables.keys())
        to_sub         = {i: cls._make_term(i, voxel) for i in to_sub}
        
        for variable_name, expr in bulk_variables.items():
            lhs   = cls._make_term(variable_name, voxel)
            rhs   = expr.format(**to_sub)
            code += f'\t{lhs} = {rhs}\n'
        
        #Parse the edge variables
        edges = datum['edges']
        for edge, neighbours in edges.items():
            #Determine the edge variables to create
            domain_type0   = shape2domain_type[edge[0]]
            domain_type1   = shape2domain_type[edge[1]]
            domain_type    = frozenset((domain_type0, domain_type1))
            edge_variables = domain_type2variable.get(domain_type, {})
            
            for variable_name, datum in edge_variables.items():
                
                for neighbour, shift in neighbours.items():
                    key = frozenset((variable_name, voxel, neighbour))
                    
                    if key in seen_items:
                        continue

                    expr           = datum['expr']
                    domain_types   = datum['domain_types']
                    empty          = {'variables': [], 'states': []}
                    states_grp0    = domain_types.get(domain_type0, empty)['states'   ]
                    states_grp1    = domain_types.get(domain_type1, empty)['states'   ]
                    variables_grp0 = domain_types.get(domain_type0, empty)['variables']
                    variables_grp1 = domain_types.get(domain_type1, empty)['variables']
                    variables_grp2 = domain_types.get(domain_type , empty)['variables']
                    
                    states_grp0    = {i: cls._make_term(i, voxel)            for i in states_grp0   }
                    states_grp1    = {i: cls._make_term(i, neighbour)        for i in states_grp1   }
                    variables_grp0 = {i: cls._make_term(i, voxel)            for i in variables_grp0}
                    variables_grp1 = {i: cls._make_term(i, neighbour)        for i in variables_grp1}
                    variables_grp2 = {i: cls._make_term(i, voxel, neighbour) for i in variables_grp2}

                    lhs = cls._make_term(variable_name, voxel, neighbour)
                    rhs = expr.format(**states_grp0,
                                      **states_grp1,
                                      **variables_grp0,
                                      **variables_grp1,
                                      **variables_grp2
                                      )
                    
                    code += f'\t{lhs} = {rhs}\n'
                    
                    #Update seen_items to prevent double-processing
                    seen_items.add(key)
        
        if code:
            return f'\t#{voxel}\n' + code + '\n'
        else:
            return ''
    
    @classmethod
    def _parse_reactions(cls,
                         voxel                : tuple,
                         voxels               : dict[tuple, dict],
                         shape2domain_type    : dict[str, Union[str, frozenset]],
                         domain_type2variable : dict[str, dict],
                         domain_type2reaction : dict[str, dict],
                         seen_items           : set[frozenset],
                         diff_terms           : dict[str],
                         spatial_data         : SpatialModelData
                         ) -> dict:
        
        reactions         = spatial_data.reactions
        variables         = spatial_data.variables
        datum             = voxels[voxel]
        size              = datum['size']
        shape             = voxels[voxel]['shape']
        voxel_domain_type = shape2domain_type[shape]
        code              = ''

        state2domain_type = spatial_data.compartments.state2domain_type
        
        #Parse the bulk reactions
        shape_name  = datum['shape']
        domain_type = shape2domain_type[shape_name]
        
        bulk_reactions = domain_type2reaction.get(domain_type, {})
        to_sub         = list(spatial_data.states.keys()) + list(variables.keys())
        to_sub         = {i: cls._make_term(i, voxel) for i in to_sub}
        
        for reaction_name, rate in bulk_reactions.items():
            lhs   = cls._make_term(reaction_name, voxel)
            rhs   = rate.format(**to_sub)
            code += f'\t{lhs} = {rhs}\n'
            
            #Update diff_terms
            reaction      = reactions[reaction_name]
            stoichiometry = reaction.stoichiometry
            
            for state, coeff in stoichiometry.items():
                chunk = f'{coeff}*{lhs} '
                cls._add2diff_terms(diff_terms, state, voxel, chunk)
                
        #Parse the edge reactions
        edges = datum['edges']
        for edge, neighbours in edges.items():
            #Determine the edge variables to create
            domain_type0   = shape2domain_type[edge[0]]
            domain_type1   = shape2domain_type[edge[1]]
            domain_type    = frozenset((domain_type0, domain_type1))
            edge_reactions = domain_type2reaction.get(domain_type, {})
            
            for reaction_name, datum in edge_reactions.items():
                
                reaction      = reactions[reaction_name]
                stoichiometry = reaction.stoichiometry
                
                for neighbour, shift in neighbours.items():
                    key = frozenset((reaction_name, voxel, neighbour))
                    
                    if key in seen_items:
                        continue
                    
                    expr           = datum['expr']
                    domain_types   = datum['domain_types']
                    empty          = {'variables': [], 'states': []}
                    states_grp0    = domain_types.get(domain_type0, empty)['states'   ]
                    states_grp1    = domain_types.get(domain_type1, empty)['states'   ]
                    variables_grp0 = domain_types.get(domain_type0, empty)['variables']
                    variables_grp1 = domain_types.get(domain_type1, empty)['variables']
                    variables_grp2 = domain_types.get(domain_type , empty)['variables']
                    
                    states_grp0    = {i: cls._make_term(i, voxel)            for i in states_grp0   }
                    states_grp1    = {i: cls._make_term(i, neighbour)        for i in states_grp1   }
                    variables_grp0 = {i: cls._make_term(i, voxel)            for i in variables_grp0}
                    variables_grp1 = {i: cls._make_term(i, neighbour)        for i in variables_grp1}
                    variables_grp2 = {i: cls._make_term(i, voxel, neighbour) for i in variables_grp2}

                    lhs = cls._make_term(reaction_name, voxel, neighbour)
                    rhs = expr.format(**states_grp0,
                                      **states_grp1,
                                      **variables_grp0,
                                      **variables_grp1,
                                      **variables_grp2
                                      )
                    size_  = min(voxels[neighbour]['size'], size)
                    ndims  = len(voxel)
                    rhs    = f'({rhs}) *{size_}**{ndims-1}'
                    code  += f'\t{lhs} = {rhs}\n'
                    
                    #Update seen_items to prevent double-processing
                    seen_items.add(key)
                    
                    #Update diff_terms
                    for state, coeff in stoichiometry.items():
                        # print(state, state2domain_type[state])
                        chunk = f'{coeff}*{lhs} '
                        
                        if state2domain_type[state] == voxel_domain_type:
                            cls._add2diff_terms(diff_terms, state, voxel, chunk)
                        else:
                            cls._add2diff_terms(diff_terms, state, neighbour, chunk)
        
        if code:
            return f'\t#{voxel}, {state}\n' + code +' \n'
        else:
            return ''
    
    @classmethod
    def _parse_mass_transfer(cls,
                             state                    : str,
                             voxel                    : tuple,
                             voxels                   : dict[tuple, dict],
                             mass_transfer_templates  : dict[str, dict[str, dict]],
                             shape2domain_type        : dict[str, Union[str, frozenset]],
                             seen_items               : set,
                             diff_terms               : dict,
                             ) -> dict:
        
        code       = ''
        datum      = voxels[voxel]
        neighbours = datum['neighbours']
        boundaries = datum['boundaries']
        shape      = datum['shape']
        size       = datum['size']
        ndims      = len(voxel)
        shifts     = [i for i in range(-ndims, ndims + 1) if i]
        adv        = mass_transfer_templates[state]['advection']
        dfn        = mass_transfer_templates[state]['diffusion']
        bcs        = mass_transfer_templates[state]['boundary_conditions']
        
        #Iterate and cache the terms 
        for shift in shifts:
            axis = abs(shift)
            
            if shift in boundaries:
                #Boundary condition
                key = state, voxel
                bc  = bcs[shift]
                
                condition_type = bc['type']
                condition      = bc['condition']
                
                if not bc:
                    continue
                
                elif condition_type == 'Dirichlet':
                    coeff = dfn[abs(shift)]
                    conc0 = f'{state}[{voxel}]'
                    conc1 = condition
                    grad  = f'({conc1} - {conc0})' 
                    rhs   = f'{coeff} *{grad} *{size}**{ndims-1} '
                
                elif condition_type == 'Neumann':
                    rhs = f'{condition} *{size}**{ndims-1} '
            
                else:
                    b   = condition_type
                    msg = f'No implementation for {b} boundary condition.' 
                    raise NotImplementedError(msg)
                
                # sign   = f'{shift}' if shift > 0 else f'{abs(shift)}_'
                prefix = cls._make_boundary(state, shift)
                lhs    = cls._make_term(prefix, voxel)
                code  += f'\t{lhs} = {rhs}\n'
                
                #Update mass transfer terms and diff terms
                cls._add2diff_terms(diff_terms, state, voxel, f'+{lhs} ')
                
            else:
                for neighbour in neighbours[shift]:
                    #Skip if the neighbour is not of the same shape
                    neighbour_shape = voxels[neighbour]['shape']
                    
                    if shape != neighbour_shape:
                        continue
                    
                    #Skip if this has been computed before
                    key = state, frozenset((voxel, neighbour))
                    if key in seen_items:
                        continue
                    
                    seen_items.add(key)
                    
                    #Advection
                    coeff = adv[axis]
                    lhs   = cls._make_term(f'_adv_{state}' , voxel, neighbour)
                    
                    if shift > 0: 
                        src = cls._make_term(state, voxel)
                    else:
                        src = cls._make_term(state, neighbour)
                        
                    size_  = voxels[neighbour]['size']
                    code  += f'\t{lhs} = {coeff} *{src} *{size_}**{ndims-1}\n'
                    
                    if shift > 0:
                        cls._add2diff_terms(diff_terms, state, voxel,     f'-{lhs} ')
                        cls._add2diff_terms(diff_terms, state, neighbour, f'+{lhs} ')
                    else:
                        cls._add2diff_terms(diff_terms, state, voxel,     f'+{lhs} ')
                        cls._add2diff_terms(diff_terms, state, neighbour, f'-{lhs} ')
                
                    #Diffusion
                    coeff = dfn[axis]
                    lhs   = cls._make_term(f'_dfn_{state}' , voxel, neighbour)
                    
                    if shift > 0:
                        conc0  = cls._make_term(state, voxel)
                        conc1  = cls._make_term(state, neighbour)
                    else:
                        conc0  = cls._make_term(state, neighbour)
                        conc1  = cls._make_term(state, voxel)
                        
                    size_  = voxels[neighbour]['size']
                    grad   = f'({conc0} - {conc1})'
                    code  += f'\t{lhs} = {coeff} *{grad} *{size_}**{ndims-1}\n'
                    
                    if shift > 0:
                        cls._add2diff_terms(diff_terms, state,     voxel, f'-{lhs} ')
                        cls._add2diff_terms(diff_terms, state, neighbour, f'+{lhs} ')
                    else:
                        cls._add2diff_terms(diff_terms, state,     voxel, f'+{lhs} ')
                        cls._add2diff_terms(diff_terms, state, neighbour, f'-{lhs} ')
        
        if code:
            return f'\t#{voxel}, {state}\n' + code +' \n'
        else:
            return ''
    
    @classmethod
    def _parse_rates(cls,
                     voxel             : tuple,
                     voxels            : dict[tuple, dict],
                     shape2domain_type : dict[str, str],
                     domain_type2rate  : dict[str, dict[str, str]],
                     diff_terms        : dict[tuple, str],
                     spatial_data      : SpatialModelData
                     ) -> str:
        
        shape       = voxels[voxel]['shape']
        domain_type = shape2domain_type[shape]
        rates       = domain_type2rate.get(domain_type)
        
        if not rates:
            return
        
        variables   = spatial_data.variables
        to_sub      = list(spatial_data.states.keys()) + list(variables.keys())
        to_sub      = {i: cls._make_term(i, voxel) for i in to_sub}
        
        for state, expr in rates.items():
            rhs = expr.format(**to_sub) + ' '
            
            cls._add2diff_terms(diff_terms, state, voxel, rhs)
    
    ###########################################################################
    #Updating Differential Equations
    ###########################################################################
    @classmethod
    def _add2diff_terms(cls, diff_terms, state, voxel, chunk):
        key = state, voxel
        lhs = ut.diff(state)
        lhs = f'{lhs}[{voxel}]'
        
        diff_terms.setdefault(key, f'\t{lhs} = ')
        diff_terms[key] += chunk
    
    ###########################################################################
    #Extension of Parent Class _add_voxel
    ###########################################################################
    def _parse(self, voxel):
        voxels       = self.voxels
        datum        = voxels[voxel]
        spatial_data = self.spatial_data
        compartments = spatial_data.compartments
        
        #Extract information and update voxel2all
        shape_name       = self.voxel2shape[voxel]
        domain_type      = self.shapes[shape_name].domain_type
        domain           = self.shape2domain[shape_name]
        compartment      = compartments.domain_type2compartment[domain_type]
        states           = compartments.domain_type2state[domain_type]
        
        template = {'domain_type' : domain_type,
                    'compartment' : compartment,
                    'domain'      : domain,
                    'states'      : states
                    }
      
        datum.update(template)
        
        #Extract auxillary mappings
        shape2domain_type        = self.shape2domain_type
        domain_type2variable     = self.domain_type2variable
        domain_type2reaction     = self.domain_type2reaction
        domain_type2rate         = self.domain_type2rate
        mass_transfer_templates  = self.mass_transfer_templates
        seen_variable_items      = self.seen_variable_items
        seen_reaction_items      = self.seen_reaction_items
        seen_mass_transfer_items = self.seen_mass_transfer_items
        diff_terms               = self.diff_terms
        
        
        self.variable_code += self._parse_variables(voxel,
                                                    voxels,
                                                    shape2domain_type, 
                                                    domain_type2variable, 
                                                    seen_variable_items, 
                                                    spatial_data
                                                    )
        
        self.reaction_code += self._parse_reactions(voxel, 
                                                    voxels, 
                                                    shape2domain_type, 
                                                    domain_type2variable, 
                                                    domain_type2reaction, 
                                                    seen_reaction_items, 
                                                    diff_terms, 
                                                    spatial_data
                                                    )
        
        self._parse_rates(voxel, 
                          voxels, 
                          shape2domain_type,
                          domain_type2rate,
                          diff_terms,
                          spatial_data
                          )
        
        shape       = datum['shape']
        domain_type = shape2domain_type[shape]
        states      = spatial_data.compartments.domain_type2state[domain_type]
        
        for state in states:
            idx  = len(self.element2idx)
            dx = ut.diff(state)
            self.element2idx[voxel, state]  = idx
            self.idx2element[idx]           = voxel, state
            self.state_code                += f'\t{state}[{voxel}] = states[{idx}]\n'
            # self.d_state_code              += f'\t_d_states[{idx}] = {dx}[{voxel}]\n' 
            self.d_state_code              += f'\t\t{dx}[{voxel}],\n'
            if state not in mass_transfer_templates:
                continue
            
            self.mass_transfer_code += self._parse_mass_transfer(state, 
                                                                 voxel, 
                                                                 voxels, 
                                                                 mass_transfer_templates, 
                                                                 shape2domain_type, 
                                                                 seen_mass_transfer_items, 
                                                                 diff_terms
                                                                 )
        
        
