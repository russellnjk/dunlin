import numpy as np
import re

import dunlin.utils as ut
# from .csgnode import parse_node 
# from .utils   import add_coordinate
from .stack   import ShapeStack
# from .voxel   import make_grids_from_config

def make_code(spatial_data, _use_numba=True):
    stack = make_stack(spatial_data)
    
    voxel2dmnt = map_voxel_domain_type(stack, spatial_data)
    
    
    x_code, x2idx      = make_x_code(spatial_data, stack)
    p_code             = make_p_code(spatial_data)
    func_code          = make_func_code(spatial_data)
    vrb_code          = make_vrb_code(spatial_data)
    rxn_code, tr_rxns  = make_bulk_rxn_code(spatial_data)
    rt_code            = make_rt_code(spatial_data)
    
    adv_code, dfn_code, tr_rxn_code = make_bulk_mass_transfer_code(spatial_data, 
                                                                   stack, 
                                                                   tr_rxns, 
                                                                   x2idx, 
                                                                   voxel2dmnt,
                                                                   )
    diffs_code = make_differentials_code(spatial_data, tr_rxns)
    bc_code    = make_boundary_condition_code(spatial_data, stack, voxel2dmnt)
    
    common_code = [x_code, p_code, func_code, vrb_code, rxn_code, rt_code,
                   adv_code, dfn_code, tr_rxn_code, diffs_code, bc_code
                   ]
    
    common_code = '\n'.join(common_code)
    
    rhs_def    = make_rhs_def(spatial_data, _use_numba)
    rhs_return = make_rhs_return(x2idx)
    
    rhs_code = rhs_def + '\n' + common_code + '\n' + rhs_return
    
    return stack, rhs_code, None

def make_stack(spatial_data):
    stack = ShapeStack.from_spatial_data(spatial_data)
    
    return stack
    
###############################################################################
#Specific Code
###############################################################################
def make_rhs_def(spatial_data, use_numba=True):
    name = '__'.join(spatial_data['ref'])
    line = f'def spatial_{name}(time, states, parameters):'
    
    if use_numba:
        line = '@__njit\n' + line
    
    return line


def make_rhs_return(x2idx):
    dx  = ', '.join([ut.diff(x) for x in x2idx.keys()])
    arr = f'__np.concatenate(({dx}))'
    
    return_code = f'\treturn {arr}'
    
    return return_code

###############################################################################  
#Non-Spatial-Related Common Code
############################################################################### 
def make_x_code(spatial_data, stack):
    voxels = stack.voxels
    
    #Unpack states
    xs         = spatial_data['model']['states'].names
    n_voxels   = len(voxels)
    x2idx      = {}
    x_code     = '\t#States\n'
    for i, x in enumerate(xs):
        start, stop  = i*n_voxels, (i+1)*n_voxels
        x2idx[x]     = start, stop
        x_code      += f'\t{x} = states[{start}: {stop}]\n'
    
    return x_code, x2idx

def make_p_code(spatial_data):
    #Unpack the params
    ps     = spatial_data['model']['parameters'].names
    p_code = '\t#Parameters\n'
    for i, p in enumerate(ps):
        p_code += f'\t{p} = parameters[{i}]\n'
    
    return p_code

def make_func_code(spatial_data):
    #Define local functions
    funcs     = spatial_data['model']['functions']
    func_code = '\t#Functions\n'
    for func in funcs.values():
        definition = f'\tdef {func.name}({func.signature}):\n'
        return_val = f'\t\treturn {func.expr}\n'
        
        func_code += definition + return_val
    
    return func_code

def make_vrb_code(spatial_data):
    #Define the variables
    vrbs = spatial_data['model'].get('variables', {})
    vrb_code = '\t#Variables\n'
    for vrb in vrbs.values():
        vrb_code += f'\t{vrb.name} = {vrb.expr}\n'
    
    return vrb_code

def make_bulk_rxn_code(spatial_data):
    
    #Define the reactions
    model_data = spatial_data['model']
    cpts       = model_data['compartments']
    rxns       = model_data.get('reactions', {})
    rxn_code   = '\t#Bulk reactions\n'
    
    tr_rxns        = {}
    state_names    = list(model_data['states'].names)
    variable_names = list(model_data['variables']) if 'variables' in model_data else []
    variable_names = [i for i in variable_names if not ut.ismath(i)]
    namespaces     = state_names + variable_names
    
    for rxn in rxns.values():
        #Extract reactants and products
        rcts  = rxn.reactants
        prods = rxn.products
        
        #Extract domain_type from compartment
        rcts_dmnt_name  = check_compartment(rxn.name, rcts, cpts)
        prods_dmnt_name = check_compartment(rxn.name, prods, cpts)
        
        #Create domain_type pairs
        #Rxns across domain types: dmnt0 are dmnt1 different
        #Rxns within domain type: dmnt0 and dmnt1 are the same
        if rcts_dmnt_name is None and prods_dmnt_name is None:
            r   = f'"{rxn.name}"'
            msg = f'Could not map reactants/products of {r} to a compartment.'
            raise ValueError(msg)  
        elif rcts_dmnt_name is None or prods_dmnt_name is None:
            rxn_code += f'\t{rxn.name} = {rxn.rate}\n'
            
        else:
            key      = (rcts_dmnt_name, prods_dmnt_name)
            template = make_reaction_template(rxn, namespaces)
            
            tr_rxns.setdefault(key, []).append([rxn, template])
            
    
    return rxn_code, tr_rxns

def make_rt_code(spatial_data):
    #Define the rates
    rts = spatial_data['model'].get('rates', {})
    rt_code = '\t#Rates\n'
    for rt in rts.values():
        rt_code += f'\t{rt.name} = {rt.expr}\n'
    
    return rt_code

###############################################################################
#Spatial-Related Common Code
###############################################################################
def make_bulk_mass_transfer_code(spatial_data, 
                                 stack, 
                                 tr_rxns, 
                                 x2idx, 
                                 voxel2dmnt,
                                 ):
    #Extract geometry-related intermediates
    ndims    = stack.ndims
    voxels   = stack.voxels
    sizes    = stack.sizes 
    shifts   = [i for i in range(-ndims, ndims+1) if i != 0]
    n_voxels = len(voxels)
    e_voxels = {voxel: (i, neighbours) for i, (voxel, neighbours) in enumerate(voxels.items())}
    
    #Extract model-related intermediates
    xs   = spatial_data['model']['states'].names
    rts  = spatial_data['model'].get('rates', {})
    advs = spatial_data['model'].get('advection', {})  
    dfns = spatial_data['model'].get('diffusion', {}) 
    
    #Iterate and cache code chunks
    adv_dct    = {}
    dfn_dct    = {}
    tr_rxn_dct = {}
    
    for x in xs:
        if x in rts:
            continue
        
        adv_dct[x]    = ['']*n_voxels
        dfn_dct[x]    = ['']*n_voxels
        tr_rxn_dct[x] = ['']*n_voxels
        start, stop   = x2idx[x]
        
        for voxel, (voxel_num, neighbours) in e_voxels.items():
            size = sizes[voxel]
            
            for shift in shifts:
                if shift not in neighbours:
                    continue
                
                for neighbour in neighbours[shift]:
                    neighbour_num  = e_voxels[neighbour][0]
                    neighbour_size = sizes[neighbour]
                    
                    #Advection (Only once per shift)
                    if x in advs:
                        chunk = make_advection(advs, 
                                               x, 
                                               voxel_num, 
                                               neighbour_num, 
                                               shift, 
                                               size,
                                               neighbour_size,
                                               ndims
                                               )
                    
                        adv_dct[x][voxel_num] += chunk + ' '
                    
                    #Diffusion
                    if x in dfns:
                        chunk = make_diffusion(dfns, 
                                               x, 
                                               voxel_num, 
                                               neighbour_num, 
                                               shift, 
                                               size, 
                                               neighbour_size
                                               )
                    
                        dfn_dct[x][voxel_num] += chunk + ' '
                    
                    #Transfer reactions
                    voxel_dmnt     = voxel2dmnt[voxel]['domain_type']
                    neighbour_dmnt = voxel2dmnt[neighbour]['domain_type']
                    
                    if voxel_dmnt != neighbour_dmnt:

                        key = (voxel_dmnt, neighbour_dmnt)
                        if key in tr_rxns:
                            tr_rxns_lst = tr_rxns[key]
                            outwards    = True
                        else:
                            tr_rxns_lst = tr_rxns[key[::-1]]
                            outwards    = False
                            
                        chunk          = make_tr_rxns(x,
                                                      voxel_num, 
                                                      neighbour_num,
                                                      voxel_dmnt,
                                                      neighbour_dmnt,
                                                      size,
                                                      tr_rxns_lst,
                                                      outwards
                                                      )
                    
                        tr_rxn_dct[x][voxel_num] += chunk + ' '
            
            # print(adv_dct['H'])
            # if voxel == (3, 1):
            #     assert False
        #Convert to code by joining chunks
        adv_code    = '\t#Advection\n'
        dfn_code    = '\t#Diffusion\n'
        tr_rxn_code = '\t#Transfer Reactions\n'
        
    for x, adv_lst in adv_dct.items():
        lhs       = ut.adv(x)
        adv_code += f'\t{lhs} = __np.array([\n'
        
        lhs       = ut.dfn(x)
        dfn_code += f'\t{lhs} = __np.array([\n'
        dfn_lst = dfn_dct[x]
        
        lhs          = f'__tr_{x}'
        tr_rxn_code +=  f'\t{lhs} = __np.array([\n'
        tr_rxn_lst   = tr_rxn_dct[x]
        
        zipped = zip(adv_lst, dfn_lst, tr_rxn_lst, voxels)
        for a_chunk, d_chunk, t_chunk, voxel in zipped:
            a_chunk = a_chunk.strip()
            d_chunk = d_chunk.strip()
            t_chunk = t_chunk.strip()
            
            if a_chunk:
                adv_code += f'\t\t{a_chunk}, #{voxel}\n'
            else:
                adv_code += f'\t\t0, #{voxel}\n'
            
            if d_chunk:
                dfn_code += f'\t\t{d_chunk}, #{voxel}\n'
            else:
                dfn_code += f'\t\t0, #{voxel}\n'
            
            if t_chunk:
                tr_rxn_code += f'\t\t{t_chunk}, #{voxel}\n'
            else:
                tr_rxn_code += f'\t\t0, #{voxel}\n'

        adv_code    += '\t\t])\n\n'
        dfn_code    += '\t\t])\n\n'
        tr_rxn_code += '\t\t])\n\n'
    
    return adv_code, dfn_code, tr_rxn_code

def make_diffusion(dfns, x, voxel_num, neighbour_num, shift, size, neighbour_size):
    dist           = size + neighbour_size
    conc           = f'{x}[{voxel_num}]' 
    neighbour_conc = f'{x}[{neighbour_num}]' 
    
    conc_diff = f'({neighbour_conc} - {conc})'
    conc_grad = f'{conc_diff}/{dist}' 
    coeff     = dfns[x][abs(shift)]
    chunk     = f'+{coeff}*{conc_grad}'
    
    return chunk

def make_advection(advs, x, voxel_num, neighbour_num, shift, size, neighbour_size, ndims):
    
    if shift > 0:
        adv_sign = '-' if shift > 0 else '+'
        conc     = f'{x}[{voxel_num}]' 
        area     = size**(ndims-1)
    else:
        adv_sign = '+'
        conc     = f'{x}[{neighbour_num}]'
        area     = neighbour_size**(ndims-1)
    
    #Advection
    velocity = advs[x][abs(shift)]
    vol_flow = f'{velocity}*{area}'
    chunk    = f'{adv_sign}{conc}*{vol_flow}'
    
    return chunk
    
def make_tr_rxns(x,
                 voxel_num, 
                 neighbour_num,
                 voxel_dmnt,
                 neighbour_dmnt,
                 size,
                 tr_rxns_lst,
                 outwards: bool
                 ):
    
    chunk = ''
    
    
    if outwards:
        voxel_num0 = voxel_num
        voxel_num1 = neighbour_num
    else:
        voxel_num0 = neighbour_num
        voxel_num1 = voxel_num
        
    for rxn, rxn_template in tr_rxns_lst:
        #Extract the reaction coefficient
        coeff = rxn.stoichiometry.get(x)
        if coeff is None:
            continue
        
        # template  = rxn_templates[rxn.name]
        rate      = rxn_template.format(voxel_num0=voxel_num0, 
                                        voxel_num1=voxel_num1
                                        )
        chunk    += f'{coeff}*({rate}) '
    
    return chunk.strip()

def make_differentials_code(spatial_data, tr_rxns):
    states    = spatial_data['model']['states'].names
    rates     = spatial_data['model']['rates'].states
    reactions = spatial_data['model']['reactions']
    tr_rxns   = set([r[0].name for lst in tr_rxns.values() for r in lst])
    reactions = [rxn for rxn in reactions.values() if rxn.name not in tr_rxns]
    
    code = '\t#Non-rate differentials\n'
    for x in states:
        if x in rates:
            continue
        
        #LHS
        lhs = ut.diff(x)
        
        #Mass transfer
        bulk_mt  = f'{ut.adv(x)} + {ut.dfn(x)}'
        transfer = f'+__tr_{x}'
        
        #Bulk reactions
        rxn      = [rxn.stoichiometry[x] + f'*{rxn.name}' for rxn in reactions if x in rxn.stoichiometry]
        if rxn:
            rxn = ' '.join(rxn)
        else:
            rxn = ''
        
        #Update
        code    += f'\t{lhs} = {bulk_mt} {transfer} {rxn}\n'
        
    return code

def make_boundary_condition_code(spatial_data, stack, voxel2dmnt):
    #Add extra lines to modify for boundary cases
    bcs                 = spatial_data['model'].get('boundary_conditions', {})
    dmnt2x              = map_compartments_to_domain_types(spatial_data)
    
    bc_code = '\t#Apply boundary conditions\n'
    for voxel, info in voxel2dmnt.items():
        boundaries = info['boundaries']
        
        if not boundaries:
            continue
        
        dmnt      = info['domain_type']
        voxel_num = info['voxel_num'] 
        xs        = dmnt2x[dmnt]
        
        for x in xs:
            lhs      = f'{ut.diff(x)}[{voxel_num}]'
            rhs, a   = apply_boundary_conditions(x, bcs, boundaries)
            bc_code += f'\t{lhs} = {rhs} #{voxel}, {a}\n'

    return bc_code

def apply_boundary_conditions(x, bcs, boundaries):
    '''Returns upon encountering the first Dirichlet condition.
    '''
    code           = ''
    condition_type = 'Neumann'
    
    for n_cond, shift in enumerate(boundaries):
        axis = abs(shift)
        bnd  = 'min' if shift > 0 else 'max'
        bc   = bcs.find(x, axis, bnd)
        
        if bc is None:
            continue
        elif bc.condition_type == 'Dirichlet':
            code           = '0'
            condition_type = 'Dirichlet'
            return code, condition_type
        else:
            expr = bc.condition
            if ut.isnum(expr):
                if expr > 0:
                    chunk = f' +{bc.condition}'
                else:
                    chunk = f' {bc.condition}'
            else:
                chunk = f' {bc.condition}'
            
            if type(code) == str:
                code += chunk
    
    code = code.strip()
    if not code:
        code           = '0'
        condition_type = 'Inferred Neumann'
        
    return code, condition_type
    
###############################################################################
#Supporting Functions
###############################################################################
def make_reaction_template(rxn, namespaces: list[str]):
    
    namespaces = set(namespaces)
    
    def repl(match):
        name = match[0]
        if name in namespaces:
            if name in rxn.rev_namespace:
                field = '{voxel_num1}'
            else:
                field = '{voxel_num0}'
                
            indexed = name + f'[{field}]'
            return indexed
        else:
            return name
            

    namespaces = '|'.join(namespaces)
    pattern    = f'(^|\W)({namespaces})(\W|$)'
    pattern    = '[a-zA-Z_][a-zA-Z0-9_.]*|[0-9]e-?[0-9]'
    template   = re.sub(pattern, repl, rxn.rate)
    
    return template

def check_compartment(rxn_name, state_lst, compartments):
    if not state_lst:
        return None
    
    curr_cpt      = compartments.locate_state(state_lst[0])
    curr_cpt_name = curr_cpt.name
    
    for state in state_lst[1:]:
        new_cpt_name = compartments.locate_state(state).name
        
        if new_cpt_name != curr_cpt_name:
            r   = f'reaction "{rxn_name}"'
            s   = f'{state_lst}'
            msg = f'state {s} for {r} are in different compartments.'
            raise ValueError(msg)    
    
    return curr_cpt.domain_type
            

def map_voxel_domain_type(shape_stack, spatial_data):
    geometry_data = spatial_data['geometry']
    gdefs         = geometry_data['geometry_definitions']
    voxels        = shape_stack.voxels
    voxel2dmnt    = {}
    ndims         = shape_stack.ndims
    shifts        = [i for i in range(-ndims, ndims+1) if i != 0]
    
    for voxel_num, (voxel, neighbours) in enumerate(voxels.items()):
        #Extract shape name
        shape_name = shape_stack.get_shape(voxel)
        dmnt       = gdefs[shape_name].domain_type
        
        temp       = {}
        boundaries = []
        
        for shift in shifts:
            if shift not in neighbours:
                boundaries.append(shift)
                continue
            
            shift_neighbours = neighbours[shift]
            temp.setdefault(shift, [])
            
            for neighbour in shift_neighbours:
                #Extract shape name
                shape_name     = shape_stack.get_shape(neighbour)
                neighbour_dmnt = gdefs[shape_name].domain_type
                
                temp[shift].append(neighbour_dmnt)
                
                #Determine boundary
                if dmnt != neighbour_dmnt:
                    boundaries.append(shift)
                
        #Update result
        voxel2dmnt[voxel] = {'domain_type' : dmnt, 
                             'neighbours'  : temp,
                             'boundaries'  : boundaries,
                             'voxel_num'   : voxel_num
                             }
        
    return voxel2dmnt

def map_compartments_to_domain_types(spatial_data):
    cpts   = spatial_data['model']['compartments']
    rts    = spatial_data['model'].get('rates', {})
    dmnt2x = {}
    
    for cpt in cpts.values():
        dmnt2x.setdefault(cpt.domain_type, [])
        
        for x in cpt.namespace:
            if x in rts:
                continue
            dmnt2x[cpt.domain_type].append(x)
    
    return dmnt2x
    
