import matplotlib.pyplot as plt

import addpath
import dunlin as dn
import dunlin.utils as ut
import dunlin.ode.ode_coder as odc
from dunlin.datastructures.spatial import SpatialModelData
from dunlin.spatial.geometrydefinition.stack import (ShapeStack,
                                                     )
plt.close('all')
plt.ion()


def map_domains_to_shape(shape_stack : ShapeStack, 
                         spatial_data: SpatialModelData
                         ):
    domain_types = spatial_data['geometry']['domain_types']
    shapes       = shape_stack.shapes[::-1]
    domain2shape = {}
    seen         = set() 
    
    for dmnt_name, dmnt in domain_types.items():
        for dmn_name, internal_points in dmnt.domains.items():
            for i, shape in enumerate(shapes):
                if all(shape.contains_points(internal_points)):
                    if i in seen:
                        msg = 'Multiple domains map to the {shape}.'
                        raise ValueError(msg)
                    else:
                       domain2shape[dmn_name] = i
                       break
                       
            if dmn_name not in domain2shape:
                msg = f'Could not associate domain {dmn_name} with a shape.'
                raise ValueError(msg)
    
    return domain2shape

def map_reactions_to_domain_types(spatial_data) -> tuple[dict, dict]:
    reactions    = spatial_data['model']['reactions']
    compartments = spatial_data['model']['compartments']
    
    def check_compartment(rxn_name, state_lst):
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
                
        #Map to domain type
        dmnt_name = curr_cpt.domain_type
        
        return dmnt_name
    
    dmnt2rxn = {}
    rxn2dmnt = {}
    for rxn_name, rxn in reactions.items():
        rcts  = rxn.reactants
        prods = rxn.products
        
        #Extract domain_type from compartment
        rcts_dmnt_name  = check_compartment(rxn_name, rcts)
        prods_dmnt_name = check_compartment(rxn_name, prods)
        
        #Create domain_type pairs
        #Rxns across domain types: dmnt0 are dmnt1 different
        #Rxns within domain type: dmnt0 and dmnt1 are the same
        if rcts_dmnt_name is None and prods_dmnt_name is None:
            msg = 'Could not map reactants or products to a compartment.'
            raise ValueError(msg)  
        elif rcts_dmnt_name is None:
            rxndmnt = prods_dmnt_name, prods_dmnt_name
        elif prods_dmnt_name is None:
            rxndmnt = rcts_dmnt_name, rcts_dmnt_name
        else:
            rxndmnt = rcts_dmnt_name, prods_dmnt_name
        
        rxn2dmnt[rxn_name] = rxndmnt
        dmnt2rxn.setdefault(rxndmnt, []).append(rxn_name)
    
    return dmnt2rxn, rxn2dmnt

class SpatialMapping:
    def __init__(self, spatial_data):
        stk = ShapeStack.from_geometry_data(geometry_data)
        
        
        cpts   = spatial_data.model.compartments
        dmnt2x = {}
        
        for cpt_name, cpt in cpts.items():
            pass
            
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

def map_domain_types(spatial_data):
    cpts     = spatial_data['model']['compartments']
    rts      = spatial_data['model']['rates']
    rxns     = spatial_data['model']['reactions']
    bcs      = spatial_data['geometry']['boundary_conditions']
    
    dmnt2rt  = {}
    dmnt2rxn = {} 
    dmnt2bc  = {} 
    dmnt2x   = {}
    
    #Map xs, rts
    for cpt_name, cpt in cpts.items():
        dmnt = cpt.domain_type
        xs   = cpt.namespace
        
        dmnt2x[dmnt] = list(xs)
        
        for x in xs:
            if x in rts.states:
                dmnt2rt.setdefault(dmnt, {})[x] = rts[x]
        
    #Map rxns
    for rxn_name, rxn in rxns.items():
        rcts  = rxn.reactants
        prods = rxn.products
        #Extract domain_type from compartment
        rcts_dmnt_name  = check_compartment(rxn_name, rcts, cpts)
        prods_dmnt_name = check_compartment(rxn_name, prods, cpts)
        
        #Create domain_type pairs
        #Rxns across domain types: dmnt0 are dmnt1 different
        #Rxns within domain type: dmnt0 and dmnt1 are the same
        if rcts_dmnt_name is None and prods_dmnt_name is None:
            r   = f'"{rxn_name}"'
            msg = f'Could not map reactants/products of {r} to a compartment.'
            raise ValueError(msg)  
        elif rcts_dmnt_name is None:
            rxndmnt = prods_dmnt_name, prods_dmnt_name
        elif prods_dmnt_name is None:
            rxndmnt = rcts_dmnt_name, rcts_dmnt_name
        else:
            rxndmnt = rcts_dmnt_name, prods_dmnt_name
        
        dmnt2rxn.setdefault(rxndmnt, []).append(rxn)
    
    #Map bcs
    for bc_name, bc in bcs.items():
        dmnt = bc.domain_type
        x    = bc.state
        dmnt2bc.setdefault(dmnt, {})[x] = bc
        
    
    return dmnt2x, dmnt2rt, dmnt2rxn, dmnt2bc
        

def point2domain_type(point, shape_stack, geometry_data, cache):
    if point in cache:
        return cache[point]
    
    shape      = shape_stack.get_shape(point)
    
    if shape is None:
        return None
    
    shape_name = shape.name
    gdefs      = geometry_data['geometry_definitions']
    dmnt       = gdefs[shape_name].domain_type
        
    return dmnt

def index_states(dmnt2x, shape_stack, geometry_data, cache):
    x_idx = {}
    i     = 0
    for point, neighbours in shape_stack.graph.items():
        dmnt = point2domain_type(point, shape_stack, geometry_data, cache)
        
        if dmnt is None:
            continue

        #Index the states
        xs           = dmnt2x[dmnt]
        x_idx[point] = dict(enumerate(xs, start=i))
        
        i += len(xs)
    
    return x_idx

def funcs2code(spatial_data):
    funcs = spatial_data['model']['functions']
    
    if not funcs:
        return ''
    
    #Set up boilerplate for vrbs/funcs
    code = '\t#Functions\n'
    for func in funcs.values():
        definition = f'\t\tdef {func.name}({func.signature}):\n'
        expr       = f'\t\t\treturn {func.expr}'
        code      += f'{definition}{expr}\n'

    return code +'\n'

def vrbs2code(spatial_data):
    vrbs  = spatial_data['model']['variables']
    
    if not vrbs:
        return ''

    code = '\t#Variables\n'
    for vrb in vrbs.values():
        code += f'\t{vrb.name} = {vrb.expr}\n'
    
    return code +'\n'

def rxns2code(rxns: list):
    if not rxns:
        return {}, ''
    
    diffs = {}
    code  = '\t#Reactions\n'
    for rxn in rxns:
        code += f'\t{rxn.name} = {rxn.rate}\n'
        
        for x, n in rxn.stoichiometry.items():
            diffs.setdefault(x, '') 
            diffs[x] += f'{n}*{rxn.name} '
    
    return diffs, code +'\n'

def advs2code(x_idx, point, shift, neighbour, advs, diffs):
    
    xs    = x_idx[neighbour]
    code  = ''
    
    for neighbour_idx, x_name in xs.items():
        if x_name not in advs:
            continue
        
        adv = advs[x_name]
        
        if shift > 0:
            coeff   = adv[shift]
            indexed = f'states[{neighbour_idx}]'
            
            
        
        
        if shi
        lhs     = ut.adv(f'{shift}_{x_name}') if shift > 0 else ut.adv(f'_{shift}_{x_name}')
        expr    = f'\t{lhs} = {coeff}*{indexed}\n'
        
        code += expr
        
        diffs.setdefault(x_name, '') 
        if shift > 0:
            diffs[x_name] += f'-({lhs}) '
        else:
            diffs[x_name] += f'+({lhs}) '
    
    return diffs, code + '\n'

def internals2code(x_idx, xs, rts, rxns, bcs):
    diffs1 = odc._rts2code(rts)
    

all_data = dn.read_dunl_file('spatial_0.dunl')

mref = 'M0'
gref = 'Geo0'
ref  = mref, gref

spatial_data  = SpatialModelData.from_all_data(all_data, mref, gref)
geometry_data = spatial_data['geometry'] 

shape_stack = ShapeStack.from_geometry_data(geometry_data)
main_grid   = shape_stack.grid
shapes      = shape_stack.shapes
gdata       = geometry_data

dmnt2x, dmnt2rt, dmnt2rxn, dmnt2bc = map_domain_types(spatial_data)
point2domain_type_cache = {}

x_idx = index_states(dmnt2x, shape_stack, geometry_data, point2domain_type_cache)

all_diffs = {}

funcs_code = funcs2code(spatial_data)
vrbs_code  = vrbs2code(spatial_data)


advs = spatial_data['model']['advection']

for point, neighbours in shape_stack.graph.items():
    curr_dmnt = point2domain_type(point, shape_stack, gdata, point2domain_type_cache)
    
    if curr_dmnt is None:
        continue
    
    section_code = f'\t#Point {point}\n{vrbs_code}'
    
    #Set up differentials
    diffs = {}
    
    #Get internals
    xs   = dmnt2x.get(curr_dmnt, [])
    rts  = dmnt2rt.get(curr_dmnt, {})
    rxns = dmnt2rxn.get((curr_dmnt, curr_dmnt), [])
    bcs  = dmnt2bc.get(curr_dmnt, {})
    
    for x_name, rt in rts.items():
        diffs[x_name] = rt.expr
    
    rxn_diffs, rxns_code = rxns2code(rxns)
    
    section_code += rxns_code
    diffs.update(rxn_diffs)
    
    #Get transfers/boundary conditions
    for shift, neighbour in neighbours.items():
        next_dmnt = point2domain_type(neighbour, 
                                      shape_stack, 
                                      gdata, 
                                      point2domain_type_cache
                                      )
        if not next_dmnt:
            continue
        
        #Check if this edge is a boundary
        is_boundary = curr_dmnt != next_dmnt
        
        if is_boundary:
            #Priority: bc, transport rxn
            pass
        else:
            #Advection
            diffs, adv_code  = advs2code(x_idx, shift, neighbour, advs, diffs)
            section_code    += adv_code
            print(diffs)
            print(section_code)
            assert False
            pass
            
            #Diffusion
            
            
            
        
        
        
    
    # for x_name in xs:
    #     if x_name in bcs:
    #         bc             = bcs[x_name]
    #         condition_type = 'condition_type'
    #         if condition_type == 'Neumann':
    #             diffs[x_name] = str(bc.condition)
    #         elif condition_type == 'Dirichlet':
    #             diffs[x_name] = '0'
    #         else:
    #             msg = f'No code implementation for {condition_type}'
    #             raise NotImplementedError(msg)
        
    # for shift, neighbour in neighbours.items():
    #     pass
    
    all_diffs[point] = diffs
    
print(all_diffs[5, 5])
        