import numpy as np
import re

import dunlin.utils as ut
from .csgnode import parse_node 
from .utils   import add_coordinate
from .stack   import ShapeStack
from .voxel   import make_grids_from_config

class MassTransferMap:
    def __init__(self, 
                 spatial_data
                ):
        stk = ShapeStack.from_spatial_data(spatial_data)
        
        self.stack = stk
        
        shapes     = stk.shapes
        shape2dmnt = {shape.name: shape.domain_type for shape in shapes}
        voxel2dmnt = {voxel: shape2dmnt[shape] for voxel, shape in stk.voxel2shape.items()}
        
        
        
        dmnt_mappings = map_domain_types(spatial_data)
        
        self.dmnt2x    = dmnt_mappings[0]
        self.dmnt2rt   = dmnt_mappings[1]
        self.dmnt2rxn  = dmnt_mappings[2]
        self.dmnt2bc   = dmnt_mappings[3]
        self.voxel2dmnt = voxel2dmnt
        self.advection = {}
        self.diffusion = {}
        self.shifts    = [i for i in range(-stk.ndims, stk.ndims+1) if i != 0]
        
        # voxel2xidx, x2idx, diffs = map_to_indices(dmnt_mappings[0], 

        idx_mappings = map_to_indices(self.dmnt2x, 
                                      self.dmnt2rxn,
                                      voxel2dmnt,
                                      stk, 
                                      spatial_data['geometry']
                                      )
        
        self.voxel2xidx        = idx_mappings[0]
        self.x2idx             = idx_mappings[1]
        self.diffs             = idx_mappings[2]
        self.voxel2bulkrxn_idx = idx_mappings[3]
        self.voxel2bndrxn_idx  = idx_mappings[4]
        self.e_voxels          = idx_mappings[5] 
        self.spatial_data      = spatial_data
        self.rxn_cache         = {}
        self.rxn_templates     = make_reaction_templates(spatial_data)
        self.voxel_calcs       = {voxel: {} for voxel in self.stack.voxels}
        
    @property
    def voxels(self) -> dict:
        return self.stack.voxels
    
    @property
    def sizes(self) -> dict:
        return self.stack.sizes
    
    def index(self, name, coordinates):
        return add_coordinate(name, coordinates)
    
    def get_rxn_template(self, rxn_name):
        if rxn_name in self.rxn_cache:
            return self.rxn_cache[rxn_name]
        
        
        rxn       = self.spatial_data['model']['reactions'][rxn_name]
        variables = '|'.join(rxn.stoichiometry())
        pattern   = f'(^|\W)({variables})(\W|$)'

        def repl(match):
            var     = match[2]
            indexed = match[1] + var + '{}' + match[3]
            return indexed


        result = re.sub(pattern, repl, rxn.rate)
        
        return result
          
    
    def make_state_mapping(self):
        e_voxels = self.e_voxels
        mapping  = {}
        for voxel, xidx in self.voxel2xidx.items():
            number = e_voxels[voxel]
    
    def apply_boundary_conditions(self, dmnt, shift):
        result = {}
        bcs = self.dmnt2bc[dmnt]
        xs  = self.dmnt2x[dmnt]
        
        for x in xs:
            if x in bcs:
                bc = bcs[x]
                
                if bc.condition_type == 'Neumann':
                    result[x] = str(bc.condition)
                else:
                    result[x] = 'Dirichlet'
        return result
    
    
    def _apply_bulk_reaction(self, voxel):
        diffs          = self.diffs
        dmnt           = self.voxel2dmnt[voxel]
        rxns           = self.dmnt2rxn[dmnt]
        voxel2xidx     = self.voxel2xidx
        x_xidx         = voxel2xidx[voxel]
        rxn_calcs      = self.voxel_calcs[voxel].setdefault('reaction', {})
        voxel_num      = self.e_voxels[voxel]
        rxn_templates  = self.rxn_templates
        rts            = self.dmnt2rt[dmnt]
        
        #Bulk reactions
        for rxn in rxns:
            rxn_name, rxn_rate = rxn_templates[rxn.name]
            lhs  = rxn_name.format(voxel_num=voxel_num)
            rhs  = rxn_rate.format(**x_xidx)
            line = f'{lhs} = {rhs}'
            
            rxn_calcs[rxn.name] = line
            
            for x in rxn.stoichiometry:
                xidx  = x_xidx[x]
                coeff = rxn.stoichiometry[x]
                chunk = f'{coeff}*{lhs}'
                
                if type(diffs[xidx]) == list:
                    diffs[xidx].append(chunk)
                
            
        #Rates
        for x, rt in rts.items():
            xidx  = x_xidx[x]
            lhs   = ut.diff(xidx)
            rhs   = rt.expr.format(**x_xidx)
            
            diffs[xidx] = rhs    
    
    def _apply_bulk_mass_transfer(self, voxel, neighbour, shift):
        diffs          = self.diffs
        voxel2xidx     = self.voxel2xidx
        advs           = self.spatial_data['model']['advection']
        dfns           = self.spatial_data['model']['diffusion']
        size           = self.sizes[voxel]
        neighbour_size = self.sizes[neighbour]
        dist           = size + neighbour_size
        area           = size**2
        adv_sign       = '-' if shift > 0 else '+'
        x_xidx         = voxel2xidx[voxel]
        
        for x, xidx in x_xidx.items():
            if x not in advs:
                continue
            
            #Get concentrations
            conc           = xidx
            neighbour_conc = voxel2xidx[neighbour][x]
            
            #Advection
            velocity = advs[x][shift]
            vol_flow = f'{velocity}*{area}'
            a_chunk  = f'{adv_sign}{conc}*{vol_flow}'
            
            #Diffusion
            conc_diff  = f'({neighbour_conc} - {conc})'
            conc_grad  = f'{conc_diff}/{dist}' 
            coeff      = dfns[x][shift]
            d_chunk    = f'+{coeff}*{conc_grad}'
            
            
            if type(diffs[xidx]) == list:
                diffs[xidx].extend([a_chunk, d_chunk])
        
        
    def apply_internal_boundary_conditions(self, 
                                           voxel, 
                                           neighbour, 
                                           shift, 
                                           diffs: dict
                                           ):
        voxel2xidx     = self.voxel2xidx 
        bcs            = self.spatial_data['geometry'].get('boundary_conditions', {})
        dmnt           = self.voxel2dmnt[voxel]
        neighbour_dmnt = self.voxel2dmnt[neighbour]
        rxns     = self.dmnt2rxn[frozenset([dmnt, neighbour_dmnt])]
        
        for x, idx in voxel2xidx[voxel].items():
            pass
            
        
            

###############################################################################
#Supporting Functions
###############################################################################
def map_domain_types(spatial_data):
    cpts     = spatial_data['model']['compartments']
    rts      = spatial_data['model']['rates']
    rxns     = spatial_data['model']['reactions']
    bcs      = spatial_data['geometry'].get('boundary_conditions', {})
    
    dmnt2rt  = {}
    dmnt2rxn = {} 
    dmnt2bc  = {} 
    dmnt2x   = {}
    x2dmnt   = {}
    
    #Map xs, rts
    for cpt_name, cpt in cpts.items():
        dmnt = cpt.domain_type
        xs   = cpt.namespace
        
        dmnt2x[dmnt] = list(xs)
        
        for x in xs:
            if x in rts.states:
                dmnt2rt.setdefault(dmnt, {})[x] = rts[x]
            
            x2dmnt[x] = dmnt
    
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
            rxndmnt = frozenset([prods_dmnt_name, prods_dmnt_name])
            rxndmnt = prods_dmnt_name
        elif prods_dmnt_name is None:
            rxndmnt = frozenset([rcts_dmnt_name, rcts_dmnt_name])
            rxndmnt = rcts_dmnt_name
        else:
            rxndmnt = frozenset([rcts_dmnt_name, prods_dmnt_name])
        
        dmnt2rxn.setdefault(rxndmnt, []).append(rxn)
    
    #Map bcs
    for bc_name, bc in bcs.items():
        x    = bc.state
        dmnt = x2dmnt[x]
        dmnt2bc.setdefault(dmnt, {})[x] = bc
        
    
    return dmnt2x, dmnt2rt, dmnt2rxn, dmnt2bc, x2dmnt

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

from collections import Counter

def map_to_indices(dmnt2x, dmnt2rxn, voxel2dmnt, shape_stack, geometry_data):
    #Voxel enumeration
    e_voxels  = {}
    
    #State indexing
    voxel2xidx = {}
    x2idx      = {}
    i          = 0
    diffs      = {}
    
    #Variable indexing
    # vrbs       =  
    voxel2vidx = {}
    
    #Reaction indexing
    voxel2bulkrxn_idx = {}
    voxel2bndrxn_idx  = {}
    seen_rxns         = Counter()
    
    for voxel_num, (voxel, neighbours) in enumerate(shape_stack.voxels.items()):
        #Enumerate the voxels
        e_voxels[voxel] = voxel_num
        
        dmnt = voxel2dmnt[voxel]
        
        #Index the states
        xs                = dmnt2x[dmnt]
        # voxel2xidx[voxel] = {x: i for i, x in enumerate(xs, start=i)}
        
        for ii, x in enumerate(xs, start=i):
            xidx = f'{x}__{voxel_num}'
            voxel2xidx.setdefault(voxel, {})[x] = xidx
            x2idx.setdefault(x, []).append(voxel_num)
            diffs[xidx] = []
        
        i += len(xs)
        
        #index the variables
        voxel2vidx[voxel_num] = {v: f'{v}__{voxel_num}' for v in vrbs}
        
        #Index the reactions
        rxns = dmnt2rxn.get(dmnt, [])
        if rxns:
            voxel2bulkrxn_idx[voxel] = seen_rxns[dmnt]
            seen_rxns[dmnt] += 1
            
        for shift, shift_neighbours in neighbours.items():
            for neighbour in shift_neighbours:
                
                pair = frozenset([voxel, neighbour])
                if pair in voxel2bndrxn_idx:
                    continue
                
                #Get reactions at boundaries
                neighbour_dmnt = voxel2dmnt[neighbour]
                key            = frozenset([dmnt, neighbour_dmnt])
                rxns           = dmnt2rxn.get(key, [])
                
                if rxns:
                    temp = {}
                    for rxn in rxns:
                        temp[rxn.name]       = seen_rxns[rxn.name]
                        seen_rxns[rxn.name] += 1
                        
                    voxel2bndrxn_idx[pair] = temp
    
    return voxel2xidx, x2idx, diffs, voxel2bulkrxn_idx, voxel2bndrxn_idx, e_voxels

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

def make_reaction_templates(spatial_data):
    def repl(match):
        name    = match[2]
        indexed = match[1] + '{' + name + '}' + match[3]
        return indexed

    model_data = spatial_data['model']
    reactions  = model_data['reactions']
    states     = list(model_data['states'].names)
    variables  = list(model_data['variables']) if 'variables' in model_data else []
    names      = states + variables
    names      = '|'.join(names)
    pattern   = f'(^|\W)({names})(\W|$)'
    templates  = {}
    
    for rxn in reactions.values():
        result   = re.sub(pattern, repl, rxn.rate)
        rxn_name = f'{rxn.name}__{{voxel_num}}' 
        
        templates[rxn.name] = rxn_name, result
        
    return templates
