import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from matplotlib.gridspec import GridSpec
from matplotlib.patches  import Rectangle, FancyArrowPatch
from seaborn             import xkcd_rgb
from numba       import njit
from numba.core  import types
from numba.typed import Dict

import addpath
import dunlin       as dn 
import dunlin.utils as ut
from dunlin.spatial.grid.grid          import make_grids_from_config
from dunlin.spatial.geometrydefinition import make_shapes
from dunlin.spatial.grid.stack         import Stack
from dunlin.spatial.shape_stack        import ShapeStack
from dunlin.datastructures.spatial     import SpatialModelData
from spatial_data0                     import all_data

#Set up
plt.close('all')
plt.ion()

spatial_data = SpatialModelData.from_all_data(all_data, 'M0')
shapes       = make_shapes(spatial_data.geometry_definitions)
grids        = make_grids_from_config(spatial_data.grid_config)
main_grid    = next(iter(grids.values()))
stk0         = Stack(main_grid, spatial_data)

fig0  = plt.figure(figsize=(10, 10))
span = -1, 11

ax = fig0.add_subplot(1, 1, 1)
AX = [ax]
ax.set_box_aspect()
ax.set_xlim(*span)
ax.set_ylim(*span)


shape_args = {'facecolor': {'cell'  : 'coral',
                            'field' : 'steel'
                            },
              }
stk0.plot_voxels(ax, shape_args=shape_args)

#Dummy Class for Testing
class Dummy:
    def __init__(self, name):
        self.name = name
        self.data = {}
      
    def __getitem__(self, key):
        mean = np.mean(np.array(key))
        return self.data.setdefault(key, mean)
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __repr__(self):
        return f'{type(self).__name__}({self.name})'
    
###############################################################################
#Parse Functions
###############################################################################
print('Parse Functions')
code = ShapeStack._parse_functions(spatial_data)

print(code)

scope = {}
exec(tw.dedent(code), scope)
func0 = scope['func0']
assert func0(2, 3) == -6
print()

###############################################################################
#Map Shapes, Domains and Domain Types
###############################################################################
temp = ShapeStack._map_shape2domain(spatial_data, *shapes)

shape2domain, domain2shape, shape2domain_type, domain_type2shape  = temp

assert shape2domain['field'] == 'medium0'
assert shape2domain['cell' ] == 'cytosol0'

###############################################################################
#Map Variables and Domain Types
###############################################################################
print('Map Variables and Domain Types')
'''
Note:
    These tests only check the keys. The values depend on the requirement of 
    downstream computation so I'm not enforcing any particular format as of now.
    
'''
temp = ShapeStack._map_variable2domain_type(spatial_data)

domain_type2variable, variable2domain_type = temp

# print(domain_type2variable)
assert len(domain_type2variable) == 3
assert 'cytosolic'                               in domain_type2variable
assert None                                      in domain_type2variable
assert frozenset(['cytosolic', 'extracellular']) in domain_type2variable

assert len(variable2domain_type) == 4
assert 'vrb0' in variable2domain_type
assert 'vrb1' in variable2domain_type
assert 'vrb2' in variable2domain_type 
assert 'vrb3' in variable2domain_type

print()

###############################################################################
#Parse Variables
###############################################################################
print('Variable overhead')
variable_overhead = ShapeStack._make_variable_overhead(variable2domain_type, 
                                                       spatial_data
                                                       )
print(variable_overhead)
print()

print('Parse variables')
voxel               = (4.75, 4.25)
seen_variable_items = set()

variable_code = ShapeStack._parse_variables(voxel, 
                                            stk0.voxels, 
                                            shape2domain_type, 
                                            domain_type2variable, 
                                            seen_variable_items, 
                                            spatial_data
                                            )

print(variable_code)
scope = {'func0'  : lambda a, b: 2,
         'k_synH' : 0,
         'A'      : Dummy('A'),
         'B'      : Dummy('B'),
         'C'      : Dummy('C')
         } 
exec(tw.dedent(variable_overhead + variable_code), scope)

assert scope['vrb0'][(4.75, 4.25)] == 2
k = ShapeStack._combine_voxels((4.75, 3.75), (4.75, 4.25))
assert scope['vrb1'][k] == 19.125
assert scope['vrb3'][k] == 38.25

k = ShapeStack._combine_voxels((4.25, 4.25), (4.75, 4.25))
assert scope['vrb1'][k] == 19.125
assert scope['vrb3'][k] == 38.25

#Test another voxel
voxel = 4.25, 4.25

variable_code = ShapeStack._parse_variables(voxel, 
                                            stk0.voxels, 
                                            shape2domain_type, 
                                            domain_type2variable, 
                                            seen_variable_items, 
                                            spatial_data
                                            )

print(variable_code)

scope['vrb0'][(4.25, 4.75)] = 1

exec(tw.dedent(variable_code), scope)

k = ShapeStack._combine_voxels((4.25, 4.25), (4.25, 4.75))
assert scope['vrb1'][k] == 19.125
assert scope['vrb3'][k] == 19.125

print()

###############################################################################
#Map Reactions and Domain Types
###############################################################################
temp = ShapeStack._map_reaction2domain_type(variable2domain_type, spatial_data)

domain_type2reaction, reaction2domain_type = temp

assert len(domain_type2reaction) == 2
assert 'cytosolic'                               in domain_type2reaction
assert frozenset({'extracellular', 'cytosolic'}) in domain_type2reaction

###############################################################################
#Parse Reactions
###############################################################################
print('Reaction overhead')
reaction_overhead = ShapeStack._make_reaction_overhead(reaction2domain_type)

print(reaction_overhead)
print()

voxel               = (4.75, 4.25)
seen_reaction_items = set()
diff_terms          = {}

print('Parse reactions')
reaction_code = ShapeStack._parse_reactions(voxel, 
                                            stk0.voxels, 
                                            shape2domain_type, 
                                            domain_type2variable, 
                                            domain_type2reaction, 
                                            seen_reaction_items, 
                                            diff_terms, 
                                            spatial_data
                                            )

print(reaction_code)
scope = {'k_synH' : 2,
         'k_synB' : 4,
         'vrb2'   : 3,
         'vrb1'   : Dummy('vrb1'),
         'A'      : Dummy('A'),
         'H'      : Dummy('H'),
         'C'      : Dummy('C')
         }

exec(tw.dedent(reaction_overhead + reaction_code), scope)

assert scope['synH'][(4.75, 4.25)] == 9
k = ShapeStack._combine_voxels((4.75, 3.75), (4.75, 4.25))
assert scope['synB'][k] == 33.9375
k = ShapeStack._combine_voxels((4.25, 4.25), (4.75, 4.25))
assert scope['synB'][k] == 33.9375

#Test another voxel
voxel = (4.25, 4.25)

reaction_code = ShapeStack._parse_reactions(voxel, 
                                            stk0.voxels, 
                                            shape2domain_type, 
                                            domain_type2variable, 
                                            domain_type2reaction, 
                                            seen_reaction_items, 
                                            diff_terms, 
                                            spatial_data
                                            )

print(reaction_code)
scope['H'][(4.75, 4.25)] = 0
exec(tw.dedent(reaction_code), scope)
#The values of the previous group of assert statements will chenge
#if  double-processing has occured
assert scope['synH'][(4.75, 4.25)] == 9
k = ShapeStack._combine_voxels((4.75, 3.75), (4.75, 4.25))
assert scope['synB'][k] == 33.9375
k = ShapeStack._combine_voxels((4.25, 4.25), (4.75, 4.25))
assert scope['synB'][k] == 33.9375

k = ShapeStack._combine_voxels((4.25, 4.25), (4.25, 4.75))
assert scope['synB'][k] == 33.9375

print('Check diff_terms')
for k, v in diff_terms.items():
    print(k)
    print(v)
    print()

diff_code = '\n'.join(diff_terms.values())
scope     = {'synH' : Dummy('synH'),
             'synB' : Dummy('synB'),
             ut.diff('H') : {},
             ut.diff('B') : {},
             ut.diff('A') : {}
             }

scope['synB'][(4.75, 3.75), (4.75, 4.25)] = 2
scope['synB'][(4.25, 4.25), (4.75, 4.25)] = 3
scope['synB'][(4.25, 4.25), (4.25, 4.75)] = 4

exec(tw.dedent(diff_code), scope)
assert scope[ut.diff('H')][4.75, 4.25] == -0.5
assert scope[ut.diff('H')][4.25, 4.75] == -4

print()

print('Interim test: Reactions and variables')
diff_terms          = {}
seen_variable_items = set()
seen_reaction_items = set()
function_code       = ShapeStack._parse_functions(spatial_data)
variable_code       = ShapeStack._make_variable_overhead(variable2domain_type, 
                                                         spatial_data
                                                         )
reaction_code       = ShapeStack._make_reaction_overhead(reaction2domain_type)

for voxel in stk0.voxels:
    variable_code += ShapeStack._parse_variables(voxel,
                                                 stk0.voxels,
                                                 shape2domain_type, 
                                                 domain_type2variable, 
                                                 seen_variable_items, 
                                                 spatial_data
                                                 )
    
    reaction_code += ShapeStack._parse_reactions(voxel, 
                                                 stk0.voxels, 
                                                 shape2domain_type, 
                                                 domain_type2variable, 
                                                 domain_type2reaction, 
                                                 seen_reaction_items, 
                                                 diff_terms, 
                                                 spatial_data
                                                 )

code = function_code +  variable_code + reaction_code

with open('output_reaction_code.txt', 'w') as file:
    file.write(code)

scope = {'H' : Dummy('H'),
         'A' : Dummy('A'),
         'B' : Dummy('B'),
         'C' : Dummy('D'),
         'k_synH'   : 1,
         'k_synB'   : 1,
         'k_synH_A' : 1,
         
         'J_A'   : 1,
         'J_B_x' : 1,
         'J_B_y' : 1,
         
         'F_A'   : 0,
         'F_B_x' : 1,
         'F_B_y' : 1
         }

exec(tw.dedent(code), scope)

fig1 = plt.figure(figsize=(12, 9))
gs   = GridSpec(6, 8, top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.8, wspace=0.8)

layout = [[0, 2, 0, 2],
          [0, 2, 2, 4],
          [0, 2, 4, 6],
          [0, 2, 6, 8],
          [2, 6, 0, 4],
          [2, 6, 4, 8]
          ]

for a in layout:
    ax = fig1.add_subplot(gs[a[0]: a[1], a[2]: a[3]])
    
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    AX.append(ax)
    plt.grid(True)
    

def get_color(value):
    if value < 0:
        return xkcd_rgb['pastel blue']
    elif value == 0:
        return xkcd_rgb['pastel green']
    else:
        return xkcd_rgb['pastel red']

vrbs = ['vrb0', 'vrb1', 'vrb2', 'vrb3']

i = 1
for vrb in vrbs:
    AX[i].set_title(vrb)
    values = scope[vrb]
    if type(values) != dict:
        AX[i].annotate('Global', xy=(5,5))
        i += 1
        continue
    
    for loc, value in scope[vrb].items():
        if type(loc[0]) == tuple:
            voxel0, voxel1 = loc
            avr = [(voxel0[0] + voxel1[0])/2], [(voxel0[1] + voxel1[1])/2]
            AX[i].scatter(*avr, 8, color=get_color(value))
        else:
            voxel  = loc
            size   = stk0.voxels[voxel]['size']
            anchor = voxel[0] - size/2, voxel[1] - size/2    
            patch  = Rectangle(anchor, size, size, facecolor=get_color(value))
            AX[i].add_patch(patch)
    
    i += 1

rxns = ['synH', 'synB']

for rxn in rxns:
    AX[i].set_title(rxn)
    values = scope[rxn]
    
    for loc, value in scope[rxn].items():
        if type(loc[0]) == tuple:
            voxel0, voxel1 = loc
            avr = [(voxel0[0] + voxel1[0])/2], [(voxel0[1] + voxel1[1])/2]
            AX[i].scatter(*avr, 32, color=get_color(value))
        else:
            voxel  = loc
            size   = stk0.voxels[voxel]['size']
            anchor = voxel[0] - size/2, voxel[1] - size/2    
            patch  = Rectangle(anchor, size, size, facecolor=get_color(value))
            AX[i].add_patch(patch)
    
    i += 1
    
###############################################################################
#Template the Mas Transfer
###############################################################################
mass_transfer_templates = ShapeStack._template_mass_transfer(variable2domain_type, 
                                                              spatial_data
                                                              )

print(mass_transfer_templates)
assert 'A' in mass_transfer_templates
assert 'B' in mass_transfer_templates
assert 'H' in mass_transfer_templates
assert 'C' not in mass_transfer_templates

###############################################################################
#Parse Mass Transfer
###############################################################################
print('Mass transfer overhead')
mass_transfer_overhead = ShapeStack._make_mass_transfer_overhead(mass_transfer_templates)

print(mass_transfer_overhead)

voxel = (4.75, 4.25)

print('Parse mass transfer')
state = 'A'
seen_mass_transfer_items = set()
diff_terms               = {}   
mass_transfer_code = ShapeStack._parse_mass_transfer(state, 
                                                      voxel, 
                                                      stk0.voxels, 
                                                      mass_transfer_templates,
                                                      shape2domain_type, 
                                                      seen_mass_transfer_items,
                                                      diff_terms, 
                                                      )

print(mass_transfer_code)
scope = {'J_A'    : 2,
          'F_A'    : 0.5, 
          'A'      : Dummy('A'),
          '_dfn_A' : {},
          '_adv_A' : {},
          }

scope['A'][(4.75, 4.25)] = 10
scope['A'][(5.25, 4.25)] = 0

exec(tw.dedent(mass_transfer_overhead + mass_transfer_code), scope)
assert scope['_adv_A'][(4.75, 4.25), (5.25, 4.25)] == 2.5
assert scope['_adv_A'][(4.75, 4.25), (4.75, 4.75)] == 2.5
assert scope['_dfn_A'][(4.75, 4.25), (5.25, 4.25)] == 10
assert scope['_dfn_A'][(4.75, 4.25), (4.75, 4.75)] == 5.25
print()

#Test another voxel
voxel = (5.25, 4.25)

mass_transfer_code = ShapeStack._parse_mass_transfer(state, 
                                                      voxel, 
                                                      stk0.voxels, 
                                                      mass_transfer_templates,
                                                      shape2domain_type, 
                                                      seen_mass_transfer_items,
                                                      diff_terms, 
                                                      )


print(mass_transfer_code)
scope['A'][(4.75, 4.25)] = 0
exec(tw.dedent(mass_transfer_code), scope)
#The values of the previous group of assert statements will chenge
#if  double-processing has occured
assert scope['_adv_A'][(4.75, 4.25), (5.25, 4.25)] == 2.5
assert scope['_adv_A'][(4.75, 4.25), (4.75, 4.75)] == 2.5
assert scope['_dfn_A'][(4.75, 4.25), (5.25, 4.25)] == 10
assert scope['_dfn_A'][(4.75, 4.25), (4.75, 4.75)] == 5.25

assert scope['_adv_A'][(5.25, 4.25), (5.25, 4.75)] ==  0
assert scope['_dfn_A'][(5.25, 4.25), (5.25, 4.75)] == -5


print('Check diff_terms')
for k, v in diff_terms.items():
    print(k)
    print(v)
    print()

diff_code = '\n'.join(diff_terms.values())
scope     = {'_adv_A'      : Dummy('_adv_A'),
              '_dfn_A'     : Dummy('_dfn_A'),
              ut.diff('A') : {}
              }

exec(tw.dedent(diff_code), scope)
assert scope[ut.diff('A')][4.75, 4.25] == -18.5
assert scope[ut.diff('A')][5.25, 4.25] == -0.5
assert scope[ut.diff('A')][4.75, 4.75] == 9.25
assert scope[ut.diff('A')][5.25, 4.75] == 9.75

print()

#Test boundary conditions
print('Test boundary conditions')
voxel = (1, 1)
state = 'B'

seen_mass_transfer_items = set()
diff_terms               = {}   
mass_transfer_code = ShapeStack._parse_mass_transfer(state, 
                                                      voxel, 
                                                      stk0.voxels, 
                                                      mass_transfer_templates,
                                                      shape2domain_type, 
                                                      seen_mass_transfer_items,
                                                      diff_terms, 
                                                      )

print(mass_transfer_code)

scope = {'J_B_x'    : 2,
          'J_B_y'    : 3,
          'F_B_x'    : 5,
          'F_B_y'    : 7,
          'B'        : Dummy('B'),
          '_bc_B_1_' : Dummy(ShapeStack._make_boundary('B', -1)),
          '_bc_B_2_' : Dummy(ShapeStack._make_boundary('B', -2)),
          '_adv_B'   : Dummy('_adv_B'),
          '_dfn_B'   : Dummy('_dfn_B')
          }

exec(tw.dedent(mass_transfer_code), scope)

k = ShapeStack._make_boundary('B', -2)
assert scope[k][(1, 1)] == 36
k = ShapeStack._make_boundary('B', -1)
assert scope[k][(1, 1)] == 0

print('Interim test: Mass transfer')
diff_terms               = {}
seen_variable_items      = set()
seen_reaction_items      = set()
seen_mass_transfer_items = set()
function_code            = ShapeStack._parse_functions(spatial_data)
variable_code            = ShapeStack._make_variable_overhead(variable2domain_type, 
                                                              spatial_data
                                                              )
reaction_code       = ShapeStack._make_reaction_overhead(reaction2domain_type)
mass_transfer_code  = ShapeStack._make_mass_transfer_overhead(mass_transfer_templates)

for voxel, datum in stk0.voxels.items():
    variable_code += ShapeStack._parse_variables(voxel,
                                                 stk0.voxels,
                                                 shape2domain_type, 
                                                 domain_type2variable, 
                                                 seen_variable_items, 
                                                 spatial_data
                                                 )
    
    reaction_code += ShapeStack._parse_reactions(voxel, 
                                                 stk0.voxels, 
                                                 shape2domain_type, 
                                                 domain_type2variable, 
                                                 domain_type2reaction, 
                                                 seen_reaction_items, 
                                                 diff_terms, 
                                                 spatial_data
                                                 )
    
    shape       = datum['shape']
    domain_type = shape2domain_type[shape]
    states      = spatial_data.compartments.domain_type2state[domain_type]
    
    for state in states:
        if state not in mass_transfer_templates:
            continue
        
        mass_transfer_code += ShapeStack._parse_mass_transfer(state, 
                                                              voxel, 
                                                              stk0.voxels, 
                                                              mass_transfer_templates, 
                                                              shape2domain_type, 
                                                              seen_mass_transfer_items, 
                                                              diff_terms
                                                              )
        
code = function_code +  variable_code + reaction_code + mass_transfer_code

with open('output_mass_transfer_code.txt', 'w') as file:
    file.write(code)

scope = {'H' : Dummy('H'),
         'A' : Dummy('A'),
         'B' : Dummy('B'),
         'C' : Dummy('D'),
         'k_synH'   : 1,
         'k_synB'   : 1,
         'k_synH_A' : 1,
         
         'J_A'   : 1,
         'J_B_x' : 1,
         'J_B_y' : 1,
         
         'F_A'   : 0,
         'F_B_x' : 1,
         'F_B_y' : 1
         }

exec(tw.dedent(code), scope)

fig2 = plt.figure(figsize=(12, 9))

for n in range(12):
    ax = fig2.add_subplot(3, 4, n+1)
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    AX.append(ax)
    plt.grid(True)

fig2.subplots_adjust(top=0.97, bottom=0.03, left=0.03, right=0.97)

i = 7

xs = ['A', 'B', 'C', 'H']

def get_direction(voxel0, voxel1, value):
    if value < 0:
        return voxel1, voxel0
    elif value == 0:
        return
    else:
        return voxel0, voxel1
    
for x in xs:
    #Advection
    AX[i].set_title('advection '+ x)
    
    mt = f'_adv_{x}'
    if mt in scope:
        for loc, value in scope[mt].items():
            
            a = get_direction(*loc, value)
            
            if a:
                patch = FancyArrowPatch(*a, 
                                        arrowstyle     = 'simple',
                                        mutation_scale = 8,
                                        color          = get_color(value)
                                        )
                AX[i].add_patch(patch)
            else:
                voxel0, voxel1 = loc
                avr = [(voxel0[0] + voxel1[0])/2], [(voxel0[1] + voxel1[1])/2]
                AX[i].scatter(*avr, 32, color=get_color(value))
                
    
    #Diffusion
    AX[i+4].set_title('diffusion '+ x)
    mt = f'_dfn_{x}'
    if mt in scope:
        for loc, value in scope[mt].items():
            
            a = get_direction(*loc, value)
            
            if a:
                patch = FancyArrowPatch(*a,
                                        arrowstyle     = 'simple',
                                        mutation_scale = 8,
                                        color          = get_color(value)
                                        )
                AX[i+4].add_patch(patch)
            else:
                voxel0, voxel1 = loc
                avr = [(voxel0[0] + voxel1[0])/2], [(voxel0[1] + voxel1[1])/2]
                AX[i+4].scatter(*avr, 32, color=get_color(value))
    
    #Boundary condition
    AX[i+8].set_title('boundary '+ x)
    for shift in [-2, -1, 1, 2]:
        mt = ShapeStack._make_boundary(state, shift)
        
        if mt in scope:
            for loc, value in scope[mt].items():
                size = stk0.voxels[loc]['size']
                if shift == -2:
                    loc  = (loc, (loc[0], loc[1]-size)) 
                elif shift == -1:
                    loc  = (loc, (loc[0]-size, loc[1])) 
                elif shift == 1:
                    loc  = (loc, (loc[0]+size, loc[1])) 
                else:
                    loc  = (loc, (loc[0], loc[1]+size)) 
                
                a = get_direction(*loc, -value)
                
                if a:
                    patch = FancyArrowPatch(*a,
                                            arrowstyle     = 'simple',
                                            mutation_scale = 8,
                                            color          = get_color(value)
                                            )
                    AX[i+8].add_patch(patch)
                else:
                    voxel0, voxel1 = loc
                    avr = [(voxel0[0] + voxel1[0])/2], [(voxel0[1] + voxel1[1])/2]
                    AX[i+8].scatter(*avr, 32, color=get_color(value))
    i += 1

###############################################################################
#Map Rates and Domain Types
###############################################################################
temp = ShapeStack._map_rate2domain_type(domain_type2variable, 
                                        variable2domain_type, 
                                        spatial_data
                                        )

domain_type2rate, rate2domain_type = temp

assert len(domain_type2rate) == 1
assert 'cytosolic' in domain_type2rate
assert len(rate2domain_type) == 1
assert 'C' in rate2domain_type

###############################################################################
#Parse Rates
###############################################################################
print('Parse rates')
voxel = (4.75, 4.25)
diff_terms = {}

ShapeStack._parse_rates(voxel, 
                        stk0.voxels, 
                        shape2domain_type,
                        domain_type2rate,
                        diff_terms,
                        spatial_data
                        )

print(diff_terms)
assert len(diff_terms) == 1

diff_code = '\n'.join(diff_terms.values())
scope     = {'vrb0'       : Dummy('vrb0'),
              ut.diff('C') : {}
              }

exec(tw.dedent(diff_code), scope)
assert len(scope[ut.diff('C')])          == 1
assert scope[ut.diff('C')][(4.75, 4.25)] == 4.5

voxel = 4.25, 4.25

ShapeStack._parse_rates(voxel, 
                        stk0.voxels, 
                        shape2domain_type,
                        domain_type2rate,
                        diff_terms,
                        spatial_data
                        )

assert len(scope[ut.diff('C')])          == 1
assert scope[ut.diff('C')][(4.75, 4.25)] == 4.5

###############################################################################
#Combined Test
###############################################################################
print('Combined test: Mass transfer')
temp = ShapeStack._map_shape2domain(spatial_data, *shapes)

shape2domain, domain2shape, shape2domain_type, domain_type2shape  = temp

temp = ShapeStack._map_variable2domain_type(spatial_data)

domain_type2variable, variable2domain_type = temp

temp = ShapeStack._map_reaction2domain_type(variable2domain_type, spatial_data)

domain_type2reaction, reaction2domain_type = temp

mass_transfer_templates = ShapeStack._template_mass_transfer(variable2domain_type, 
                                                              spatial_data
                                                              )

temp = ShapeStack._map_rate2domain_type(domain_type2variable, 
                                        variable2domain_type, 
                                        spatial_data
                                        )

domain_type2rate, rate2domain_type = temp


diff_terms               = {}
seen_variable_items      = set()
seen_reaction_items      = set()
seen_mass_transfer_items = set()
function_code            = ShapeStack._parse_functions(spatial_data)
variable_code            = ShapeStack._make_variable_overhead(variable2domain_type, 
                                                              spatial_data
                                                              )
reaction_code       = ShapeStack._make_reaction_overhead(reaction2domain_type)
mass_transfer_code  = ShapeStack._make_mass_transfer_overhead(mass_transfer_templates)

for voxel, datum in stk0.voxels.items():
    variable_code += ShapeStack._parse_variables(voxel,
                                                 stk0.voxels,
                                                 shape2domain_type, 
                                                 domain_type2variable, 
                                                 seen_variable_items, 
                                                 spatial_data
                                                 )
    
    reaction_code += ShapeStack._parse_reactions(voxel, 
                                                 stk0.voxels, 
                                                 shape2domain_type, 
                                                 domain_type2variable, 
                                                 domain_type2reaction, 
                                                 seen_reaction_items, 
                                                 diff_terms, 
                                                 spatial_data
                                                 )
    
    ShapeStack._parse_rates(voxel, 
                            stk0.voxels, 
                            shape2domain_type,
                            domain_type2rate,
                            diff_terms,
                            spatial_data
                            )
    
    shape       = datum['shape']
    domain_type = shape2domain_type[shape]
    states      = spatial_data.compartments.domain_type2state[domain_type]
    
    for state in states:
        if state not in mass_transfer_templates:
            continue
        
        mass_transfer_code += ShapeStack._parse_mass_transfer(state, 
                                                              voxel, 
                                                              stk0.voxels, 
                                                              mass_transfer_templates, 
                                                              shape2domain_type, 
                                                              seen_mass_transfer_items, 
                                                              diff_terms
                                                              )

diff_code = '\n'.join(diff_terms.values())
code = '\n'.join(['\t#Functions',
                  function_code,
                  '\t#Variables',
                  variable_code,
                  '\t#Reactions',
                  reaction_code,
                  '\t#Mass Transfer',
                  mass_transfer_code,
                  '\t#Differentials',
                  diff_code
                  ])


with open('output_combined_body_code.txt', 'w') as file:
    file.write(code)

###############################################################################
#Instantiate
###############################################################################
stk1 = ShapeStack(spatial_data)

with open('output_final_combined_body_code.txt', 'w') as file:
    file.write(stk1.body_code)
