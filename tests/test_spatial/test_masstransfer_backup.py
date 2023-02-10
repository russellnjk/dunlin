import matplotlib.pyplot as plt
import numpy             as np

import addpath
import dunlin as dn
import dunlin.utils as ut
from dunlin.datastructures.spatial        import SpatialModelData
from dunlin.spatial.geometry.masstransfer import *

plt.close('all')
plt.ion()

all_data = dn.read_dunl_file('spatial_0.dunl')

mref = 'M0'
gref = 'Geo0'
ref  = mref, gref

spatial_data = SpatialModelData.from_all_data(all_data, mref, gref)


stack = make_stack(spatial_data)

span = -1, 11
fig  = plt.figure(figsize=(18, 10))
AX   = []
for i in range(6):
    ax  = fig.add_subplot(2, 3, i+1)
    ax.set_box_aspect()
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # plt.grid(True)
    
    AX.append(ax)
fig.tight_layout()

stack.plot_voxels(AX[0], 
                  facecolor={'field': 'cobalt', 
                             'cell': 'ocean'
                             }
                  )

with open('output.txt', 'w') as f:
    voxel2dmnt = map_voxel_domain_type(stack, spatial_data)
    dmnt2x     = map_compartments_to_domain_types(spatial_data)
    # rxn_templates = make_reaction_rate_templates(spatial_data)
    
    rhs_def            = make_rhs_def(spatial_data)
    x_code, x2idx      = make_x_code(spatial_data, stack)
    p_code             = make_p_code(spatial_data)
    func_code          = make_func_code(spatial_data)
    vrb_code          = make_vrb_code(spatial_data)
    rxn_code, tr_rxns  = make_bulk_rxn_code(spatial_data)
    rt_code            = make_rt_code(spatial_data)
    
    f.write(rhs_def + '\n')
    f.write(x_code + '\n')
    f.write(p_code + '\n')
    f.write(func_code + '\n')
    f.write(vrb_code + '\n')
    f.write(rxn_code + '\n')
    f.write(rt_code + '\n')
    
    
    adv_code, dfn_code, tr_rxn_code = make_bulk_mass_transfer_code(spatial_data, 
                                                                   stack, 
                                                                   tr_rxns, 
                                                                   x2idx, 
                                                                   voxel2dmnt,
                                                                   dmnt2x
                                                                   )
    
    f.write(adv_code + '\n')
    f.write(dfn_code + '\n')
    f.write(tr_rxn_code + '\n')
    
    diffs_code = make_differentials_code(spatial_data, tr_rxns)
    bc_code    = make_boundary_condition_code(spatial_data, stack, voxel2dmnt)
    
    f.write(diffs_code + '\n')
    f.write(bc_code + '\n')
    
    return_code = make_rhs_return(x2idx)
    
    f.write(return_code)

with open('rhs.py', 'w') as f:
    stack, rhs_code, *_ = make_code(spatial_data, _use_numba=False)
    
    f.write('import numpy as __np\n')
    f.write('from numba import njit as __njit\n\n')
    f.write(rhs_code)

t = 0
y = np.concatenate([np.linspace(0, 99, 100)]*4)
p = np.array([1, 10, 0.05, 
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
              0, 0, 0, 0, 0.05, 0.05
              ])
__np = np

e_voxels = {k:v for v, k in enumerate(stack.voxels)}

x_code, x2idx      = make_x_code(spatial_data, stack)
p_code             = make_p_code(spatial_data)
func_code          = make_func_code(spatial_data)
vrb_code           = make_vrb_code(spatial_data)
rxn_code, tr_rxns  = make_bulk_rxn_code(spatial_data)
rt_code            = make_rt_code(spatial_data)

#x_code
code = f'def f(time, states, parameters):\n{x_code}\n\treturn H, A, B, C'
exec(code)

H, A, B, C = f(t, y, p)

assert all(np.isclose(H, y[0:100]))
assert all(np.isclose(A, y[100:200]))
assert all(np.isclose(B, y[200:300]))
assert all(np.isclose(C, y[300:400]))

#p_code
pnames = spatial_data['model']['parameters'].names
pnames = ', '.join(pnames)
code   = f'def f(time, states, parameters):\n{p_code}\n\treturn {pnames}'
exec(code)

p_ = f(t, y, p)
(k_synH, k_synB, k_synH_A, 
 J_H_x, J_H_y, J_A_x, J_A_y, J_B_x, J_B_y, 
 F_H_x, F_H_y, F_A_x, F_A_y, F_B_x, F_B_y) = p_

assert all(np.isclose(p, p_))

#func_code
code = f'def f():\n{func_code}\n\treturn func0'
exec(code)

func0 = f()
assert func0(2, 3) == -6

#vrb_code
code = f'def f(func0, k_synH, C):\n{vrb_code}\n\treturn vrb0, vrb1'
exec(code)
vrb0, vrb1 = f(func0, k_synH, C)

assert all(np.isclose(vrb0, -k_synH*C))
assert vrb1 == 1

#rxn_code
code = f'def f(k_synH, C):\n{rxn_code}\n\treturn synH'
exec(code)
synH = f(k_synH, C)

assert all(np.isclose(synH, k_synH*C))

#rt_code
code = f'def f(vrb0):\n{rt_code}\n\treturn {ut.diff("C")}'
exec(code)
_d_C = f(vrb0)

assert all(np.isclose(_d_C, vrb0))


###############################################################################
#Spatial Items
###############################################################################
dmnt2x = map_compartments_to_domain_types(spatial_data)
dmnt2x = map_compartments_to_domain_types(spatial_data)

adv_code, dfn_code, tr_rxn_code = make_bulk_mass_transfer_code(spatial_data, 
                                                               stack, 
                                                               tr_rxns, 
                                                               x2idx, 
                                                               voxel2dmnt,
                                                               dmnt2x
                                                               )

#adv_code
r    = ', '.join([ut.adv(i) for i in 'HAB'])
code = f'def f(H, A, B, F_H_x, F_H_y, F_A_x, F_A_y, F_B_x, F_B_y):\n{adv_code}\n\treturn {r}'
exec(code)

B                          = np.ones(100)
F_H_x, F_H_y, F_A_x, F_A_y = 0, 0, 0, 0
F_B_x, F_B_y               = 1, 1 
r                          = f(H, A, B, F_H_x, F_H_y, F_A_x, F_A_y, F_B_x, F_B_y)
_adv_H, _adv_A, _adv_B     = r

assert all(np.isclose(_adv_H, 0))
assert all(np.isclose(_adv_A, 0))

s          = list(stack.sizes.values())
normalized = _adv_B/s

for voxel, m in zip(e_voxels, normalized):
    if voxel2dmnt[voxel] != 'extracellular':
        assert m == 0
    elif voxel == (1, 1):
        assert m == -2
    elif voxel == (9, 1):
        assert m == 0
    elif voxel == (1, 9):
        assert m == 0
    elif voxel == (9, 9):
        assert m == 2
    elif 1 in voxel:
        assert m == -1
    elif 9 in voxel:
        assert m == 1
    else:
        assert m == 0

#dfn code
'''
The concentration of A and B in all voxels are 0. The diffusion must thus be zero 
for these species.

Voxels with size 1 are set so that the concentration of H is one. H therefore 
diffuses into the adjacent voxels. 
'''
r    = ', '.join([ut.dfn(i) for i in 'HAB'])
code = f'def f(H, A, B, J_H_x, J_H_y, J_A_x, J_A_y, J_B_x, J_B_y):\n{dfn_code}\n\treturn {r}'
exec(code)

B                          = np.zeros(100)
J_H_x, J_H_y, J_A_x, J_A_y = 0, 0, 0, 0
J_B_x, J_B_y               = 1, 1

for voxel, voxel_num in e_voxels.items():
    if stack.sizes[voxel] == 1:
        B[voxel_num] = 1

r = f(H, A, B, J_H_x, J_H_y, J_A_x, J_A_y, J_B_x, J_B_y)
_dfn_H, _dfn_A, _dfn_B     = r

assert all(np.isclose(_dfn_H, 0))
assert all(np.isclose(_dfn_A, 0))

s          = list(stack.sizes.values())
normalized = _dfn_B/s

for voxel, m in zip(e_voxels, _dfn_B):
    voxel_num = e_voxels[voxel]
    if not set(voxel).difference([1, 9]):
        assert m == 0
    elif 1 in voxel or 9 in voxel:
        dist = 1.5
        area = 1
        n    = 2
        grad = 1
        J    = 1
        assert m == J*area*n*grad/dist
    elif not set(voxel).difference([3.25, 6.75]):
        dist = 0.75
        area = 0.5
        n    = 2
        grad = 1
        J    = 1
        assert m == J*area*n*grad/dist
    elif 3.25 in voxel or 6.75 in voxel:
        dist = 0.75
        area = 0.5
        n    = 1
        grad = 1
        J    = 1
        assert m == J*area*n*grad/dist
    elif stack.sizes[voxel] == 1 and not set(voxel).difference([2.5, 7.5]):
        dist = 1.5
        area = 1
        n    = 2
        grad = 1
        J    = 1
        
        big = J*area*n*grad/dist
        
        assert m == -big
    elif stack.sizes[voxel] == 1:
        dist = 1.5
        area = 1
        n    = 1
        grad = 1
        J    = 1
        
        big = J*area*n*grad/dist
        
        dist = 0.75
        area = 0.5
        n    = 2
        grad = 1
        J    = 1
        
        small = J*area*n*grad/dist
        
        assert m == -big -small
    else:
        assert m == 0
        
#tr rxn code
r    = ', '.join([f'_tr_{i}' for i in 'HAB'])
code = f'def f(H, A, B, k_synB, k_synH_A):\n{tr_rxn_code}\n\treturn {r}'
exec(code)

H, A, B = np.zeros(100), np.zeros(100), np.ones(100)
k_synB   = 2
k_synH_A = 2

r = f(H, A, B, k_synB, k_synH_A)
_tr_H, _tr_A, _tr_B = r

for voxel, h, a, b in zip(e_voxels, _tr_H, _tr_A, _tr_B):
    if any([i > 6.5 or i < 3.5 for i in voxel]):
        assert h == a == b == 0
    elif not set(voxel).difference([6.25, 3.75]):
        assert h == a == b == 0
    elif voxel in [[3.75, 5.75], [4.25, 6.25], [3.75, 4.25], [4.25, 3.75],  
                   [5.75, 3.25], [6.25, 4.25], [6.25, 5.75], [5.75, 6.25]
                   ]:
        assert h == a == b == 0
    elif all([4.5 < i < 5.5 for i in voxel]):
        assert h == a == b == 0
    elif all([4 < i < 6 for i in voxel]) and not set(voxel).difference([4.25, 5.75]):
        conc_B   = 1
        rxn_rate = k_synH_A*conc_B
        n_trans  = 2
        sign     = -1
        assert b == rxn_rate*n_trans*sign
        assert h == a == 0
    elif all([4 < i < 6 for i in voxel]):
        conc_B   = 1
        rxn_rate = k_synH_A*conc_B
        n_trans  = 1
        sign     = 1
        assert b == 0
        assert h == a == rxn_rate*n_trans*sign
        
        
        
        
        pass

# ###############################################################################
# #Test Code
# ###############################################################################
# import rhs       as _rhs
# import importlib as im

# stack, rhs_code, *_ = make_code(spatial_data, _use_numba=False)

# with open('rhs.py', 'w') as f:
#     f.write('import numpy as __np\n')
#     f.write('from numba import njit as __njit\n\n')
#     f.write(rhs_code)
    
# im.reload(_rhs)

# rhs = _rhs.spatial_M0__Geo0


# dy = rhs(t, y, p)

