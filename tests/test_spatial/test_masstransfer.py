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

e_voxels = dict(enumerate(stack.voxels))

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
d_C = f(vrb0)

assert all(np.isclose(d_C, vrb0))

#adv_code
r    = ', '.join([ut.adv(i) for i in 'HAB'])
code = f'def f(H, A, B):\n{adv_code}\n\treturn '
exec(code)

adv_ = [-H[0]*F_H_x*4 -H[0]*F_H_y*4, #(1.0, 1.0)
        +H[0]*F_H_x*4 -H[1]*F_H_x*4 -H[1]*F_H_y*4, #(3.0, 1.0)
        +H[1]*F_H_x*4 -H[2]*F_H_x*4 -H[2]*F_H_y*4, #(5.0, 1.0)
        +H[2]*F_H_x*4 -H[3]*F_H_x*4 -H[3]*F_H_y*4, #(7.0, 1.0)
        +H[3]*F_H_x*4 -H[4]*F_H_y*4, #(9.0, 1.0)
        +H[0]*F_H_y*4 -H[5]*F_H_x*4 -H[5]*F_H_y*4, #(1.0, 3.0)
        +H[4]*F_H_y*4 +H[21]*F_H_x*4 -H[6]*F_H_y*4, #(9.0, 3.0)
        +H[5]*F_H_y*4 -H[7]*F_H_x*4 -H[7]*F_H_y*4, #(1.0, 5.0)
        ]

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

