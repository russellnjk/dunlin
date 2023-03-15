

import numpy    as np
import textwrap as tw

import addpath
import dunlin       as dn
import dunlin.utils as ut
from dunlin.spatial.dynamic.masstransfer import (mt_template,
                                                 vrb_template,
                                                 rxn_template1,
                                                 rxn_template_bulk,
                                                 rxn_template_edge,                                                 
                                                 Neumann_template_plus
                                                 )
'''
Imagine the following grid.
    
    #######
    #0#1#2#
    #3#4#5#
    #6#7#8#
    #######

Where 4 belongs to a different shape.

If A and C are only found on the outside, then it would be an array of length 8. 
The indices for each element in the array would correspond to

    #######
    #0#1#2#
    #3# #4#
    #5#6#7#
    #######

Where there is no element for the center voxel.

If B is only found in the inside, then it would be an array of length 1.
The indices for each element in the array would correspond to

    #######
    # # # #
    # #0# #
    # # # #
    #######

Where there is only one element and it corresponds to the center voxel.


'''

#For testing only
A  = np.arange(0, 8)
C  = np.arange(0, 8) + 0.5
B  = np.array([1])

###############################################################################
#Bulk Variables
###############################################################################
print('Testing bulk variable')
'''
If vrb1 is only found on the outside, then it would be an array of length 8. 
The indices for each element in the array would correspond to

    #######
    #0#1#2#
    #3# #4#
    #5#6#7#
    #######

Where there is no element for the center voxel.
'''
variable = 'vrb1'

#Inferred arguments from preprocessing stage
expr     = 'A + 1'

#This will be part of the algorithm for this module
string = vrb_template.format(name = variable,
                             expr = expr
                             )

print(string)

#Check values
scope = {'_np' : np,
         'A'   : A,
         }

exec(tw.dedent(string), scope)

a = A + 1
r = np.isclose(a, scope['vrb1'])
assert all(r)
print()

###############################################################################
#Edge Variables
###############################################################################
print('Testing edge variable')
#Choosen arguments
variable = 'vrb2'

#Inferred arguments from preprocessing stage
edge_A2B = [1, 3, 4, 6]
edge_B2A = [0, 0, 0, 0]
expr     = 'A[_edge_A2B] - B[_edge_B2A]'

#This will be part of the algorithm for this module
string = vrb_template.format(name = variable,
                             expr = expr
                             )

print(string)

#Check values
scope = {'_np'       : np,
         'A'         : A,
         'B'         : B,
         '_edge_A2B' : edge_A2B,
         '_edge_B2A' : edge_B2A
         }

exec(tw.dedent(string), scope)

a = A[edge_A2B] - 1
r = np.isclose(a, scope['vrb2'])
assert all(r)
print()

###############################################################################
#Bulk Reaction
###############################################################################
print('Testing bulk reaction')
#Chosen arguments
reaction = 'rxn0'

#Inferred arguments
expr   = 'k0*A'
stoich = {'A' : -1, 'C': 1}

#This will be part of the algorithm for this module
string = rxn_template1.format(name = reaction,
                              expr = expr
                              )

for state, coeff in stoich.items():
    string += rxn_template_bulk.format(name  = reaction,
                                   coeff = coeff,
                                   diff  = ut.diff(state),
                                   )

print(string)

scope = {'k0'         : 0.5,
         'A'          : A,
         'C'          : C,
         ut.diff('A') : np.zeros(8),
         ut.diff('C') : np.zeros(8)
         }

exec(tw.dedent(string), scope)

a = np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5])
r = np.isclose( a, scope['rxn0'])
w = np.isclose(-a, scope[ut.diff('A')])
q = np.isclose( a, scope[ut.diff('C')])
assert all(r)
assert all(w)
assert all(q)
print()

###############################################################################
#Edge Reaction
###############################################################################
print('Testing edge reaction')
#Chosen arguments
reaction = 'rxn1'

#Inferred arguments
edges    = {'A': [1, 3, 4, 6],
            'B': [0, 0, 0, 0]
            }
expr     = 'k0*A[_edge_A2B]*B[_edge_B2A]'
stoich   = {'A' : 1, 'B': -1}

#This will be part of the algorithm for this module
string = rxn_template1.format(name = reaction,
                              expr = expr,
                              )

for state, coeff in stoich.items():
    string += rxn_template_edge.format(name  = reaction,
                                       coeff = coeff,
                                       diff  = ut.diff(state),
                                       edge = edges[state]
                                       )

print(string)

scope = {'k0'         : 0.5,
         'A'          : A,
         'B'          : B,
         '_edge_A2B'  : edge_A2B,
         '_edge_B2A'  : edge_B2A,
         ut.diff('A') : np.zeros(8),
         ut.diff('B') : np.zeros(1)
         }

exec(tw.dedent(string), scope)

h = np.array([0.5, 1.5, 2. , 3. ])
a = np.array([0. , 0.5, 0. , 1.5, 2. , 0. , 3. , 0. ])

r = np.isclose( h, scope['rxn1'])
w = np.isclose( a, scope[ut.diff('A')])
q = np.isclose( sum(h), scope[ut.diff('B')])
assert all(r)
assert all(w)
assert all(q)
print()

###############################################################################
#Mass Transfer
###############################################################################
print('Testing mass transfer')
#Choosen arguments
state     = 'A'
axis      = 'x'

#Inferred arguments from preprocessing stage
adv_coeff = 'F'
dfn_coeff = 'J'
ndims     = 2
src       = [0, 1, 5, 6]
dst       = [1, 2, 6, 7]

src_size = [1]*len(src)
dst_size = [1]*len(dst)

size    = list(np.minimum(src_size, dst_size))
nvoxels = 8

#This will be part of the algorithm for this module
string = mt_template.format(state     = state,
                            axis      = axis,
                            adv_coeff = adv_coeff,
                            dfn_coeff = dfn_coeff,
                            src       = src,
                            dst       = dst,
                            size      = size,
                            dims      = ndims-1,
                            nzero     = nvoxels,
                            diff      = ut.diff(state)
                            )

print(string)

#Check values
#Advection only
scope = {'_np'         : np,
          'F'          : 0.5,
          'J'          : 0,
          'A'          : A,
          ut.diff('A') : np.zeros(8)
          }

exec(tw.dedent(string), scope)

a = np.array([ 0. , -0.5,  0.5,  0. ,  0. , -2.5, -0.5,  3. ])
r = np.isclose(a, scope['_advA_x'])
w = np.isclose(a, scope[ut.diff('A')])
assert all(r)
assert all(w)

#Diffusion only
scope = {'_np'         : np,
          'F'          : 0,
          'J'          : 0.5,
          'A'          : A,
          ut.diff('A') : np.zeros(8)
          }

exec(tw.dedent(string), scope)

a = np.array([ 0.5,  0. , -0.5,  0. ,  0. ,  0.5,  0. , -0.5])
r = np.isclose(a, scope['_dfnA_x'])
w = np.isclose(a, scope[ut.diff('A')])
assert all(r)
assert all(w)

#Advection and diffusion
scope = {'_np'         : np,
          'F'          : 0.5,
          'J'          : 0.5,
          'A'          : A,
          ut.diff('A') : np.zeros(8)
          }

exec(tw.dedent(string), scope)

a = np.array([ 0. , -0.5,  0.5,  0. ,  0. , -2.5, -0.5,  3. ])
r = np.isclose(a, scope['_advA_x'])
assert all(r)
a = np.array([ 0.5,  0. , -0.5,  0. ,  0. ,  0.5,  0. , -0.5])
r = np.isclose(a, scope['_dfnA_x'])
assert all(r)
a = np.array([ 0.5, -0.5,  0. ,  0. ,  0. , -2. , -0.5,  2.5])
w = np.isclose(a, scope[ut.diff('A')])
assert all(w)
print()

###############################################################################
#Boundary Conditions
###############################################################################
print('Testing boundary condition')
#Choosen arguments
state     = 'A'
axis      = 'x'

#Inferred arguments from preprocessing stage
ndims     = 2
dfn_coeff = 'J'
cond_type = 'Neumann'
cond      = 1
src       = [2, 4, 7]
size      = [1]*len(src)
nvoxels   = 8

#This will be part of the algorithm for this module
string = Neumann_template_plus.format(state     = state,
                                      axis      = axis,
                                      dfn_coeff = dfn_coeff,
                                      expr      = cond,
                                      src       = src,
                                      size      = size,
                                      dims      = ndims-1,
                                      diff      = ut.diff(state)
                                      )

print(string)

#Check values
scope = {'_np'         : np,
          'J'          : 0,
          'A'          : A,
          ut.diff('A') : np.zeros(8)
          }

exec(tw.dedent(string), scope)

# a = np.array([ 0. , -0.5,  0.5,  0. ,  0. , -2.5, -0.5,  3. ])
# r = np.isclose(a, scope['_advA_x'])
# w = np.isclose(a, scope[ut.diff('A')])
# assert all(r)
# assert all(w)