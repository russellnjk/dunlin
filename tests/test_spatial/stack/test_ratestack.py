import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba                   import njit
from time                    import time as get_time
   
import addpath
import dunlin         as dn 
import dunlin.ode.ivp as ivp
import dunlin.utils   as ut
from dunlin.spatial.stack.ratestack import RateStack as Stack
from dunlin.datastructures.spatial  import SpatialModelData
from test_spatial_data              import all_data

#Set up
plt.close('all')
plt.ion()

spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

def make_fig(AX):
    span = -1, 5
    fig  = plt.figure(figsize=(10, 10))
    
    for i in range(4):
        ax  = fig.add_subplot(2, 2, i+1)#, projection='3d')
        ax.set_box_aspect(1)
        
        ax.set_xlim(*span)
        ax.set_ylim(*span)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        plt.grid(True)
        
        AX.append(ax)
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    return fig, AX

def make_colorbar_ax(ax):
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    return cax

AX = []

template  = '''
{body}\n\treturn {return_val}
    
'''

###############################################################################
#Test Instantiation
###############################################################################
stk = Stack(spatial_data)

fig, AX = make_fig(AX)

domain_type_args = {'facecolor': {'cytosolic'     : 'steel',
                                  'extracellular' : 'salmon'
                                  }
                    }

stk.plot_voxels(AX[0], domain_type_args=domain_type_args)

###############################################################################
#Test Rate
###############################################################################
code  = (stk.rhsdef           + '\n' 
         + stk.state_code     + '\n' 
         + stk.diff_code      + '\n'
         + stk.parameter_code + '\n' 
         + stk.function_code  + '\n'
         + stk.variable_code  + '\n'
         + stk.rate_code      + '\n' 
         )
code  = template.format(body       = code,
                        return_val = '_d_A, '
                        )

# print(code)

scope = {}
exec(code, {**stk.rhs_functions}, scope)
# print(scope.keys())

model      = njit(scope['model_M0'])
time       = 0
states     = np.array([1, 2, 3, 4,
                       0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       ])
parameters = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

r = model(time, states, parameters, **stk.args)

assert all(r[0] == [-1, -2, -3, -4]) 

cax = make_colorbar_ax(AX[1])
stk.plot_rate(AX[1], 'A', r[0], cmap='coolwarm', colorbar_ax=cax)
AX[1].set_title('Rate A')

###############################################################################
#Test RHS
###############################################################################
stk.numba = False
code      = stk.rhs.code

with open('output_rhs.txt', 'w') as file:
    file.write(code)

time       = 0
states     = np.arange(0, 32)
parameters = spatial_data.parameters.df.loc[0].values

d_states = stk.rhs(time, states, parameters)

fig, AX = make_fig(AX)

cax = make_colorbar_ax(AX[4])
stk.plot_rate(AX[4], 'A', stk.get_state_from_array('A', d_states), cmap='coolwarm', colorbar_ax=cax)
AX[4].set_title(ut.diff('A'))

cax = make_colorbar_ax(AX[5])
stk.plot_rate(AX[5], 'B', stk.get_state_from_array('B', d_states), cmap='coolwarm', colorbar_ax=cax)
AX[5].set_title(ut.diff('B'))

cax = make_colorbar_ax(AX[6])
stk.plot_rate(AX[6], 'C', stk.get_state_from_array('C', d_states), cmap='coolwarm', colorbar_ax=cax)
AX[6].set_title(ut.diff('C'))

cax = make_colorbar_ax(AX[7])
stk.plot_rate(AX[7], 'D', stk.get_state_from_array('C', d_states), cmap='coolwarm', colorbar_ax=cax)
AX[7].set_title(ut.diff('D'))

###############################################################################
#Test RHS dct
###############################################################################
code = stk.rhsdct.code

with open('output_rhsdct.txt', 'w') as file:
    file.write(code)


#Before running the code
#Check that time, states and parameters are 
#formatted according to the output of the integration function
time       = np.array([0, 1])
states     = np.array([np.arange(0, 32), np.arange(0, 32)]).T
parameters = spatial_data.parameters.df.loc[0].values
parameters = np.array([parameters, parameters]).T

stk.numba = False
dct       = stk.rhsdct(time, states, parameters)

# print(dct)
# for key, value in dct.items():
#     print(key)
#     print(value)
    # print()
    
fig, AX = make_fig(AX)

cax = make_colorbar_ax(AX[8])
stk.plot_rate(AX[8], 'A', dct[ut.diff('A')][:,0], cmap='coolwarm', colorbar_ax=cax)
AX[8].set_title(ut.diff('A'))

cax = make_colorbar_ax(AX[9])
stk.plot_rate(AX[9], 'B', dct[ut.diff('B')][:,0], cmap='coolwarm', colorbar_ax=cax)
AX[9].set_title(ut.diff('B'))

cax = make_colorbar_ax(AX[10])
stk.plot_rate(AX[10], 'C', dct[ut.diff('C')][:,0], cmap='coolwarm', colorbar_ax=cax)
AX[10].set_title(ut.diff('C'))

cax = make_colorbar_ax(AX[11])
stk.plot_rate(AX[11], 'D', dct[ut.diff('D')][:,0], cmap='coolwarm', colorbar_ax=cax)
AX[11].set_title(ut.diff('D'))

###############################################################################
#Test Integration
###############################################################################
states     = np.arange(0, 32)
parameters = spatial_data.parameters.df.loc[0].values
tspan      = np.linspace(0, 50, 11)

rhs     = stk.rhs
t, y, p = ivp.integrate(rhs, tspan, states, parameters)

A = stk.get_state_from_array('A', y)
assert A.shape == (4, 11)

B = stk.get_state_from_array('B', y)
assert B.shape == (4, 11)

C = stk.get_state_from_array('C', y)
assert C.shape == (12, 11)

D = stk.get_state_from_array('D', y)
assert D.shape == (12, 11)

#Get rhsdct
dct = stk.rhsdct(t, y, p)

###############################################################################
#Test RHS Performance
###############################################################################
states     = np.array([1, 2, 3, 4])
states     = stk.expand_init(states)
parameters = spatial_data.parameters.df.loc[0].values
rhs        = stk.rhs

rhs(time, states, parameters)

#Test performance by increasing voxels
print('Test model M1')

spatial_data = SpatialModelData.from_all_data(all_data, 'M1')

print('Making stack')
start = get_time()
stk_  = Stack(spatial_data)
stop  = get_time()
print('time taken', stop - start)
print()

with open('Performance test code 1.txt' , 'w') as file:
    file.write(stk_.rhs.code)
    
states_     = np.array([1, 2, 3, 4])
states_     = stk_.expand_init(states_)
parameters_ = spatial_data.parameters.df.loc[0].values
rhs_        = stk_.rhs

rhs_(time, states_, parameters_)

print('Test model M2')

spatial_data = SpatialModelData.from_all_data(all_data, 'M2')

print('Making stack')
start = get_time()
stk__  = Stack(spatial_data)
stop  = get_time()
print('time taken', stop - start)
print()

with open('Performance test code 2.txt' , 'w') as file:
    file.write(stk__.rhs.code)
    
states__     = np.array([1, 2, 3, 4])
states__     = stk__.expand_init(states__)
parameters__ = spatial_data.parameters.df.loc[0].values
rhs__        = stk__.rhs

rhs__(time, states__, parameters__)

###############################################################################
#Testing Large Numerical Integration
###############################################################################
states_     = np.array([1, 2, 3, 4])
states_     = stk_.expand_init(states_)
parameters_ = spatial_data.parameters.df.loc[0].values
rhs_        = stk_.rhs

print('Test large integration for M1')
start = get_time()
t, y, p = ivp.integrate(rhs_, tspan, states_, parameters_, )
stop  = get_time()
print('time taken', stop - start)
print()

states__     = np.array([1, 2, 3, 4])
states__     = stk__.expand_init(states__)
parameters__ = spatial_data.parameters.df.loc[0].values
rhs__        = stk__.rhs

# print('Test large integration for M2')
# start = get_time()
# t, y, p = ivp.integrate(rhs__, tspan, states__, parameters__)
# stop  = get_time()
# print('time taken', stop - start)
# print()
