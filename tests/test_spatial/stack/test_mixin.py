import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import textwrap          as tw
from scipy.interpolate       import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

import addpath
import dunlin         as dn 
import dunlin.ode.ivp as ivp
import dunlin.utils   as ut
from dunlin.spatial.stack.mixin    import Imager as Stack
from dunlin.datastructures.spatial import SpatialModelData
from test_spatial_data             import all_data

#Set up
plt.close('all')
plt.ion()

spatial_data = SpatialModelData.from_all_data(all_data, 'M0')

def make_fig(AX):
    span = -1, 5
    fig  = plt.figure(figsize=(10, 10))
    
    for i in range(4):
        ax  = fig.add_subplot(2, 2, i+1)#, projection='3d')
        # ax.set_box_aspect(1)
        
        # ax.set_xlim(*span)
        # ax.set_ylim(*span)
        
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        
        # plt.grid(True)
        
        AX.append(ax)
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    return fig, AX

def make_colorbar_ax(ax):
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    return cax

AX = []

stk = Stack(spatial_data)

time       = 0
init       = np.array([1, 2, 3, 4])
states     = stk.expand_init(init)
states     = np.arange(0, 32)
parameters = spatial_data.parameters.df.loc[0].values
tspan      = np.linspace(0, 50, 11)

t, y, p = ivp.integrate(stk.rhs, tspan, states, parameters)

#for ndims, make a ndims+1 array
state       = 'C'
resolution  = 8

domain_type = stk.state2domain_type[state] 
voxels      = stk.voxel2domain_type.inverse[domain_type]
values      = stk.get_state_from_array(state, y)

voxel2domain_type     = stk.voxel2domain_type
domain_type_idx2voxel = stk.voxel2domain_type_idx.inverse

ndims             = stk.ndims
all_voxels        = stk.voxels
step              = stk.grid.step
voxels_outside    = []

for v, datum in all_voxels.items():
    if voxel2domain_type[v] != domain_type:
        voxels_outside.append(v)
    
    for shift in datum['boundary']:
        idx = abs(shift)-1
        new = v[idx] + step if shift > 0 else v[idx] - step
        pad = v[:idx] + (new,) + v[idx+1:]
        
        voxels_outside.append(pad)
        
x_coordinates  = []
y_coordinates  = []
z_coordinates  = []
t_coordinates  = []
to_interpolate = []
spans          = stk.grid.spans
resolution     = 1j*resolution
grid_t, grid_x, grid_y = np.mgrid[tspan[0]        : tspan[-1]       : len(tspan)*1j,
                                  spans[0][0]-step: spans[0][1]+step: resolution,
                                  spans[1][0]-step: spans[1][1]+step: resolution,
                                  ]

for i, t_point in enumerate(t):
    for domain_type_idx in range(len(values)):
        #Determine the voxel
        voxel = domain_type_idx2voxel[domain_type_idx, domain_type]
    
    
        t_coordinates.append(t_point)
        x_coordinates.append(voxel[0])
        y_coordinates.append(voxel[1])
        
        if ndims == 3:
            z_coordinates.append(voxel[2])

for i, t_point in enumerate(t):
    for voxel in voxels_outside:
        t_coordinates.append(t_point)
        x_coordinates.append(voxel[0])
        y_coordinates.append(voxel[1])
        
        if ndims == 3:
            z_coordinates.append(voxel[2])
        
#Convert the coordinates into an array
if ndims == 2:
    points = zip(t_coordinates, x_coordinates, y_coordinates)
else:
    raise NotImplementedError()

points         = np.array(list(points))    

#xi must be a tuple because the behaviour of the function changes with its type
#The linear and cubic methods don't work for 3D
flattened                 = values.T.flatten()
values_                   = np.zeros(len(points))
values_[0:len(flattened)] = flattened

grid = griddata(points, 
                values_, 
                xi = (grid_t, grid_x, grid_y), 
                method='cubic'
                )

fig, AX = make_fig(AX)
vmax    = np.max(values)
AX[0].imshow(grid[0], vmin=0, vmax=vmax, origin='lower')
AX[1].imshow(grid[2], vmin=0, vmax=vmax, origin='lower')
AX[2].imshow(grid[4], vmin=0, vmax=vmax, origin='lower')

