import addpath
import dunlin                   as dn
import dunlin.standardfile.dunl as sfd
from dunlin.datastructures.ode          import ODEModelData
from dunlin.datastructures.spatial      import SpatialModelData 
from spatial_data import *

#Test spatial data
ref = 'M0'

spldata = SpatialModelData.from_all_data(all_data, ref)
d0 = spldata.to_data()
d1 = spldata.to_dunl_dict()

dunl = sfd.write_dunl_code(d1)
a    = sfd.read_dunl_code(dunl)

for k, v in d0.items():
    if k == 'ref':
        continue
    
    if k == 'states' or k == 'parameters':
        v_ = {k_: list(v_.values()) for k_, v_ in v.items()}
        assert a[ref][k] == v_
    else:
        assert a[ref][k] == v
