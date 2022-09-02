import addpath
import dunlin                   as dn
import dunlin.standardfile.dunl as sfd
from dunlin.datastructures.geometrydata import GeometryData
from dunlin.datastructures.ode          import ODEModelData
from dunlin.datastructures.spatial      import SpatialModelData 
from spatial_data import *

ref   = 'Geo0'
gdata = GeometryData.from_all_data(all_data, ref)
d     = gdata.to_data()

dunl = sfd.write_dunl_code({gdata['ref']: gdata})
a    = sfd.read_dunl_code(dunl)
assert a[ref] == all_data[ref] == d[ref]

ref = 'M0'
mdata     = ODEModelData.from_all_data(all_data, ref)

d = mdata.to_data()

dunl = sfd.write_dunl_code({mdata['ref']: mdata})
a    = sfd.read_dunl_code(dunl)
assert a[ref] == d[ref]


mref = 'M0'
gref = 'Geo0'
ref  = mref, gref

spldata = SpatialModelData.from_all_data(all_data, *ref)
d       = spldata.to_data()

dunl = sfd.write_dunl_code({ref: spldata})
a    = sfd.read_dunl_code(dunl)
assert  a[gref] == all_data[gref] == d[gref]
assert a[mref] == d[mref]
#print(dunl)