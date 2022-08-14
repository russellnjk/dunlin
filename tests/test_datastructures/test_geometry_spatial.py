import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl          as sfd
import dunlin.comp                       as cmp
from dunlin.datastructures.geometrydata import GeometryData
from spatial_data import *

# ref       = 'M0'
# flattened = cmp.flatten_ode(all_data, ref)
# md0       = ODEModelData(**flattened)

# d = md0.to_data()

ref   = 'Geo0'
gdata = GeometryData(ref, **all_data[ref])
d     = gdata.to_data()
d.pop('ref')
# assert set(d.keys()).difference( set(all_data[ref].keys()) ) == {'ref'}

dunl = sfd.write_dunl_code({ref: gdata})
a    = sfd.read_dunl_code(dunl)
assert a[ref] == all_data[ref] == d

#print(dunl)