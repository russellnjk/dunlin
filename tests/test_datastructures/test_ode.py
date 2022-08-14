import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl          as sfd
import dunlin.comp                       as cmp
from dunlin.datastructures.ode import ODEModelData
from data import *


ref       = 'm0'
flattened = cmp.flatten_ode(all_data, ref)
mdata     = ODEModelData(**flattened)

d = mdata.to_data()
d.pop('ref')

dunl = sfd.write_dunl_code({ref: mdata})
a    = sfd.read_dunl_code(dunl)
assert a[ref] == d
