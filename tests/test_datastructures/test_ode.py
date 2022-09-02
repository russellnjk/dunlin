import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl          as sfd
from dunlin.datastructures.ode import ODEModelData
from data import *


ref       = 'm0'
mdata     = ODEModelData.from_all_data(all_data, ref)

d = mdata.to_data()

dunl = sfd.write_dunl_code({mdata['ref']: mdata})
a    = sfd.read_dunl_code(dunl)
assert a[ref] == d[ref]
