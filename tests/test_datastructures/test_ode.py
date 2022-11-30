import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl          as sfd
from dunlin.datastructures.ode import ODEModelData
from data import *


ref       = 'm0'
mdata     = ODEModelData.from_all_data(all_data, ref)

d0 = mdata.to_data()
d1 = mdata.to_dunl_dict()

dunl = sfd.write_dunl_code(d1)
a    = sfd.read_dunl_code(dunl)
assert a == d0

