import addpath
import dunlin                            as dn
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
import dunlin.comp                       as cmp
from dunlin.datastructures.ode import ODEModelData
from data import *


ref       = 'm0'
flattened = cmp.flatten_ode(all_data, ref)
md0       = ODEModelData(**flattened)

d = md0.to_data()

assert set(flattened.keys()) == set(d.keys())