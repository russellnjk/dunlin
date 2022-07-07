import addpath
import dunlin.utils                      as ut
import dunlin.standardfile.dunl.readdunl as rdn
from dunlin.datastructures.coordinatecomponent import CoordinateComponent, CoordinateComponentDict

ge0 = {'coordinate_components': {'x': {'bounds': [0, 10], 'unit': 'm'},
                                 'y': {'bounds': [0, 10], 'unit': 'm'},
                                 'z': {'bounds': [0, 10], 'unit': 'm'},
                                 }
       }

data0 = {'x': {'bounds': [0, 10], 'unit': 'm'},
         'y': {'bounds': [0, 10], 'unit': 'm'},
         'z': {'bounds': [0, 10], 'unit': 'm'},
         }

C = CoordinateComponentDict

#Test instantiation
F0 = C(data0)

#Test access
f0 = F0['x']

#Test export/roundtrip
data1 = F0.to_data()
dunl = F0.to_dunl()
data2 = rdn.read_string(';A\n' + dunl)['A']
assert data2 == data1 == data0