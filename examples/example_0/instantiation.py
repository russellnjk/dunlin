import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import addpath
import dunlin as dn

plt.ion()
plt.close('all')

'''
The recommended way to instantiate a model is to use the markup format unique 
to Dunlin. The markup is designed to be general purpose and thus encode other 
data structures as well. 

When the function dn.load_file(model_filename) is called, it parses the markup 
file to produce a Python dictionary. Along the way, it also instantiates any 
models it detects.

If you only need raw Python data, use dn.read_dunl_file(model_filename) instead.
'''
#From .dunl file
model_filename    = 'modeldata.dunl'
raw, instantiated = dn.load_file(model_filename)

#Get the model and integrate numerically
model0 = instantiated['firstorder'] 

'''
Instantiation from Python dictionaries is also possible. This is useful if you 
are generating the model formulation on the fly.
'''
#Instantiation from Python dictionary
model_data = raw['firstorder']

#Uncomment the following code block to see the model data
# for key, value in model_data.items():
#     print(key)
#     print(value)
#     print()

model1  = dn.ODEModel('firstorder', **model_data)
#OR
model1a = dn.ODEModel.from_data(raw, 'firstorder')

#Note: If your model is hierarchical i.e. uses submodels, then you must 
#use the second approach i.e. model1a

'''
Instantiation from other file formats using dn.load_model is also possible. 
You will need to define a function that reads the file and generates the raw 
data.

def myloader(model_filename: str) -> dict:
    raw = {}
    
    #Update the raw data
    ...
    
    return raw

raw, instantiated = dn.load_file(model_filename, loader)
'''


