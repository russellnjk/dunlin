import numpy  as np 
import pandas as pd

def str2num(x):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x
            
def state2dataset(raw_data, state):
    #Get time and mean of replicates
    time      = np.array(raw_data.index)
    mean      = raw_data.groupby(axis=1, level=0).mean()
    dataset   = {}
    
    for scenario, y_data in mean.items():
        #Create key and values
        scenario = str2num(scenario)
        y_data   = y_data.values
        data_key = ('Data', scenario, state)
        time_key = ('Time', scenario, state)
        
        #Assign
        dataset[data_key] = y_data
        dataset[time_key] = time
    return dataset
