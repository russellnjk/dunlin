import numpy  as np 
import pandas as pd

from optimize import plot_dataset

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

def format_multiindex(df, drop_lvl=None, truncate=None, thin=None):
    if drop_lvl:
        df = df.droplevel(drop_lvl, axis=1)
    
    if truncate:
        df = df.loc[truncate[0]:truncate[1]]
    
    if thin:
        df = df.loc[::thin]
    
    scenarios = []
    for lvl in range(1, df.columns.nlevels):
        scenarios.append(df.columns.get_level_values(lvl))
    
    scenarios = [x if len(x) > 1 else x[0] for x in zip(*scenarios)]
    states    = df.columns.get_level_values(0)
    
    new_cols = pd.MultiIndex.from_tuples(zip(states, scenarios))
    new_df   = pd.DataFrame(df.values, index=df.index, columns=new_cols)
    
    return new_df
