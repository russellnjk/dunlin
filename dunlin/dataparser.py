import numpy  as np 
import pandas as pd

def read_dataset(filename, states=None, **pd_args):
    '''
    Reads a csv file
    '''
    raw_data  = filename if type(filename) == pd.DataFrame else pd.read_csv(filename, **pd_args)  
    dataset   = raw_data.to_dict('list')
    dataset   = {key: value.dropna().values for key, value in raw_data.items()}
    scenarios = list( raw_data.columns.levels[1] )
            
    for state, group in raw_data.xs('Data', axis=1, level=2).groupby(level=0, axis=1):
        variance = np.median(group, axis=[0, 1]) /10
        
        dataset[(state, 'Variance')] = variance
    return dataset, scenarios
    
def read_timeseries(filename, state, time_col=('Time', 'Time'), **pd_args):
    '''
    Reads a csv file
    '''
    raw_data = filename if type(filename) == pd.DataFrame else pd.read_csv(filename, **pd_args)  
    time     = raw_data[time_col].values
    
    if raw_data.columns.nlevels == 1:
        stacked = raw_data[[x for x in raw_data.columns if x != 'Time']]
            
    elif raw_data.columns.nlevels == 2:
        stacked = raw_data[[x for x in raw_data.columns if x != ('Time', 'Time')]]
        n       = len(stacked.columns.unique(1))
        stacked = stacked.stack().reset_index(drop=True).to_dict('list')    
        time    = np.repeat(time, n)
    
    else:
        raise NotImplementedError('No implementation for reading data with 3 or more levels.')
        
    dataset   = {}
    scenarios = []
    
    for scenario in stacked:
        
        time_key = state, scenario, 'Time'
        data_key = state, scenario, 'Data' 
        y_data   = np.array(stacked[scenario])
        
        non_nan_idx       = ~np.isnan(y_data)
        dataset[time_key] = time[non_nan_idx]
        dataset[data_key] = y_data[non_nan_idx]
            
        scenarios.append(scenario)
    
    #Calculate Variance
    variance = raw_data.drop(time_col, axis=1).std(axis=1, level=1)
    variance = variance.median().median()
    
    dataset[(state, 'Variance')] = variance
    
    return dataset, scenarios