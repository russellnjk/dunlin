import numpy  as np 
import pandas as pd

def get_exp_data(filenames, **pd_args):
    all_df    = {}
    exp_data  = {} 
    scenarios = {}
    for model_key in filenames:
        dataset = {}
        scenarios_ = set()
        for filename in filenames[model_key]:
            if type(filename) == dict:
                key  = filename['filename']
                if key in all_df:
                    temp, s = all_df[key]
                else:
                    args        = {**pd_args, **filename}
                    temp, s     = read_csv(**filename)
                    all_df[key] = temp, s
                    
            elif type(filename) in [tuple, list]:
                key = filename[0]
                if key in all_df:
                    temp, s = all_df[key]
                else:
                    temp, s     = read_csv(*filename, **pd_args)
                    all_df[key] = temp, s

            else:
                raise TypeError('File name must be dict of keyword arguments or a tuple/list of arguments.')
            
            dataset = {**dataset, **temp}
            scenarios_.update(s)
            
        exp_data[model_key]  = dataset
        scenarios[model_key] = scenarios_
        
        return exp_data, scenarios

def read_csv(filename, state, **pd_args):
    '''
    Reads a csv file
    '''
    raw_data = pd.read_csv(filename, **pd_args)  
    time     = raw_data[('Time', 'Time')]
    
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
        
        dataset[time_key] = time
        dataset[data_key] = np.array(stacked[scenario])
            
        scenarios.append(scenario)

    return dataset, scenarios