import pandas as pd
from collections import namedtuple
from numbers     import Number

State    = str
Scenario = Number|str

Name = namedtuple('Name', 'scenario state')

def read_time_response(config: dict) -> dict[Scenario, dict[State, pd.Series]]:
    #Maps scenario -> state -> pandas series 
    data = {}
    
    for key, dct in config.items():
        if key == 'dtype':
            dtype = dct
            if dtype != 'time_response':
                msg = f'Attempted to extract time response data from dtype {dtype}.'
                raise TypeError(msg)
                
            continue
                
        match dct:
            case {'filename'       : filename,  
                  **kwargs
                  }:
                
                df             = read_file(key, 
                                           filename, 
                                           **kwargs
                                           )

                state_level    = kwargs.get('state_level', 'state')
                scenario_level = kwargs.get('scenario_level', 'scenario')
                trial_level    = kwargs.get('trial_level', 'trial')
                df             = df.stack(trial_level)
                
                for state, v in df.groupby(axis=1, level=state_level):
                    for scenario, vv in v.groupby(axis=1, level=scenario_level):
                        #Create the key for indexing
                        name = (state, scenario)
                        
                        
                        #Update data
                        vv_      = vv.squeeze().dropna()
                        vv_.name = name
                        
                        #Check if an entry already exists in data
                        if name in data:
                            old = data[name]
                            new = pd.concat([old, vv_])
                            new = new.sort_index(level='time')
                            
                            data[name] = new
                        
                        else:
                            data[name] = vv_
             
            case _:
                msg  = f'Could not parse data for {key}.'
                msg += 'Each datum in the config must be a dict with the key "filename".'

                raise ValueError(msg)
        
    return data

def read_file(name           : str|Number,
              filename       : str, 
              scenario_level : str|list[str] = 'scenario',
              state_level    : str           = 'state',
              trial_level    : str|list[str] = 'trial',
              **kwargs
              ) -> pd.DataFrame:
    
    
    try:
        #Check the input
        if type(state_level) != str:
            msg = f'state_level must be a string. Received {type(state_level)}.'
            raise TypeError(msg)
        
        if type(scenario_level) == list:
            if any([type(i) != str for i in scenario_level]):
                msg = f'scenario_level must be a string or list of strings. Received {type(scenario_level)}.'
                raise TypeError(msg)
                
        elif type(scenario_level) != str:
            msg = f'scenario_level must be a string or list of strings. Received {type(scenario_level)}.'
            raise TypeError(msg)
        
        if type(trial_level) == list:
            if any([type(i) != str for i in trial_level]):
                msg = f'trial_level must be a string or list of strings. Received {type(trial_level)}.'
                raise TypeError(msg)
                
        elif type(trial_level) != str:
            msg = f'trial_level must be a string or list of strings. Received {type(trial_level)}.'
            raise TypeError(msg)
        
        #Generate the data frame
        if '.csv' == filename[-4:]:
            df = pd.read_csv(filename, **kwargs)
            
        elif '.xlsx' == filename[-5:]:
            df = pd.read_excel(filename, **kwargs)
        else:
            msg = 'Unexpected file extension for filename : {filename}'
            raise ValueError(msg)
        
        #Check the dataframe
        if df.index.nlevels != 1:
            n    = df.index.nlevels
            msg  = f'The DataFrame has an index with {n} levels. '
            msg += 'It should only have one.'
            raise ValueError(msg)
        
        if 'time' in df.columns.names:
            msg = 'Detected a column named "time".'
            raise ValueError(msg)
        
        if df.columns.nlevels < 3:
            msg = 'Dataframe columns must have 3 or more levels.'
            raise ValueError(msg)
        
        #For some reason pandas read_csv always treats column indices as strings 
        #Convert the numerical values into strings here
        columns = df.columns
        names   = columns.names
        columns = [tuple([trynum(x) for x in c]) for c in columns]
        
        df.columns = pd.MultiIndex.from_tuples(columns, names=names)
        
    except Exception as e:
        msg     = f'Error in parsing data for {name}'
        raise ExceptionGroup(msg, [e])
    
    #Rename the index as "time"
    df.index.name = 'time'
    
    return df

def trynum(x) -> float|int|str:
    try:
        nf = float(x)
        ni = int(nf)
    except:
        return x
    
    if nf == ni:
        return ni
    else:
        return nf