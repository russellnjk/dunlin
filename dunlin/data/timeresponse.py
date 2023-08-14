import numpy  as np
import pandas as pd
from numbers import Number
from pathlib import Path
from typing  import Literal, Optional, Sequence, TypeVar, Union

import dunlin.utils      as ut
import dunlin.utils_plot as upp

State    = str
Scenario = str|Number

class TimeResponseData:
    @staticmethod
    def preprocess(series   : pd.Series,
                   truncate : list[Number, Number] = None,
                   roll     : int                  = None,
                   thin     : int                  = 1,
                   halflife : Number               = None  
                   ) -> pd.Series:
        pass
    
    @classmethod
    def read_excel(cls, io, sheet_name=0, **kwargs) -> dict:
        pass
    
    @classmethod
    def read_csv(cls, 
                 filepath_or_buffer, 
                 state          : State|None    = None,
                 scenario       : Scenario|None = None,
                 state_level    : int|None = 0,
                 scenario_level : int|None = 1,
                 **pd_kwargs
                 ) -> dict:
        df = pd.read_csv(filepath_or_buffer, **pd_kwargs)
        
        if df.index.nlevels != 1:
            msg = ''
            raise ValueError(msg)
        
        dct = {}
        
        #Case 0: No state is given. Raise an exception.
        if state is None and state_level is None:
            msg = ''
            raise ValueError()
        #Case 1: No scenario is given. Raise an exception.
        elif scenario is None and scenario_level is None:
            msg = ''
            raise ValueError()
        elif state is not None and state_level is not None:
            msg = ''
            raise ValueError()
        elif scenario is not None and scenario_level is not None:
            msg = ''
            raise ValueError()
        #Case 2: States and scenarios are in the columns
        elif state is None and scenario is None:
            pass
        #Case 3: States are in the columns. Scenario given in the input.
        elif state is None:
            pass
        #Case 4: State is given. Scenarios are in the columns
        elif scenario is None:
            pass
        #Case 5: 
        else:
            pass
            
        #Group by state
        
        
        #Group by scenario
        
        
    def __init__(self, 
                 data  : dict[Scenario, dict[State, pd.Series]]|dict[State, dict[Scenario, pd.Series]], 
                 by    : Literal['scenario', 'state'] = 'scenario'
                 ):
        if by == 'scenario':
            pass
        elif by == 'state':
            pass
        else:
            msg =''
            raise ValueError(msg)