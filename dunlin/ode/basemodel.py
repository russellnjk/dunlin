import numpy  as np
import pandas as pd
from abc     import ABC, abstractmethod
from numbers import Number
from typing  import Any, Union

###############################################################################
#Non-Standard Imports
###############################################################################
import dunlin.utils     as ut
import dunlin.comp      as cmp
import dunlin.ode.event as oev
import dunlin.ode.ivp   as ivp

#Type hints
Scenario   = Union[str, float, int, tuple]

class BaseModel (ABC):
    '''Meant to be used as the parent class for ODE models, spatial models etc.
    
    Provides a framework for:
        1. Attributes that all child classes should have
        2. Modifying states/parameters 
        3. Locking attributes related to model definition
        4. Methods for numerical integration
    
    Numerical integration can be overidden if the subclass is meant for other 
    types of modelling.
    
    '''
    
    #Viewable but not editable by front-end users
    #To be set once during instantiation and subsequently locked
    states     : tuple[str]
    parameters : tuple[str]
    functions  : frozenset[str]
    variables  : frozenset[str]
    reactions  : frozenset[str]
    events     : frozenset[str]
    scalars    : frozenset[str]
    meta       : dict[str, Union[str, Number, tuple[str, Number]]]
    
    #Viewable and editable by front-end users
    tspan      : dict[Scenario, np.array]
    int_args   : dict
    optim_args : dict
    
    #For back-end only
    #Store the datastructures generated from the raw input
    _model_data : type
    
    #For back-end only
    #Required for integration
    rhs              : callable
    rhsdct           : callable
    rhsscalar        : callable
    _rhs_funcs       : tuple[callable]
    _rhsdct_funcs    : tuple[callable]
    _rhsscalar_funcs : tuple[callable]
    _events          : tuple[oev.Event]
    _default_tspan = np.linspace(0, 1000, 21)
    
    #For back-end only
    #For storing state and parameters
    _state_df     : pd.DataFrame
    _parameter_df : pd.DataFrame
    
    #For back-end only
    #For checking the input data for instantiation
    _dtype : str
    
    #These attributes will be protected 
    #When __setattr__ is called, this list is checked
    #If an attribute listed in _locked has been defined previously, 
    #it cannot be modified 
    _locked = ['states', 
               'parameters', 
               'functions', 
               'variables',
               'reactions', 
               'events',
               'scalars',
               'meta',
               '_model_data',
               '_rhs_funcs',
               '_rhsdct_funcs',
               '_rhsscalar_funcs',
               '_events'
               ]
    
    @classmethod
    def from_dict(cls, all_data: dict, ref: str) -> type:
        flattened = cmp.flatten_ode(all_data, ref)
        
        return cls(**flattened)
        
    def __init__(self, 
                 datastructure : callable,
                 ref           : str, 
                 dtype         : str = 'ode',
                 **kwargs
                 ):
        
        if dtype != self._dtype:
            msg = f'Attempted to instantiate {type(self).__name__} with data labelled as {dtype}.'
            raise TypeError(msg)
        
        self._model_data = datastructure(ref=ref, **kwargs)
        
    ###########################################################################
    #State and Parameter Management
    ###########################################################################
    def _set_df(self, 
                name2scenario_values: Union[dict, pd.DataFrame, pd.Series],
                expected_columns    : tuple[str]
                ) -> pd.DataFrame:
        if type(name2scenario_values) == dict:
            df = pd.DataFrame(name2scenario_values).T
        elif type(name2scenario_values) == pd.DataFrame:
            df = name2scenario_values.copy()
        elif type(name2scenario_values) == pd.Series:
            df = pd.DataFrame([name2scenario_values])
        
        if df.columns != expected_columns:
            msg = ''
            raise ValueError(msg)
        
        return df

    @property
    def state_df(self) -> pd.DataFrame:
        return self._state_df
    
    @state_df.setter
    def state_df(self, 
               state2scenario_values: Union[dict, pd.DataFrame, pd.Series]
               ) -> None:
        
        expected_columns = self._state_names
        self._set_df(state2scenario_values, expected_columns)
    
    @property
    def state_dict(self) -> dict:
        df = self.state_df
        return dict(zip(df.index, df.values))
    
    @property
    def parameter_df(self) -> pd.DataFrame:
        return self._parameter_df
    
    @parameter_df.setter
    def parameter_df(self, 
               parameter2scenario_values: Union[dict, pd.DataFrame, pd.Series]
               ) -> None:
        
        expected_columns = self._parameter_names
        self._set_df(parameter2scenario_values, expected_columns)
    
    @property
    def parameter_dict(self) -> dict:
        df = self.parameter_df
        return dict(zip(df.index, df.values))
    
    ###########################################################################
    #Time Span Management
    ###########################################################################
    @property
    def tspans(self) -> dict:
        return self._tspans
    
    @tspans.setter
    def tspans(self, scenario2tspan: dict) -> None:
        if scenario2tspan is None:
            self._tspan = {}
        
        elif not ut.isdictlike(scenario2tspan):
            raise TypeError('tspan must be a dict or None.')
        
        #Check dimensions and type
        formatted = {}
        for scenario, tspan in scenario2tspan.items():
            arr = np.unique(tspan)
            if len(arr.shape) != 1:
                raise ValueError('tspan for each scenario must be one dimensional.')
        self.tspans = formatted
            
    def get_tspan(self, scenario: Scenario) -> np.ndarray:
        tspans = self.tspans 
        
        return tspans.get(scenario, self.default_tspan)
    
    ###########################################################################
    #Other Attribute Management
    ###########################################################################
    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in self._locked and hasattr(self, attr):
            msg = f'{attr} attribute has already been set and is locked.'
            raise AttributeError(msg)
            
        else:
            super().__setattr__(attr, value)
    
    ###########################################################################
    #Integration
    ###########################################################################
    def __call__(self, 
                 y0             = None, 
                 p0             = None, 
                 tspan          = None, 
                 overlap        = True, 
                 raw            = False, 
                 include_events = True,
                 ) -> np.array:
        #Reassign and/or extract
        events   = self._events
        int_args = self.int_args
        t, y, p  = ivp.integrate(self.rhs, 
                                 tspan          = tspan, 
                                 y0             = y0, 
                                 p0             = p0, 
                                 events         = events, 
                                 overlap        = overlap, 
                                 include_events = include_events,
                                 **int_args
                                 )
        if raw:
            return t, y, p
        else:
            return self._convert_raw_output(t, y, p)
   
    @abstractmethod
    def _convert_raw_output(self, t, y, p) -> type:
        pass
        
    def integrate(self, 
                  scenario, 
                  y0, 
                  p0, 
                  tspans, 
                  **kwargs
                  ) -> dict:
        result     = {}
        states     = self.state_dict
        parameters = self.parameter_dict
        
        #Check that scenarios are correct
        if set(states) != set(parameters):
            msg = 'Scenarios of states and parameters do not match.'
            raise ValueError(msg)
            
        for scenario, y0 in states.items():
            p0    = parameters[scenario]
            tspan = self.get_tspan(scenario)
            
            result[scenario] = self(y0, p0, tspan, **kwargs)
            
        return result
