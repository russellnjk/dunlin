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

class BaseModel(ABC):
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
    functions  : tuple[str]
    variables  : tuple[str]
    reactions  : tuple[str]
    events     : tuple[str]
    meta       : dict[str, Union[str, Number, tuple[str, Number]]]
    
    #Viewable and editable by front-end users
    numba      : bool
    tspan      : dict[Scenario, np.array]
    int_args   : dict
    
    #For back-end only
    #Store the datastructures generated from the raw input
    _model_data : dict
    
    #For back-end only
    #Required for integration
    _rhs_functions    : tuple[callable]
    _rhsdct_functions : tuple[callable]
    _events           : tuple[oev.Event]
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
               '_model_data',
               '_rhs_funcs',
               '_rhsdct_funcs',
               '_events'
               ]
    
    @classmethod
    def from_dict(cls, all_data: dict, ref: str) -> type:
        flattened = cmp.flatten_ode(all_data, ref)
        
        return cls(**flattened)
        
    def __init__(self, 
                 model_data    : Any,
                 ref           : str, 
                 tspans        : dict = None,
                 int_args      : dict = None,
                 dtype         : str  = 'ode',
                 events        : list = ()
                 ):
        
        if dtype != self._dtype:
            msg = f'Attempted to instantiate {type(self).__name__} with data labelled as {dtype}.'
            raise TypeError(msg)
        
        self._model_data      = model_data
        self.ref              = model_data.ref
        self.numba            = True
        self.tspans           = tspans
        self.int_args         = int_args
        self._state_names     = list(model_data.states)
        self._parameter_names = list(model_data.parameters)
        self._events          = events
        self._externals       = {}
    
    ###########################################################################
    #State and Parameter Management
    ###########################################################################
    def _as_df(self, 
                name2scenario_values: Union[dict, pd.DataFrame, pd.Series],
                expected_columns    : tuple[str]
                ) -> pd.DataFrame:
        
        if type(name2scenario_values) == dict:
            df = pd.DataFrame(name2scenario_values)
        elif type(name2scenario_values) == pd.DataFrame:
            df = name2scenario_values.copy()
        elif type(name2scenario_values) == pd.Series:
            df = pd.DataFrame([name2scenario_values])
        
        if any(df.columns != expected_columns):
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
        df               = self._as_df(state2scenario_values, expected_columns)
        self._state_df   = df
        
    
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
        
        expected_columns   = self._parameter_names
        df                 = self._as_df(parameter2scenario_values, expected_columns)
        self._parameter_df = df
    
    @property
    def parameter_dict(self) -> dict:
        df = self.parameter_df
        return dict(zip(df.index, df.values))
    
    @property
    def externals(self) -> dict:
        return self._externals
    
    def add_external(self, name: str, function: callable) -> None:
        ut.check_valid_name(name)
        
        self._externals[name] = function
    
    def pop_external(self, name: str) -> callable:
        return self._externals.pop(name)
    
    ###########################################################################
    #Time Span Management
    ###########################################################################
    @property
    def tspans(self) -> dict:
        return self._tspans
    
    @tspans.setter
    def tspans(self, scenario2tspan: dict) -> None:
        if scenario2tspan is None:
            self._tspans = {}
            return
        
        elif not ut.isdictlike(scenario2tspan):
            raise TypeError('tspan must be a dict or None.')
        
        #Check dimensions and type
        formatted = {}
        for scenario, tspan in scenario2tspan.items():
            arr = np.unique(tspan)
            if len(arr.shape) != 1:
                raise ValueError('tspan for each scenario must be one dimensional.')
            formatted[scenario] = arr
        self._tspans = formatted
            
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
        elif attr == 'int_args' and type(value) != dict:
            if value is None:
                super().__setattr__(attr, {})
            else:
                msg = 'int_args attribute must be a dict or None.'
                raise ValueError(msg)
        else:
            super().__setattr__(attr, value)
    
    ###########################################################################
    #Integration
    ########################################################################### 
    @property
    def rhs(self) -> callable:
        if self.numba:
            return self._rhs_functions[1]
        else:
            return self._rhs_functions[0]
    
    @property
    def rhsdct(self) -> callable:
        if self.numba:
            return self._rhsdct_functions[1]
        else:
            return self._rhsdct_functions[0]
    
    def __call__(self, 
                 y0             : np.array, 
                 p0             : np.array, 
                 tspan          : np.array, 
                 raw            : bool = False, 
                 include_events : bool = True,
                 scenario       : str  = ''
                 ) -> np.array:
        #Reassign and/or extract
        events   = self._events
        int_args = self.int_args
        t, y, p  = ivp.integrate(self.rhs, 
                                 tspan          = tspan, 
                                 y0             = y0, 
                                 p0             = p0, 
                                 events         = events, 
                                 include_events = include_events,
                                 **int_args
                                 )
        if raw:
            return t, y, p
        else:
            return self._convert_call(t, y, p, scenario)
   
    @abstractmethod
    def _convert_call(self, t, y, p, c) -> Any:
        pass
    
    def integrate(self, 
                  scenarios      : list = None,
                  raw            : bool = False, 
                  include_events : bool = True,
                  _y0            : dict[Scenario, np.ndarray] = None,
                  _p0            : dict[Scenario, np.ndarray] = None,
                  _tspans        : dict[Scenario, np.ndarray] = None
                  ) -> dict|Any:
        '''
        Numerical integration from the front end. 

        Parameters
        ----------
        scenarios: list, optional
            A list of scenarios to integrate over. An exception is not raised if 
            the scenario does not exist in the model's states.
        raw : bool, optional
            If True, returns Numpy arrays. If False, returns the integration 
            results as part of another class designed to package the results. The 
            default is False.
        include_events : bool, optional
            Include overlapping time points for events. The default is True.
        _y0 : dict[Scenario, np.ndarray], optional
            For overriding the model's stored state values. The default is None. 
            Not meant for use by end-users.
        _p0 : dict[Scenario, np.ndarray], optional
            For overriding the model's stored parameter values. The default is None.
            Not meant for use by end-users.
        _tspans : dict[Scenario, np.ndarray], optional
            For overriding the model's stored tspan values. The default is None.
            Not meant for use by end-users.
            
        Returns
        -------
        dict
            A dictionary where keys are scenarios and the values are the results 
            of the numerical integration. 

        '''
        
        result     = {}
        states     = self.state_dict     if _y0 is None else _y0
        parameters = self.parameter_dict if _p0 is None else _p0
        
        
        for scenario, y0 in states.items():
            if scenarios:
                if scenario in scenarios:
                    continue
                
            p0    = parameters[scenario]
            tspan = self.get_tspan(scenario) if _tspans is None else _tspans[scenario]
            
            result[scenario] = self(y0, 
                                    p0, 
                                    tspan, 
                                    raw            = raw, 
                                    include_events = include_events,
                                    scenario       = scenario
                                    )
        if raw:
            return result
        else:
            return self._convert_integrate(result)
    
    @abstractmethod
    def _convert_integrate(self, scenario2intresult: dict[Scenario, Any]) -> Any:
        pass