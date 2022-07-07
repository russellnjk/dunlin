import numpy  as np
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence, TypeVar, Union

import dunlin.utils      as ut
import dunlin.utils_plot as upp
from dunlin.utils.typing import (ODict, OStr, OScenario,
                                 Dflst, Dfdct, Index, Num, 
                                 Model, Scenario, VScenario,
                                 VData
                                 )


class TimeResponseData:
    ###########################################################################
    #Preprocessing
    ###########################################################################
    @staticmethod
    def concat_by_trial(dfs_lst: Dflst) -> Dflst:
        to_concat = {}
        for trial, df in enumerate(dfs_lst):
            to_concat[trial] = df
        
        names  = ['trial', 'time']
        new_df = pd.concat(to_concat, axis=0, names=names)
        new_df = new_df.swaplevel(axis=0)
        return new_df
        
    @staticmethod
    def concat_scenario(dfs_dct: Dfdct) -> Dflst:
        #TODO complete the implementation
        raise NotImplementedError('Not tested yet.')
        
        if dfs_dct is None:
            return []
        
        dfs = []
        
        for scenario, df in dfs_dct.items():
            df_         = pd.concat({scenario: df}, axis=1)
            cols        = df_.columns
            df_.columns = cols.reorder_levels([-1, *range(cols.nlevels-1)])
            
            dfs.append(df_)
        
        return dfs
    
    @staticmethod
    def concat_variable(dfs_dct: Dfdct) -> Dflst:
        #TODO complete the implementation
        raise NotImplementedError('Not tested yet.')
        
        if dfs_dct is None:
            return []
            
        dfs = []
        
        for variable, df in dfs_dct.items():
            df_ = pd.concat({variable: df}, axis=1)
            dfs.append(df_)
            
        return dfs
        
    @staticmethod
    def split_df(*dfs: Dflst) -> tuple[dict, set, set]:
        dct       = {}
        seen_ex   = set()
        seen_reg  = set()
        scenarios = set()
        
        for df in dfs:
            # #Make a working copy that can be modified 
            # df = df.copy()
            
            #Check index
            if df.index.nlevels > 2:
                raise ValueError('Expected a DataFrame with 1 or 2 levels in the index.')
            
            #Split into series
            if df.columns.nlevels == 1:
                #Assume it is an extra variable
                for (scenario, variable), series in df.stack().groupby(level=[0, -1]):
                    #Check formatting
                    if variable in seen_reg:
                        msg = f'Could not determine if {variable} was an extra variable.' 
                        raise ValueError(msg)
                    
                    #Add the variable
                    seen_ex.add(variable)    
                    scenarios.add(scenario)
                    dct.setdefault(scenario, {})
                    
                    #Preprocess the values
                    series.name = variable, scenario
                    series      = series.reset_index(drop=True)
                    series      = series.astype(np.float64).dropna()
                    
                    #Add the values
                    dct[scenario][variable] = series
                
            else:
                #Assume it is a regular variable
                for column_name, column in df.items():
                    #Extrace variable and scenario
                    if len(column_name) == 2:
                        variable, scenario = column_name
                    else:
                        variable, *scenario = column_name
                        scenario            = tuple(scenario)
                    
                    #Check formatting
                    if variable in seen_ex:
                        msg = f'Could not determine if {variable} was an extra variable.' 
                        raise ValueError(msg)
                    
                    #Add the variable
                    seen_reg.add(variable)    
                    scenarios.add(scenario)
                    dct.setdefault(scenario, {})
                    
                    #Preprocess the values
                    column.name = variable, scenario
                    column      = column.sort_index(axis=0)
                    column      = column.astype(np.float64).dropna()
                    
                    #Add the values
                    dct[scenario][variable] = column
        
        extra_variables = seen_ex
        namespace       = seen_ex | seen_reg
        return dct, namespace, extra_variables
    
    @staticmethod
    def _format_df(df, 
                   trial: bool = False, 
                   thin: int = 1, 
                   truncate: tuple[float, float]=None, 
                   drop: Optional[list[Scenario]] = None, 
                   roll: Optional[int] = None, 
                   halflife: Optional[float] = None
                   ) -> pd.DataFrame:
        #Truncate
        if truncate:
            start, stop = truncate
        else:
            start, stop = df.index[0], df.index[-1]
            
        df = df.loc[start:stop] 
        
        #Smoothing
        if roll:
            df = df.rolling(window=1, min_periods=1).median()
        if halflife:
            df = df.ewm(halflife=halflife, ignore_na=True).mean()
        
        #Thin
        df = df.iloc[::thin]
        
        #Get rid of unwanted levels
        if drop is not None:
            df = df.droplevel(level=drop, axis=1)
        
        #Create a multiindex from the trials
        if trial:
            df = df.stack(-1)
        
        return df
        
    @classmethod
    def csv2df(cls, filename, *, header=2, **kwargs):
        header_   = list(range(header))
        index_col = [0]
        
        df = pd.read_csv(filename, header=header_, index_col=index_col)
        df = df.rename(columns=ut.try2num)
        
        #Additional preprocessing for regular variables
        if header > 1:
            df = cls._format_df(df, **kwargs)
            
        return df
    
    @classmethod
    def xlsx2df(cls, filename, *, sheet_name, header=2, **kwargs):
        header_   = list(range(header))
        index_col = [0]
            
        df = pd.read_excel(filename, sheet_name=sheet_name, header=header_, index_col=index_col)
        df = df.rename(columns=ut.try2num)
        
        #Additional preprocessing for regular variables
        if header > 1:
            df = cls._format_df(df, **kwargs)

        return df
    
    ###########################################################################
    #Constructors
    ###########################################################################
    @classmethod
    def load_files(cls, 
                   file_data: ODict = None, 
                   concat_by_trial: ODict = None, 
                   dataset_args: ODict = None, 
                   model: Optional[Model] = None, 
                   no_fit: Optional[bool] = None,
                   **kwargs
                   ):
        
        dfs = []
        
        #file_data
        if file_data:
            
            for filename, args in file_data.items():
                filename = Path(filename)
                ftype    = filename.suffix[1:]
                
                func = getattr(cls, ftype+'2df', None)
                if func is None:
                    msg = f'Could not find a method for opening {filename}'
                    raise AttributeError(msg)
                
                df = func(filename, **args)

                dfs.append(df)
        
        #concat by trial
        if concat_by_trial:
            for group, group_data in concat_by_trial.items():
                to_concat = {}
                for i, (filename, args) in enumerate(group_data.items()):
                    filename = Path(filename)
                    ftype    = filename.suffix[1:]
                    
                    func = getattr(cls, ftype+'2df', None)
                    if func is None:
                        msg = f'Could not find a method for opening {filename}'
                        raise AttributeError(msg)
                    
                    if 'trial' in args:
                        raise ValueError('Cannot use the trial argument.')
                    
                    df = func(filename, **args)
                    to_concat[i] = df
                
                names  = ['trial', 'time']
                new_df = pd.concat(to_concat, axis=0, names=names)
                new_df = new_df.swaplevel(axis=0)
                
                dfs.append(new_df)
        
        
        #Add model
        if model is None:
            dataset = cls(*dfs, no_fit=no_fit, dataset_args=dataset_args, **kwargs)
        else:
            dataset = cls(*dfs, model=model, no_fit=no_fit, dataset_args=dataset_args, **kwargs)
        
        return dataset
        
    def __init__(self, *dfs: Sequence[pd.DataFrame], 
                 model: Optional[Model] = None, 
                 no_fit: Sequence[str] = None, 
                 dataset_args: ODict = None, 
                 dtype: str = 'time_response_data'
                 ):
        
        if dtype != 'time_response_data':
            msg = f'Attempted to instantiate {type(self).__name__} with {dtype} data.'
            raise TypeError(msg)
            
        dct, namespace, extra_variables = self.split_df(*dfs)
        
        self.data            = dct
        self.namespace       = namespace
        self.scenarios       = set(dct)
        self.extra_variables = extra_variables
        
        #Process the no-fit
        if no_fit is None:
            self.no_fit = set() 
        elif no_fit == 'all':
            self.no_fit = set(namespace) 
        elif type(no_fit) in [list, tuple, set, dict]:
            self.no_fit = set(no_fit)
        elif type(no_fit) == str:
            self.no_fit.add(no_fit)
        else: 
            raise ValueError('no_fit must be an iterable or the string "all".')
        
        #Extract dataset args from model if applicable
        if model is None:
            self.dataset_args = {} if dataset_args is None else dataset_args
            self.ref          = ''
        else:
            default           = {} if model.sim_args  is None else model.sim_args
            data_args         = {} if model.data_args is None else model.data_args 
            self.dataset_args = data_args.get('dataset', default)
            self.ref          = model.ref
        
    ###########################################################################
    #Access and Modification
    ###########################################################################
    def get(self, variable: str = None, scenario: Scenario = None, _extract: bool = True
            ) -> VData:
        
        if type(variable) == list:
            dct = {}
            for v in variable:
                temp = self.get(v, scenario, False)
                dct.update(temp)
            return dct
        elif type(scenario) == list:
            dct = {}
            for c in scenario:
                temp = self.get(variable, c, False)
                
                for key in temp:
                    dct[key] = {**dct.setdefault(key, {}), **temp[key]}
            return dct
        
        
        if variable is not None and variable not in self.namespace:
            raise ValueError(f'Unexpected variable: {repr(variable)}')
        elif scenario is not None and scenario not in self.data:
            raise ValueError(f'Unexpected scenario: {repr(scenario)}')
        
        dct  = {}
        
        for c, c_data in self.data.items():
            if not ut.compare_scenarios(c, scenario):
                continue

            for v in c_data:
                
                if not ut.compare_variables(v, variable):
                    continue
                
                series = c_data[v]
                
                if v in self.extra_variables:
                    dct.setdefault(v, {})[c] = series
                else:
                    dct.setdefault(v, {})[c] = series
        
        if variable is not None and scenario is not None and _extract:
            return dct[variable][scenario]
        else:
            return dct
        
    def __getitem__(self, key: VScenario) -> VData:

        if type(key) == tuple:
            if len(key) != 2:
                raise ValueError('Expected a tuple of length 2.')
            variable, scenario = key  
        else:
            variable = key
            scenario = None
            
        return self.get(variable, scenario)
    
    def has(self, variable: OStr = None, 
            scenario: OScenario = None
            ) -> bool:
        if type(variable) in [list, tuple]:
            return all([self.has(v, scenario) for v in variable])
        elif type(scenario) == list:
            return all([self.has(variable, c) for c in scenario])
        
        
        if variable is None and scenario is None:
            raise ValueError('variable and scenario cannot both be None.')
        elif variable is None:
            return scenario in self.scenarios
        elif scenario is None:
            return variable in self.namespace
        else:
            return variable in self.namespace and scenario in self.scenarios
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        return f'{type(self).__name__}{tuple(self.scenarios)}'
    
    def __repr__(self):
        return self.__str__()
    
    ###########################################################################
    #Derived Variables
    ###########################################################################
    def _check_derived_variable(self, name, no_fit, *variables, allow_num=True):
        if allow_num:
            test = lambda v: type(v) not in [str, dict] and not ut.isnum(v)
        else:
            test = lambda v: type(v) not in [str, dict]
            
        if any([test(v) for v in variables]):
            temp = ', '.join([str(v) for v in variables])
            msg = f'All variables must be strings. Received: {temp}'
            
            raise ValueError(msg)
        
        if name in self.namespace:
            raise NameError(f'{name} is already in use.')
        elif name is not None:
            self.namespace.add(name)
            
            if no_fit:
                self.no_fit.add(name)
            
    def _setderived(self, name: str, scenario: Scenario, series: pd.Series
                    ) -> None:
        if name is None:
            return
        
        else:
            self.data[scenario][name] = series
    
    def dup(self, v0: str, /, name: str=None, fillna: Num = None, 
            no_fit: bool = True
            ) -> dict:
        self._check_derived_variable(name, no_fit, v0)
        
        x0_dct     = self[v0][v0] 
        result_dct = {}
        
        for scenario, series0 in x0_dct.items():
            series1 = series0.copy()
            
            if fillna is not None:
                series1 = series1.fillna(fillna)
           
            #Update
            result_dct[scenario] = series1
            self._setderived(name, scenario, series1)
        
        return result_dct
    
    def apply(self, f: callable, *variables: Sequence[str], name: OStr = None, 
              fillna: Num = None, no_fit: bool = True
              ):
        self._check_derived_variable(name, no_fit, *variables)
        
        #First variable cannot be a number
        dcts      = {}
        scenarios = {}
        for v in variables:
            if type(v) == str:
                temp         = self[v][v]
                dcts[v]      = temp
                scenarios[v] = set(temp)
            elif type(v) == dict:
                dcts[v]      = v
                scenarios[v] = set(v)
            else:
                dcts[v] = v
        
        scenarios  = set.intersection(*scenarios.values())
        result_dct = {}
        
        for scenario in scenarios:
            args = []
            for k, dct in dcts.items():
                value = dct[scenario] if type(dct) == dict else dct
                args.append(value)
            
            #Compute and fill
            series1 = f(*args)
            
            if fillna is not None:
                series1 = series1.fillna(fillna)
            
            #Update
            result_dct[scenario] = series1
            self._setderived(name, scenario, series1)
        
        return result_dct
        
    def arith(self, v0, v1, f, /, name=None, fillna=None, no_fit=True):
        self._check_derived_variable(name, no_fit, v0, v1)
        
        #First variable cannot be a number
        x0_dct     = self[v0][v0] if type(v1) == str else v0
        x1_dct     = self[v1][v1] if type(v1) == str else v1
        result_dct = {}
        
        for scenario, series0 in x0_dct.items():
            if ut.isnum(v1):
                series1 = v1
            else:
                series1 = x1_dct.get(scenario)
                
                #Check if the two are compatible
                if series1 is None:
                    continue
                elif np.any(series0.index != series1.index):
                    raise ValueError('Indices of both series are not the same.')

            #Compute and fill
            series2 = f(series0, series1)
            
            if fillna is not None:
                series2 = series2.fillna(fillna)
            
            #Update
            result_dct[scenario] = series2
            self._setderived(name, scenario, series2)
        
        return result_dct
    
    def div(self, v0, v1,/, name=None, fillna=None, no_fit=True):
        f = lambda x, y: x/y
        return self.arith(v0, v1, f, name, fillna, no_fit)
        
    
    def mul(self, v0, v1,/, name=None, fillna=None, no_fit=True):
        f = lambda x, y: x*y
        return self.arith(v0, v1, f, name, fillna, no_fit)
    
    def add(self, v0, v1,/, name=None, fillna=None, no_fit=True):
        f = lambda x, y: x+y
        return self.arith(v0, v1, f, name, fillna, no_fit)
    
    def sub(self, v0, v1,/, name=None, fillna=None, no_fit=True):
        f = lambda x, y: x-y
        return self.arith(v0, v1, f, name, fillna, no_fit)
    
    def diff_t(self, v0, *, name=None, no_fit=True, fillna=0, periods=1):
        self._check_derived_variable(name, v0)
        
        x0_dct     = self[v0][v0]
        result_dct = {}
        
        for scenario, series0 in x0_dct.items():
            if series0.index.nlevels == 1:
                series1 = series0.diff(periods=1)
                
                t  = series0.index.get_level_values(0)
                t  = pd.Series(t, index=series0.index)
                dt = t.diff(periods)
            else:
                series1 = series0.groupby(level=1).diff(periods)
                
                t  = series0.index.get_level_values(0)
                t  = pd.Series(t, index=series0.index)
                dt = t.groupby(level=1).diff(periods)
            
            series1 = series1/dt
            
            if fillna is not None:
                series1 = series1.fillna(fillna)
                
            result_dct[scenario] = series1
            self._setderived(name, scenario, series1)
        
        return result_dct
    
    def diff(self, v0, *, name=None, no_fit=True, fillna=0, periods=1):
        self._check_derived_variable(name, v0)
        
        x0_dct     = self[v0][v0]
        result_dct = {}
        
        for scenario, series0 in x0_dct.items():
            if series0.index.nlevels == 1:
                series1 = series0.diff(periods=1)
            else:
                series1 = series0.groupby(level=-1).diff(periods)
            
            if fillna is not None:
                series1 = series1.fillna(fillna)
                
            result_dct[scenario] = series1
            self._setderived(name, scenario, series1)
        
        return result_dct
    
    def spec_diff(self, v0, *, name=None, no_fit=True, fillna=0, periods=1):
        self._check_derived_variable(name, v0)
        
        x0_dct     = self[v0][v0]
        result_dct = {}
        
        for scenario, series0 in x0_dct.items():
            #Compute and fill
            if series0.index.nlevels == 1:
                series1 = series0.diff(periods=1)
                
                t  = series0.index.get_level_values(0)
                t  = pd.Series(t, index=series0.index)
                dt = t.diff(periods)
            else:
                series1 = series0.groupby(level=1).diff(periods)
                
                t  = series0.index.get_level_values(0)
                t  = pd.Series(t, index=series0.index)
                dt = t.groupby(level=1).diff(periods)
            
            series1 = series1/dt/series0
            
            if fillna is not None:
                series1 = series1.fillna(fillna)
            
            result_dct[scenario] = series1
            self._setderived(name, scenario, series1)
        
        return result_dct
        
    def first_order_gen(self, v0, dil, *, name=None, no_fit=True, fillna=0, periods=1):
        self._check_derived_variable(name, v0)
        
        x0_dct     = self[v0][v0]
        dil_dct    = self[dil][dil]
        result_dct = {}
        
        for scenario, series0 in x0_dct.items():
            if series0.index.nlevels == 1:
                series1 = series0.diff(periods=1)
                
                t  = series0.index.get_level_values(0)
                t  = pd.Series(t, index=series0.index)
                dt = t.diff(periods)
            else:
                series1 = series0.groupby(level=1).diff(periods)
                
                t  = series0.index.get_level_values(0)
                t  = pd.Series(t, index=series0.index)
                dt = t.groupby(level=1).diff(periods)
            
            series1 = series1/dt
            
            #Extract dilution
            series2 = dil_dct.get(scenario)
            
            #Check if the two are compatible
            if series2 is None:
                continue
            elif np.any(series1.index != series2.index):
                raise ValueError('Indices of both series are not the same.')
            
            #Compute generation and fill
            series3 = series1 + series2*series0
            
            if fillna is not None:
                series1 = series1.fillna(fillna)
                
            result_dct[scenario] = series3
            self._setderived(name, scenario, series3)
        
        return result_dct
    
    ###########################################################################
    #Intermediate Processing for Curve-fitting
    ###########################################################################
    def reindex(self, variables: Sequence[str], scenarios: Sequence[Scenario], 
                model: Model = None, dataset_args: ODict = None, 
                no_fit: Sequence[str] = None
                ):
        variables = variables if ut.isdictlike(variables) else {x: x for x in variables}
        scenarios = scenarios if ut.isdictlike(scenarios) else {x: x for x in scenarios}
        
        #Parse dataset args
        if model is None:
            if dataset_args is None:
                dataset_args = self.dataset_args 
            else:
                dataset_args = dataset_args
        else:
            dataset_args = model.data_args.get('dataset')
        
        #Instantiate
        new_dataset   = type(self)(dataset_args=dataset_args)
        
        #Fill the data in the new object
        for old_scenario, new_scenario in scenarios.items():
            #Create a renamed deep copy of the dicts
            old = self.data[old_scenario]
            new = {variables[v] : series.copy() for v, series in old.items() if v in variables}
            
            #Update
            new_dataset.data[new_scenario] = new
            new_dataset.scenarios.add(new_scenario)
            new_dataset.namespace.update(new.keys())
        
        #Determine which extra variables are included
        old_extra_variables = self.extra_variables.intersection(variables)
        new_extra_variables = {variables[v] for v in old_extra_variables}
        
        #Update
        new_dataset.extra_variables = new_extra_variables
        new_dataset.ref             = self.ref
        
        #Process the no-fit
        if no_fit is None:
            new_dataset.no_fit = set() 
        elif no_fit == 'all':
            new_dataset.no_fit = set(new_dataset.namespace) 
        elif type(no_fit) in [list, tuple, set, dict]:
            new_dataset.no_fit = set(no_fit)
        elif type(no_fit) == str:
            new_dataset.no_fit.add(no_fit)
        else: 
            raise ValueError('no_fit must be an iterable or the string "all".')
           
        return new_dataset
    
    def adjust_model_init(self, model: Model, states: Sequence[str]) -> pd.DataFrame:
        df     = model.states
        states = list(states)
        
        #Check states
        unexpected = set([x for x in states if x not in df.columns])
        if unexpected:
            raise ValueError(f'Unexpected states: {unexpected}')
            
        
        #Update the df
        for scenario, dct in self.data.items():
            for variable, series in dct.items():
                if variable in states:
                    #Extract from dataset
                    if series.index.nlevels == 1:
                        init = series.iloc[0]
                    else:
                        init = series.groupby(level=1).first().mean()
                    
                    #Assign
                    df.loc[scenario, variable] = init
        
        #Update the model
        model.states = df
        
        return df
    
    @property
    def n_points(self) -> int:
        n = 0
        for scenario, dct in self.data.items():
            for variable, series in dct.items():
                if variable in self.no_fit:
                    continue
                elif hasattr(series, '__len__'):
                    n += len(series)
                else:
                    n += 1
        return n
                    
    ###########################################################################
    #Plotting Functions
    ###########################################################################
    def plot_line(self, ax_dct, variable, xlabel=None, ylabel=None, title=None,
                  skip=None, thin=1, **line_args
                  ):
        #Determine which variables to plot
        if ut.islistlike(variable):
            x, y = variable
            
            if y not in self.namespace or x not in self.namespace:
                raise ValueError('Unexpected variable: "{variable}"')
        else:
            x, y = 'time', variable
            
            if y not in self.namespace:
                raise ValueError('Unexpected variable: "{variable}"')
            
        #Prepare the sim args
        ext_args  = line_args
        self_args = self.dataset_args.get('line_args', {})
        
        result      = {}
        #Iterate and plot
        for c, c_data in self.data.items():
            if upp.check_skip(skip, c):
                continue
            
            #Get values
            series = c_data[y]
            gb     = series.groupby(level=0)
            y_vals = gb.mean()
            y_vals = y_vals.iloc[::thin].values
            yerr   = None if series.index.nlevels == 1 else gb.std().iloc[::thin].values
            
            if x == 'time':
                x_vals = np.array(list(gb.groups.keys()))
                x_vals = x_vals[::thin]
                xerr   = None
            else:
                series = c_data[x]
                gb     = series.groupby(level=0)
                x_vals = gb.mean()
                x_vals = x_vals[::thin].values
                xerr   = None if series.index.nlevels == 1 else gb.std()[::thin].values
            
            #Parse kwargs
            kwargs = self._process_plot_args(self_args, 
                                             ext_args, 
                                             scenario=c, 
                                             var=variable
                                             )
            
            #Determine ax
            ax = upp.recursive_get(ax_dct, c)
            
            if ax is None:
                continue

            #Plot
            result[c] = ax.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, **kwargs)
            
            #Label axes
            upp.label_ax(ax, x, xlabel, y, ylabel)
            upp.set_title(ax, title, self.ref, variable, c)
            
        return result
    
    def _process_plot_args(self, self_args, ext_args, **to_recurse):
        #Recurse through the arguments
        keys = to_recurse.values()
        def recurse(dct):
            if dct is None:
                return {}
            return {k: upp.recursive_get(v, *keys) for k, v in dct.items()}
        
        
        self_args = recurse(self_args)
        ext_args  = recurse(ext_args) 
        kwargs    = {**self_args, **ext_args}
        
        #Collate args for processing
        scenario = to_recurse.get('scenario', '')
        var      = to_recurse.get('var', '') 
        args     = dict(scenario=scenario, variable=var, ref=self.ref)
        #Process the label
        upp.replace(kwargs, 'label', '{scenario}', **args)
        #Process color
        upp.replace(kwargs, 'color', None, upp.get_color, **args)
        #Process marker
        upp.replace(kwargs, 'marker', 'o', **args)
        #Process linestyle
        upp.replace(kwargs, 'linestyle', 'None', **args)
        
        #Replace other callable args
        upp.call(kwargs, ['label', 'color'])
        
        return kwargs
        
    
    def plot_bar(self, ax, variable, by='scenario', xnames=None, ynames=None, 
                 **bar_args):
        '''For extra only
        '''
        #Determine which variables to plot
        if ut.islistlike(variable):
            variables = list(variable)
        else:
            variables = [variable]
        
        #Create the dataframe
        dct = {}
        for c, c_data in self.data.items():
            for v in variables:
                
                if v not in self.extra_variables:
                    msg = f'plot_bar can only be used extra variables. Received {v}'
                    raise ValueError(msg)
                    
                dct.setdefault(v, {})[c] = c_data[v]
        
        #Concat the series before calling df constructor
        dct = {k: pd.concat(v) for k, v in dct.items()}
        df_ = pd.DataFrame(dct)
        
        #Get height and sd
        gb     = df_.groupby(level=0)
        df     = gb.mean()
        sd     = None if df_.index.nlevels == 1 else gb.std() 
    
        #Determine how the bars should be grouped
        if by == 'scenario':
            pass
        elif by == 'variable':
            df = df.T
            sd = sd.T
        else:
            df = df.unstack(by).stack(0)
            sd = sd.unstack(by).stack(0)
            
        if callable(xnames):
            df.index = [xnames(i) for i in df.index]
            sd.index = [xnames(i) for i in sd.index]
        elif type(xnames) == str:
            if df.index.nlevels == 1:
                df.index = [xnames.format(i) for i in df.index]
                sd.index = [xnames.format(i) for i in sd.index]
            else:
                df.index = [xnames.format(*i) for i in df.index]
                sd.index = [xnames.format(*i) for i in sd.index]
        
        #Prepare the sim args
        self_args = self.dataset_args.get('bar_args', {})
        kwargs    = {**self_args, **bar_args}
        
        #Replace other callable args
        upp.call(kwargs, 
                 [], 
                 ref=self.ref,
                 scenarios=tuple(df.index), 
                 variables=variables
                 )
        
        #ynames
        cols            = []
        ori             = kwargs.get('color', {})
        kwargs['color'] = {}
        for c in df.columns:
            if callable(ynames):
                new_c = ynames(c)
            elif type(ynames) == str:
                if df.columns.nlevels == 1:    
                    new_c = ynames.format(c)
                else:
                    new_c = ynames.format(*c)
            else:
                new_c = c
            
            kwargs['color'][new_c] = upp.get_color(ori.get(c))   
            cols.append(new_c)
        
        df.columns = cols
        
        #Plot the bars
        result = df.plot.bar(ax=ax, yerr=sd, **kwargs)
        ax.get_legend().remove()
        
        return result
        
        