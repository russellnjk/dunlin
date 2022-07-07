import numpy  as np
import pandas as pd
from abc    import ABC, abstractmethod
from typing import Any, Optional, Union, ItemsView, KeysView, ValuesView, Iterable


import dunlin.utils             as ut
import dunlin.standardfile.dunl as sfd
from dunlin.utils.typing import Dflike, Num


class _AItem(ABC, ut.FrozenObject):
    '''
    Base class for non-tabular datastructures.
    
    Contains:
        1. `namespace` : Names used by the stored values not including reserved 
        words. Needs to be implemented in the subclass. If the subclass does not 
        implement this, the default is a blank tuple.
    
    Export methods:
        1. `to_data` : Returns a dict/list that can be used to re-instantiate 
        the object not including the `name` argument.
        2. `to_dunl` : Return dunl code that can be parsed and used to 
        re-instantiate the object not including the `name` argument.
        
    '''
    @staticmethod
    def format_primitive(x: Union[str, int, float]):
        if ut.isstrlike(x):
            return x.strip()
        
        float_value = float(x)
        int_value   = int(x)
        if float_value == int_value:
            return int_value
        else:
            return float_value
    
    def __init__(self, ext_namespace: set, name: set, new_name: Optional[str]=None
                 ) -> None:
        if not hasattr(self, 'namespace'):
            self.namespace = ()
        
        ut.check_valid_name(name)
        
        if new_name is None:
            new_name = name
            
        self.name = new_name
        
        if new_name in ext_namespace:
            raise NameError(f'Redefinition of {new_name}.')
        else:
            #Update the namespace
            ext_namespace.add(new_name)
        
    def __contains__(self, name: str):
        return name in self.namespace
        
    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def get(self, key: str, default: Any=None):
        return getattr(self, key, default)
    
    def __str__(self):
        return self.to_dunl()
    
    def __repr__(self):
        return f'{type(self).__name__} {self.name}({str(self)})'
    
    @abstractmethod
    def to_data(self) -> Union[list, dict]:
        ...
        
class _ADict(ut.FrozenObject, ABC):
    '''
    Base class for containers for subclasess of _AItem.
    
    It contains key-value pairs where the keys are strings and the values are  
    of the type given by the `itype` argument in the constructor. To prevent 
    unexpected changes in attributes, the object can be frozen at the end of 
    instantiation by calling the `freeze` method. `itype` must be a subclass of 
    _ADict for things to work properly.
    
    Contains:
        1. `itype` : Only values of the type given by this attribute can be 
        added.
        2. `_data` : The underlying dictionary storing key-value pairs. This 
        should not be accessed directly.
        3. `namespace` : Names used by the stored values not including reserved 
        words. Needs to be implemented in the subclass. If the subclass does not 
        implement this, the default is a blank tuple.
    
    Access methods/properties:
        `__getitem__`, `keys`, `values`, `items` work the same way as a regular 
        dictionary.
    
    Export methods:
        1. `to_data` : Returns a dict/list that can be used to re-instantiate 
        the object not including the `name` argument.
        2. `to_dunl` : Return dunl code that can be parsed and used to 
        re-instantiate the object not including the `name` argument.
        
    '''
    itype     : type
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, mapping: dict, ext_namespace: set, 
                 callback: Optional[callable]=None, /, 
                 **additional_args
                 ) -> None:
        super().__init__()
        self._data = {}

        if mapping is not None:
            for name, args in mapping.items():
                #Ensures the name is an allowed string
                if ut.isdictlike(args):
                    try:
                        value = self.itype(ext_namespace, name, **args,  **additional_args)
                    except Exception as e:
                        self._raise(e, name)
                elif ut.islistlike(args):
                    try:
                        value = self.itype(ext_namespace, name, *args,  **additional_args)
                    except Exception as e:
                        self._raise(e, name)
                else:
                    try:
                        value = self.itype(ext_namespace, name, args,  **additional_args)
                    except Exception as e:
                        self._raise(e, name)

                self._data[name] = value

                if callback:
                    if callable(callback):
                        callback(name, value)
                    else:
                        [c(name, value) for c in callback]
        
        if not hasattr(self, 'namespace'):
            self.namespace = ()
    
    def _raise(self, e, name):
        msg = e.args[0]
        msg = f'Error when parsing {name}\n{msg}'
        
        raise type(e)(msg)
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        return self.to_dunl()
    
    def __repr__(self):
        return f'{type(self).__name__}{tuple(self.keys())}'
    
    ###########################################################################
    #Access
    ###########################################################################
    def __getitem__(self, key: str):
        return self._data[key]
    
    def keys(self) -> KeysView:
        return self._data.keys()
    
    def values(self) -> ValuesView:
        return self._data.values()
    
    def items(self) -> ItemsView:
        return self._data.items()
    
    def __iter__(self) -> Iterable:
        return iter(self._data)
    
    def __contains__(self, item) -> bool:
        return item in self._data
    
    def __len__(self) -> int:
        return len(self._data)
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> dict:
        dct = {k: v.to_data() for k,v in self.items()}

        return dct
    
    def to_dunl(self, indent_type='\t', **ignored) -> str:
        return sfd.write_dict(self.to_data())

class _BDict(ABC, ut.FrozenObject):
    '''
    Base class for containers for tabular data.
    
    
    
    '''
    itype   : str
    
    @staticmethod
    def mapping2df(mapping: Dflike):
        #Convert to df
        if type(mapping) == pd.DataFrame:
            df = mapping
        elif type(mapping) == pd.Series:
            df = pd.DataFrame(mapping).T
        elif ut.isdictlike(mapping):
            temp = dict(mapping)
            try:
                df = pd.DataFrame(temp)
            except:
                df = pd.DataFrame(pd.Series(temp)).T
        else:
            msg = f"Expected a DataFrame, Series or dict. Received {type(mapping)}"
            raise TypeError(msg)
    
        return df

    def __init__(self, name: str, mapping: Dflike, 
                 ext_namespace: set, n_format: callable = sfd.format_num
                 ) -> None:
        #Convert to df
        df = self.mapping2df(mapping)
        
        if 0 in df.shape:
            msg  = 'Error in parsing {name}. '
            msg += 'The resulting DataFrame had shape {df.shape}. '
            msg += '{name} must have at least one column and one row.'
            raise ValueError(msg)
            
        #Check names
        names = tuple(df.columns)
        [ut.check_valid_name(name) for name in names]
        if type(name) != str:
            raise TypeError('Expected "name" argument to be a string.')
        
        repeated = ext_namespace.intersection(names)
        if repeated:
            raise NameError(f'Redefinition of {repeated}.')
        
        ext_namespace.update(names)
        
        #Save attributes
        self.name     = name
        self.names    = names
        self._df      = df
        self.n_format = n_format
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self) -> str:
        return self.to_dunl()
    
    def __repr__(self) -> str:
        return f'{type(self).__name__}{tuple(self.keys())}'
    
    ###########################################################################
    #Access/Modification
    ###########################################################################
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @df.setter
    def df(self, new_df: pd.DataFrame) -> None:
        names = self.names
        columns = new_df.columns
        if len(set(names).intersection(columns)) != len(columns):
            msg = 'Column names do not match.'
            msg = f'{msg} Expected: {names}\nReceived: {tuple(columns)}'
            raise ValueError(msg)
        
        self._df = new_df[list(names)]
    
    def by_index(self) -> dict[Union[Num, str], np.ndarray]:
        df  = self.df
        dct = dict(zip(df.index, df.values))
        return dct
    
    def keys(self):
        return self.df.keys()
    
    def values(self):
        return self.df.values()
    
    def items(self):
        return self.df.items()
    
    def __iter__(self):
        return iter(self.names)
    
    def __contains__(self, item: str) -> bool:
        return item in self.names
    
    def __len__(self):
        return len(self.names)
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> dict:
        return self._df.to_dict()
    
    def to_dunl(self) -> str:
        df          = self._df
        n_format    = self.n_format
        
        return sfd.write_numeric_df(df, n_format)
    
class _CItem(ABC, ut.FrozenObject):
    '''For generic dict-like objects that don't require namespace checking.
    '''
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    @abstractmethod
    def to_data(self) -> Union[list, dict]:
        ...

class _CDict(ABC, ut.FrozenObject):
    itype: type
    def __init__(self, mapping: dict):
        _data = {}
        for name, dct in mapping.items():
            item        = self.itype(name, **dct)
            _data[name] = item
          
        self._data = _data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __iter__(self):
        return iter(self._data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    def to_data(self) -> dict:
        dct = {k: v.to_data() for k,v in self.items()}

        return dct
    
    def to_dunl(self) -> str:
        return sfd.write_dict(self.to_data())
        
    