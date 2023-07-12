import numpy  as np
import pandas as pd
from abc      import ABC, abstractmethod
from datetime import datetime
from numbers  import Number
from typing   import (Any, Callable, Optional, Union, 
                      ItemsView, KeysView, ValuesView, Iterable
                      )


import dunlin.utils             as ut
import dunlin.standardfile.dunl as sfd
from dunlin.standardfile.dunl    import writeelement as we
from dunlin.utils.typing import Dflike, Num

'''
Base classes to use:
    1. Table: The data is of a tabular nature. 
    2. DataDicts with DataValue: The data is a collection of dict/list-like 
    items.
    3. Tree: The data is dict-like and recursive
    
'''
class DataValue(ABC):
    '''For most items.
    
    Attributes:
        1. `name`: Most items in models need to be uniquely identified. This 
        is achieved via the `name` attribute which obeys SBML conventions. 
        During instantiation, `name` is checked against the `all_namespace` 
        argument in the constructor. An exception is raised if `name` is in 
        `ext_namespace`. Certain items may not require a unique identifier. In 
        this case, the `name` argument should be None.
        
    Export methods:
        1. `to_data` : Returns a dict/list that can be used to re-instantiate 
        the object.
    '''
    
    #For formatting numbers when converting to code
    n_format: Callable = sfd.format_num
    
    @staticmethod
    def primitive2string(x: Union[str, int, float]):
        if ut.isstrlike(x):
            return x.strip()
        
        float_value = float(x)
        int_value   = int(x)
        if float_value == int_value:
            return str(int_value)
        else:
            return str(float_value)
    
    def __init__(self, all_names: set, name: str, **attributes):
        #Set attributes
        self.name = name
    
        for k, v in attributes.items():
            setattr(self, k, v)
        
        if name is None:
            pass
        elif not ut.is_valid_name(name):
            msg = f'Invalid name {name} provided when instantiating {type(self).__name__} object.'
            raise NameError(msg)
        elif name in all_names:
            raise NameError(f'Redefinition of {name}.')
        else:
            #Update the namespace
            all_names.add(name)
    
    ###########################################################################
    #Attribute Management
    ###########################################################################
    def __setattr__(self, attr: str, value: Any) -> None:
        if hasattr(self, attr):
            msg = f'Attribute {attr} has already been set and cannot be modified.'
            raise AttributeError(msg)
        else:
            super().__setattr__(attr, value)
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self) -> str:
        return f'{type(self).__name__}({repr(self.name)})'
    
    def __repr__(self) -> str:
        return str(self)
    
    ###########################################################################
    #Export
    ###########################################################################
    @abstractmethod
    def to_dict(self) -> dict:
        ...
    
    def to_dunl_elements(self) -> str:
        dct = self.to_dict()
        return we.write_dict(dct)
        
class DataDict(ABC):
    '''
    Allows dict-like access and iteration but bans modification. Also enforces 
    that all items inside it are of the the type specified by the `itype` 
    attribute. Finally, it contains methods for export into regular Python 
    dictionaries and dunl code.
    
    Attributes:
        1.`_data`: Stores data in dictionary form. Used with the `__getitem__`, 
        '__iter__', `__contains__`, `keys`,  `values` and `items` methods to 
        duck-type the object as frozen dictionary.
        2. `itype`: The `type` of the objects to be stored as values. During 
        instantiation, the values of the `_data` will be created by calling the 
        `__init__` of `itype`.
        
    '''
    #Intialized in this class
    _data: dict 
    
    #Specified in subclass
    itype: type
    
    def __init__(self, 
                 ext_namespace : set, 
                 mapping       : dict, 
                 *args
                 ) -> None:
        
        #Check mapping and edge cases
        if type(mapping) != dict:
            msg = f'Expected a dict. Received {type(mapping)}.'
            raise TypeError(msg)
        elif not mapping:
            self._data = {}
            return
        
        #Instantiate itype for each item in the mapping
        _data = {}
        for name, kwargs in mapping.items():
            if type(kwargs) == dict:
                item = self.itype(ext_namespace, *args, name, **kwargs)
            
            elif type(kwargs) == list or type(kwargs) == tuple:
                item = self.itype(ext_namespace, *args, name, *kwargs)
            
            elif isinstance(kwargs, (Number, str, datetime)):
                item = self.itype(ext_namespace, *args, name, kwargs)
            
            else:
                msg = f'Mapping values must be of type list or dict. Received {type(kwargs)}.'
                raise TypeError(msg)
                
            _data[name] = item
          
        self._data = _data
    
    ###########################################################################
    #Attribute Management
    ###########################################################################
    def __setattr__(self, attr: str, value: Any) -> None:
        if hasattr(self, attr):
            msg = f'Attribute {attr} has already been set and cannot be modified.'
            raise AttributeError(msg)
        else:
            super().__setattr__(attr, value)
            
    ###########################################################################
    #Access
    ###########################################################################
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __iter__(self):
        return iter(self._data)
    
    def __contains__(self, key):
        return key in self._data
    
    def __len__(self):
        return len(self._data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        dct = {}
        for v in self.values():
            dct.update(v.to_dict())
        return dct
    
    def to_dunl_elements(self, **kwargs) -> str:
        #Currently ignores kwargs
        chunks = [v.to_dunl_elements() for v in self.values()]
        code   = '\n'.join(chunks)
        
        return code
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f'{type(self).__name__}{tuple(self.keys())}'
 
class Tree(DataValue):
    recurse_at : str
    def __init__(self, all_names: set, name: str, *args, **attributes):
        
        recurse_at = self.recurse_at
        
        if attributes.get(recurse_at):
            children = {}
            for child_name, child_attributes in attributes[recurse_at].items():
                children[child_name] = type(self)(all_names, 
                                                  child_name,
                                                  *args,
                                                  **child_attributes
                                                  )
            
        else:
            children = None
        
        attributes_ = {**attributes, recurse_at: children}
        
        super().__init__(all_names, name, **attributes_)
    
class Table(ABC):
    '''
    Base class for tabular data. 
    '''
    #Specified in the subclass
    itype        : str
    
    #Can be overwritten in the subclass but cannot be modified
    is_numeric   : bool = True
    can_be_empty : bool = True
    
    #Underlying data
    _df: pd.DataFrame
    
    def __init__(self, 
                 all_names : set, 
                 mapping   : Union[dict, pd.DataFrame], 
                 ) -> None:
        
        #Convert to df
        if type(mapping) == pd.DataFrame:
            df = mapping
        elif type(mapping) == pd.Series:
            df = pd.DataFrame(mapping).T
        elif type(mapping) == dict:
            df = pd.DataFrame(mapping)
        else:
            msg = f"Expected a DataFrame, Series or dict. Received {type(mapping)}"
            raise TypeError(msg)
        
        #Check for edge case
        if 0 in df.shape and not self.can_be_empty:
            msg  = 'Error in parsing {self.itype}. '
            msg += 'The input appears to be empty. '
            raise ValueError(msg)
        
        #Check names
        names = tuple(df.columns)
        for name in names:
            ut.check_valid_name(name) 
            
            if type(name) != str:
                msg  = 'Error in parsing {self.itype}. '
                msg += 'Keys in {self.itype} must be strings.'
                raise TypeError(msg)
            elif not ut.is_valid_name(name):
                msg = f'Invalid name {name} provided when instantiating {self.itype}.'
                raise NameError(msg)
            elif name in all_names:
                raise NameError(f'Redefinition of {name}.')
            else:
                #Update the namespace
                all_names.add(name)
         
        #Save attributes
        self._df = df
    
    ###########################################################################
    #Attribute Management
    ###########################################################################
    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in {'n_format', '_df'}:
            super().__setattr__(attr, value)
        elif hasattr(self, attr):
            msg = f'Attribute {attr} has already been set and cannot be modified.'
            raise AttributeError(msg)
        else:
            super().__setattr__(attr, value)
            
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self) -> str:
        return self.to_dunl()
    
    def __repr__(self) -> str:
        return f'{type(self).__name__}{tuple(self.names)}'
    
    ###########################################################################
    #Access/Modification
    ###########################################################################
    @property
    def names(self) -> tuple:
        return tuple(self._df.columns)
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @df.setter
    def df(self, new_df: pd.DataFrame) -> None:
        names   = self.names
        columns = new_df.columns
        if len(set(names).intersection(columns)) != len(columns):
            msg = 'Column names do not match.'
            msg = f'{msg} Expected: {names}\nReceived: {tuple(columns)}'
            raise ValueError(msg)
        
        self._df = new_df[list(names)]
    
    def __iter__(self):
        return iter(self.names)
    
    def __contains__(self, item: str) -> bool:
        return item in self.names
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, key):
        return self._df[key]
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        df = self._df
        if all(df.index == list(range(0, len(df.index )))):
            return df.to_dict('list')
        else:
            return df.to_dict()
            
    
    def to_dunl_elements(self, n_format: Callable=sfd.format_num) -> str:
        #kwargs are ignored
        
        df = self._df
        if self.is_numeric:
            return sfd.write_numeric_df(df, n_format)
        
        else:
            return sfd.write_non_numeric_df(df)


