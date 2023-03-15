import numpy  as np
import pandas as pd
from abc    import ABC, abstractmethod
from typing import (Any, Callable, Optional, Union, 
                    ItemsView, KeysView, ValuesView, Iterable
                    )


import dunlin.utils             as ut
import dunlin.standardfile.dunl as sfd
from dunlin.utils.typing import Dflike, Num

class GenericItem(ABC, ut.FrozenObject):
    '''For most items.
    
    Attributes:
        1. `name`: All items in models will have `name` attributes corresponding 
        SIds in SBML while obeying this package's conventions. Names should be 
        unique so that each object can be referred to unambiguously. This is 
        ensured by the `ext_namespace` argument in the constructor; when the 
        constructor `__init__` is called, an error will occur if the `name` 
        argument is already in `ext_namespace`.
        
        2. `namespace`: Many items refer to other items in a model. For example, 
        a rate law might involve several parameters and molecular concentrations. 
        To keep track of this, each `GenericItem` has a namespace attribute which 
        tells the developer what other items are being referred to. The 
        `__contains__` dunder method accepts a string and checks if that string 
        is inside namespace.
        
        However, not all items require such tracking. In such cases, the `namespace` 
        attribute should be left as an empy tuple. If the item requires some other 
        behaviour, the dunder method can be overridden. However, a different 
        attribute other than `namespace` should be used so as to avoid confusion.
    
    Export methods:
        1. `to_data` : Returns a dict/list that can be used to re-instantiate 
        the object not including the `name` argument.
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
    
    def __init__(self, ext_namespace, name, /, **data):
        for k, v in data.items():
            setattr(self, k, v)
        
        if not hasattr(self, 'namespace'):
            self.namespace = ()
        
        if not ut.is_valid_name(name):
            msg = f'Invalid name {name} provided when instantiating {type(self).__name__} object.'
            raise ValueError(msg)
        
        if name in ext_namespace:
            raise NameError(f'Redefinition of {name}.')
        else:
            #Update the namespace
            ext_namespace.add(name)
        
        self.name = name
    
    def __str__(self):
        return f'{type(self).__name__}({repr(self.name)})'
    
    def __repr__(self):
        return str(self)
    
    def __contains__(self, name: str):
        return name in self.namespace
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    @abstractmethod
    def to_data(self) -> Union[list, dict]:
        ...

class GenericDict(ut.FrozenObject):
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
    
    _data: dict 
    itype: type
    
    def __init__(self, 
                 ext_namespace : set, 
                 mapping       : dict, 
                 *args
                 ) -> None:
        _data = {}
        
        if mapping:
            for name, kwargs in mapping.items():
                if hasattr(kwargs, 'items'):
                    try:
                        item = self.itype(ext_namespace, *args, name, **kwargs)
                    except Exception as e:
                        self._raise(e, name)
                
                elif ut.islistlike(kwargs):
                    try:
                        item = self.itype(ext_namespace, *args, name, *kwargs)
                    except Exception as e:
                        self._raise(e, name)
                
                else:
                    try:
                        item = self.itype(ext_namespace, *args, name, kwargs)
                    except Exception as e:
                        self._raise(e, name)
                
                _data[name] = item
          
        self._data = _data
        
    def _raise(self, e, name):
        if not e.args:
            raise e
            
        msg = e.args[0]
        n   = self.itype.__name__
        msg = f'Error when parsing {n} "{name}"\n{msg}'
        
        raise type(e)(msg)
    
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
    def to_data(self) -> dict:
        dct = {k: v.to_data() for k,v in self.items()}
        return dct
    
    def to_dunl(self) -> str:
        return sfd.write_dict(self.to_data())
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        return self.to_dunl()
    
    def __repr__(self):
        return f'{type(self).__name__}{tuple(self.keys())}'
    
class NamespaceDict(GenericDict):
    '''
    Extends the GenericDict class with the addition of a `namespace` attribute. 
    This attribute is used to keep track of namespaces to prevent overlaps. 
    The user can set their own `namespace` but an empty one will be created 
    otherwise. The contents of `namespace` depend on the user's needs.
    '''
    itype     : type
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 ext_namespace: set, 
                 mapping      : dict, 
                 *args
                 ) -> None:
        super().__init__(ext_namespace, mapping, *args)
        
        if not hasattr(self, 'namespace'):
            self.namespace = ()

class TabularDict(ABC, ut.FrozenObject):
    '''
    Base class for containers for tabular data. 
    '''
    is_numeric   : bool  = True
    can_be_empty : bool = True
    
    @staticmethod
    def mapping2df(mapping: Dflike, ):
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
        elif not mapping:
            df = pd.DataFrame()
        else:
            msg = f"Expected a DataFrame, Series or dict. Received {type(mapping)}"
            raise TypeError(msg)
    
        return df

    def __init__(self, 
                 ext_namespace : set, 
                 name          : str, 
                 mapping       : Union[dict, pd.DataFrame], 
                 n_format      : Callable = sfd.format_num
                 ) -> None:
        #Convert to df
        df = self.mapping2df(mapping)
        
        if 0 in df.shape and not self.can_be_empty:
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
    def names(self) -> tuple:
        return tuple(self._df.columns)
    
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
    
    def __getitem__(self, key):
        return self._df[key]
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> dict:
        return self._df.to_dict()
    
    def to_dunl(self) -> str:
        df = self._df
        if self.is_numeric:
            n_format = self.n_format
            
            return sfd.write_numeric_df(df, n_format)
        
        else:
            return sfd.write_non_numeric_df(df)

