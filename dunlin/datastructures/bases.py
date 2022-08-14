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
        return self.to_dunl()
    
    def __repr__(self):
        return f'{type(self).__name__} {self.name}({str(self)})'
    
    def __contains__(self, name: str):
        return name in self.namespace
    
    @abstractmethod
    def to_data(self) -> Union[list, dict]:
        ...

class GenericDict(ut.FrozenObject):
    '''
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
    
    def __init__(self, ext_namespace, mapping: dict, *args):
        _data = {}
        for name, kwargs in mapping.items():
            if hasattr(kwargs, 'items'):
                item = self.itype(ext_namespace, *args, name, **kwargs)
            else:
                item = self.itype(ext_namespace, *args, name, *kwargs)
            _data[name] = item
          
        self._data = _data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __iter__(self):
        return iter(self._data)
    
    def __contains__(self, key):
        return key in self._data
    
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
    
class NamespaceDict(ut.FrozenObject, ABC):
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

class TabularDict(ABC, ut.FrozenObject):
    '''
    Base class for containers for tabular data.
    
    
    
    '''
    is_numeric: bool = True
    
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

    def __init__(self, 
                 ext_namespace: set, 
                 name: str, 
                 mapping: Union[dict, pd.DataFrame], 
                 n_format: Callable = sfd.format_num
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

