from abc    import ABC, abstractclassmethod
from numbers                import Number
from datetime import datetime

from typing import Any
    
class ModelData(ABC):
    '''
    Base class for model data. Contains templated functions for export.
    '''
    @abstractclassmethod
    def from_all_data(cls, all_data, ref):
        ...
    
    def _set_exportable_attributes(self, exportable_attributes: list[str|tuple[str]]):
        self.__exportable_attributes = tuple(exportable_attributes)
    
    @classmethod
    def deep_copy(cls, 
                  name   : str, 
                  data   : dict|list|str|Number|datetime|None,
                  _first : bool = True
                  ) -> Any:
        if data is None and _first:
            return data

        elif type(data) == dict:
            result = {}
            for k, v in data.items():
                k_ = cls.deep_copy(name, k, False)
                v_ = cls.deep_copy(name, v, False)
                
                result[k_] = v_
            return result
        
        elif type(data) == list or type(data) == tuple:
            return [cls._deep_copy(name, x, False) for x in data]
        elif isinstance(data, (Number, str, datetime)):
            return data
        else:
            msg  = 'Error when parsing {name}. '
            msg += 'Expected a dict, list, str, number or datetime. '
            msg += f'Received {type(data)}.'
            raise TypeError(msg)
            
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        #Will not work if to_dict has not been implemented or without ref attribute 
        s =  f'{type(self).__name__}'
        return s
    
    def __repr__(self):
        return str(self)
    
    ###########################################################################
    #Attribute Management
    ###########################################################################
    def __setattr__(self, attr: str, value: Any) -> None:
        if hasattr(self, attr):
            msg = f'Attribute {attr} has already been set and cannot be modified.'
            raise AttributeError(msg)
        else:
            super().__setattr__(attr, value)
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        self.key = value
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self, recurse=True) -> dict:
        def is_empty(x):
            try:
                len(x)
                return not len(x)
            except:
                return True
          
        data = {}
        for attribute in self.__exportable_attributes:
            #Extract the associated value
            if type(attribute) == str:
                value = getattr(self, attribute)
            else:
                value = getattr(self, attribute[-1])
            
            #Skip empty values including None
            if is_empty(value):
                continue
            
            #Recursively call to_dict if necessary
            if recurse and hasattr(value, 'to_dict'):    
                value = value.to_dict()
            
            #Assign the attribute and value
            if type(attribute) == str:
                data[attribute] = value
            else:
                current = data
                for i in attribute:
                    current = current.setdefault(i, {})
                
                current[i] = value
        
        return data
                
    def to_dunl_dict(self) -> dict:
        data = self.to_dict(recurse=False)
        ref  = data.pop('ref')
        return {ref: data}