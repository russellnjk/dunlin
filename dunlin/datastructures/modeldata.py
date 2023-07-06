from typing import Any, Union

import dunlin.comp  as cmp
    
class ModelData:
    '''
    Base class for model data. Contains templated functions for export.
    '''
    @classmethod
    def from_all_data(cls, all_data, ref):
        flattened  = cmp.flatten_ode(all_data, ref)
        
        return cls(**flattened)
    
    def __init__(self, exportable_attributes: list[Union[str, tuple[str]]]):
        self.__exportable_attributes = tuple(exportable_attributes)
        
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        #Will not work if to_data has not been implemented or without ref attribute 
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
    def to_data(self, recurse=True) -> dict:
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
            
            #Recursively call to_data if necessary
            if recurse and hasattr(value, 'to_data'):    
                value = value.to_data()
            
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
        data = self.to_data(recurse=False)
        ref  = data.pop('ref')
        return {ref: data}