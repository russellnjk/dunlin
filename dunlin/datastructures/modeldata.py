from abc import ABC, abstractmethod

import dunlin.utils as ut
    
class ModelData(ut.FrozenObject, ABC):
    '''
    Base class for model data. Contains templated functions for export.
    
    '''
    ref : str
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        #Will not work if to_data has not been implemented or without ref attribute 
        s =  f'{type(self).__name__}'
        d = self.to_data(recurse=False)[self.ref]
        s = f'{s}{d}'
        
        return s
    
    def __repr__(self):
        return str(self)
    
    ###########################################################################
    #Representation
    ###########################################################################
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
    def _to_data(self, keys, recurse=True) -> dict:
        def is_empty(x):
            if hasattr(x, '__len__'):
                return len(x) == 0
            else:
                return False
            
        
        data = {}
        for key in keys:
            value = getattr(self, key, None)
            
            if value is None:
                continue
            elif is_empty(value):
                continue
            elif recurse and hasattr(value, 'to_data'):
                data[key] = value.to_data()
            else:
                data[key] = value
       
        data = {self.ref: data}
        return data
    
    @abstractmethod
    def to_data(self) -> dict:
        ...
    
    def to_dunl_dict(self) -> dict:
        return self.to_data(recurse=False)