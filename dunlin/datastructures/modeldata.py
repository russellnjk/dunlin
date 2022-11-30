from abc import ABC, abstractmethod

import dunlin.utils as ut
    
class ModelData(ut.FrozenObject, ABC):
    '''
    Base class for model data. Contains templated functions for export.
    
    '''

    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        s =  f'{type(self).__name__}'
        d = [f'{attr} : {getattr(self, attr)}' for attr in self._attrs if hasattr(self, attr)]
        d = '{\n' + ', '.join(d) + '\n}'
        s = f'{s}{d}'
        
        return s
    
    def __repr__(self):
        return str(self)
    
    ###########################################################################
    #Export
    ###########################################################################
    def _to_data(self, keys, recurse=True) -> dict:
        
        data = {}
        for key in keys:
            value = getattr(self, key, None)
            
            if value is None:
                continue
            elif not len(value):
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