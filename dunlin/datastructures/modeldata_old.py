from abc import ABC, abstractmethod

import dunlin.standardfile.dunl as sfd

class ModelData(dict, ABC):
    '''
    Base class for model data but packs methods for export into input dictionary 
    data. Imitates a normal dictionary but allows attribute-style access. 
    '''
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            msg = f'{type(self).__name__} has no attribute {attr}'
            raise AttributeError(msg)
    
    def __setattr__(self, attr, value):
        msg = f'{type(self).__name__} cannot take on new attributes.'
        raise AttributeError(msg)
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __str__(self):
        s =  f'{type(self).__name__}'+super().__str__()
        return s
    
    def __repr__(self):
        s =  f'{type(self).__name__}'+super().__repr__()
        return s
    
    ###########################################################################
    #Export
    ###########################################################################
    def _to_data(self, keys, recurse=True) -> dict:
        '''When implemented in the subclasses, should only export the core data. 
        '''
        dct = {}
        for k, v in self.items():
            if k not in keys:
                continue
            elif not v:
                
                continue
            
            if hasattr(v, 'to_data') and recurse:
                dct[k] = v.to_data()
            else:
                dct[k] = v
        
        dct = {self['ref']: dct}
        return dct
    
    @abstractmethod
    def to_data(self, recurse=True) -> dict:
        ...
    
    def to_dunl_dictzz(self) -> str:
        dct = self.to_data(recurse=False)
        return dct