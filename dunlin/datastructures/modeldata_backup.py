from abc import ABC, abstractmethod

import dunlin.standardfile.dunl as sfd
    
class ModelData(ABC):
    '''
    Base class for model data but packs methods for export into input dictionary 
    data. 
    '''
    
    _attrs: dict[str, type]
    
    ###########################################################################
    #Subclassing
    ###########################################################################
    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, '_attrs'):
            msg = f'Subclass {cls.__name__} could not be created because it is missing the _attrs attribute.'
            
            raise AttributeError(msg)
            
        for attr, allowed in cls._attrs.items():
            if type(attr) != str:
                msg = f'Keys in _attrs must be strings. Received {type(attr)} under key {attr}.'
            elif not hasattr(allowed, '__iter__'):
                msg = f'Values in _attrs must be iterable. Received {type(allowed)} under key {attr}.'
            
    ###########################################################################
    #Access
    ###########################################################################
    def __setattr__(self, attr, value):
        try:
            allowed = self._attrs[attr]
        except KeyError:
            msg = f'{type(self).__name__} cannot take on new attributes.'
            raise AttributeError(msg)
        except Exception as e:
            raise e
        
        for t in allowed:
            if type(t) == type: 
                if isinstance(value, t):
                    super().__setattr__(attr, value)
                    return
            elif t == value:
                super().__setattr__(attr, value)
                return
        
        msg = f'Expected one of {allowed}. Received a {type(value)}.'
        raise ValueError(msg)
    
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
    @abstractmethod
    def to_data(self) -> dict:
        ...
    
    @abstractmethod
    def to_dunl(self) -> str:
        ...
        
        # dct  = {}
        # _dct = self.__dict__
        
        # for attr in self._attrs:
        #     try:
        #         value = _dct[attr]
        #     except KeyError:
        #         msg = f'{type(self)} is missing attribute {attr}.'
        #         raise AttributeError(msg)
        #     except Exception as e:
        #         raise e
            
        #     if hasattr(value, 'to_data'):
        #         dct[attr] = value.to_data()
        #     else:
        #         dct[attr] = value
        
        # return dct
