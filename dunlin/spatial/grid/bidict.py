from abc    import ABC, abstractclassmethod
from typing import GenericAlias

###############################################################################
#Base/Helper Classes
############################################################################### 
class _BiDict(ABC):
    def __init__(self, src, dst, *args, **kwargs):
        if type(src) != str:
            raise TypeError('src must be a string.')
        if type(dst) != str:
            raise TypeError('dst must be a string.')
        
        self.name = f'{src}->{dst}'
        forward   = {}
        backward  = {}
        
        self._forward      = forward
        self._backward     = backward
        self._inverse      = Inverse(self)
        self._inverse.name = f'{dst}->{src}'
        
        #Update the dictionary
        temp = dict(*args)
        temp = {**temp, **kwargs}
            
        for key, value in temp.items():
            self[key] = value

    ###########################################################################
    #Forward Access/Modification
    ###########################################################################
    def __contains__(self, key):
        return key in self._forward
    
    def __len__(self):
        return len(self._forward)
    
    def __iter__(self):
        return iter(self._forward)
    
    def __eq__(self, other):
        return self._forward == other
    
    def __getitem__(self, key):
        return self._forward[key]
    
    def get(self, key, default=None):
        return self._forward.get(key, default)
    
    @abstractclassmethod
    def pop(self, key):
        pass
    
    @abstractclassmethod
    def __setitem__(self, key):
        pass
    
    @abstractclassmethod
    def _inverse_pop(self, value):
        pass

    def setdefault(self, key, default):
        try:
            value = self._forward[key]
        except KeyError:
            value     = default
            self[key] = value
        except Exception as e:
            raise e
            
        return value
    
    def keys(self):
        return self._forward.keys()
    
    def values(self):
        return self._forward.values()
    
    def items(self):
        return self._forward.items()
    
    ###########################################################################
    #Backward Access/Modification
    ###########################################################################
    @property
    def inverse(self):
        return self._inverse
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __repr__(self):
        return self.name + repr(dict(self))
    
    def __str__(self):
        return self.name + str(dict(self))
    
    ###########################################################################
    #Type Hinting
    ###########################################################################
    def __class_getitem__(cls, key):
        key_type, value_type = key

        return GenericAlias(cls, (key_type, value_type))
        
class Inverse:
    def __init__(self, mapping):
        self._mapping  = mapping
        self._backward = mapping._backward
    
    ###########################################################################
    #Backward Access/Modification
    ###########################################################################
    def __contains__(self, value):
        return value in self._backward
    
    def __len__(self):
        return len(self._backward)
    
    def __iter__(self):
        return iter(self._backward)
    
    def __eq__(self, other):
        return self._backward == other
    
    def __getitem__(self, key):
        return self._backward[key]
    
    def get(self, key, default=None):
        return self._backward.get(key, default)
    
    def pop(self, value):
        return self._mapping._inverse_pop(value)
    
    def setdefault(self, value, default):
        try:
            key = self[value]
        except KeyError:
            key         = default
            self[value] = key
        except Exception as e:
            raise e
        
        return key
    
    def keys(self):
        return self._backward.keys()
    
    def values(self):
        return self._backward.values()
    
    def items(self):
        return self._backward.items()
    
    ###########################################################################
    #Foreward Access/Modification
    ###########################################################################
    @property
    def inverse(self):
        return self._mapping
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __repr__(self):
        return self.name + repr(dict(self))
    
    def __str__(self):
        return self.name + str(dict(self))
        
###############################################################################
#User Classes
###############################################################################  
class One2One(_BiDict):
    def pop(self, key):
        value = self._forward.pop(key)
        self._backward.pop(value)
        
        return value
    
    def __setitem__(self, key, value):
        if key in self:
            self.pop(key)

        self._forward[key]    = value
        self._backward[value] = key
    
    def _inverse_pop(self, value):
        key = self._backward.pop(value)
        self._forward.pop(key, )
        return key
    
class One2Many(_BiDict):
    def pop(self, key):
        value = self._forward.pop(key)
        self._backward[value].remove(key)
        if not self._backward[value]:
            self._backward.pop(value)
        
        return value
    
    def __setitem__(self, key, value):
        if key in self:
            self.pop(key)
            
        self._forward[key] = value
        self._backward.setdefault(value, set()).add(key)
    
    def _inverse_pop(self, value):
        keys = self._backward.pop(value)
        
        for key in keys:
            self._forward.pop(key)
            
        return keys
    