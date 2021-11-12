from abc     import ABC, abstractmethod
from pathlib import Path

class RawCode(ABC):
    name : str
    
    @classmethod
    def read_file(cls, filename):
        '''File -> RawCode
        '''
        pass

    @classmethod
    def read_string(cls, string, name=''):
        '''String -> RawCode
        '''
        pass
    
    def __repr__(self):
        code = str(self)
        if len(code) > 800:
            code = code[:800] + '\n...'
        return type(self).__name__ + f' {self.name} ' + '{\n' + code + '\n}'
    
    @abstractmethod
    def to_data(self):
        pass
    
    @abstractmethod
    def to_file(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
    
