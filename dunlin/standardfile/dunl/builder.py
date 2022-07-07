def write_dict(dct_data):
    pass

class MixinAttr:
    @property
    def dunl(self):
        if self._dunl is None:
            if self.writer is None:
                self._dunl = write_dict(self._data)
            else:
                self._dunl = self.writer(self._data)
        else:
            return self._dunl
    
    @dunl.setter
    def dunl(self, string):
        if self._dunl is not None:
            raise AttributeError('This attribute has already been set.')
        
        else:
            self._dunl = string
        
class FileData:
    def __init__(self, *data):
        pass

class Interpolator:
    def __init__(self,):
        pass

class Directory:
    def __init__(self):
        pass

class Element(MixinAttr):
    def __init__(self, data, itype='dict', writer=None):
        self._data  = None
        self._dunl  = None
        self.writer = writer



data0 = {'a' : 'b', 'c' : 'd'}
data1 = {'a' : 'b', 'c' : 'd', 'e' : ['f', 'g']}

e = Element(data0, 'dict')

