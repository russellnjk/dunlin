class FrozenObject:  
    _frozen = False
    
    def __setattr__(self, attr, value):
        if attr == '_frozen':
            super.__setattr__(self, attr, value)
        elif self._frozen:
            msg = f'{type(self).__name__} is locked and cannot be modified.'
            raise AttributeError(msg)
        else:
            super.__setattr__(self, attr, value)
    
    def freeze(self):
        self._frozen = True
    
    def unfreeze(self):
        self._frozen = False