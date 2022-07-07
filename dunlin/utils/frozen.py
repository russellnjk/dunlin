class FrozenObject:   
    def __setattr__(self, attr, value):
        frozen = getattr(self, '_frozen', False)
        
        if frozen:
            msg = f'{type(self).__name__} is locked and cannot be modified.'
            raise AttributeError(msg)
        else:
            super.__setattr__(self, attr, value)
    
    def freeze(self):
        self._frozen = True