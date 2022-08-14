import dunlin.standardfile.dunl as sfd

class ModelData(dict):
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
    def to_data(self, keys) -> dict:
        '''When implemented in the subclasses, should only export the core data. 
        '''
        dct = {}
        for k, v in self.items():
            if k not in keys:
                continue
            elif not v:
                
                continue
            
            if hasattr(v, 'to_data'):
                dct[k] = v.to_data()
            else:
                dct[k] = v
        
        return dct
    
    def to_dunl_dict(self) -> str:
        dct = self.to_data()
        dct.pop('ref', None)
        
        return dct