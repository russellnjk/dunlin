class Base:
    
    def __init__(self, uid, name, container=None):
        #Check that uid is unique
        if container is not None:
            if uid in container:
                raise NameError('IDs must be unique.')
            else:
                container.add(uid)
        
        self.uid  = uid
        self.name = name