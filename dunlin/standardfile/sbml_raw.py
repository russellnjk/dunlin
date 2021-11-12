import simplesbml as ssb

class SBMLRawCode:
    @classmethod
    def read_file(cls, filename):
        ssb_model = ssb.loadSBMLFile(filename)
        return cls(ssb_model, filename)
    
    @classmethod
    def read_string(cls, string, name=''):
        ssb_model = ssb.loadSBMLStr(string)
        return cls(ssb_model, name)
    
    def __init__(self, ssb_model, name=''):
        self.ssb_model = ssb_model
        self.name      = name
    
    def __str__(self):
        return self.ssb_model.toSBML()
    
    def to_data(self):
        pass
    
    def to_file(self):
        code = str(self)
        with open(self.name, 'w') as file:
            file.write(code)
        
        return code