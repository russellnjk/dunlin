def make_exvs(func_data, model_data):
    exvs = {exv_name: make_exv(exv_name, func_data, model_data) for exv_name in func_data.get('exvs')}
    return exvs
    
def make_exv(exv_name, func_data, model_data):
    model_key  = model_data['model_key']
    funcs      = func_data['exvs']
    exv_func   = funcs[exv_name]
    exv_data   = model_data['exvs']
    exv_data   = None if exv_data is None else exv_data.get(exv_name, None)
    exv        = EXV(exv_name, exv_func, model_key)
    exv._dun   = exv_data
    return exv

class EXV:
    def __init__(self, name, func, model_key=None):
        self.name      = name
        self.func      = func
        self.model_key = model_key
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def __str__(self):
        return f'{type(self).__name__}<{self.model_key}: {self.name}>'
    
    def __repr__(self):
        return self.__str__()

    
    