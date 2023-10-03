from typing      import Callable

from .standardfile import read_dunl_file 
from .ode          import ODEModel
from .optimize     import read_time_response

def load_file(filename, loader: Callable=read_dunl_file) -> tuple[dict, dict]:
    
    raw          = loader(filename)
    instantiated = {}
    
    for ref, data in raw.items():
        dtype = data.get('dtype', 'ode')
        
        if dtype == 'ode':
            instantiated[ref] = ODEModel.from_data(raw, ref) 
        elif dtype == 'time_response':
            instantiated[ref] = read_time_response(data)
        elif dtype == 'generic':
            instantiated[ref] = data
        else:
            raise ValueError(f'Unexpected data type "{dtype}"')
            
    return raw, instantiated

