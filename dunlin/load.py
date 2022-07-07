from collections import namedtuple

import dunlin.standardfile as dsf
import dunlin.ode          as dom
import dunlin.data         as ddt

parsed_result = namedtuple('LoadedData', 'all_data parsed')

def load_file(*filenames):
    
    all_data = dsf.read_dunl_file(*filenames)
    trd_refs = {}
    parsed   = {}
    
    for ref, data in all_data.items():
        dtype = data.get('dtype', 'ode')
        
        if dtype == 'ode':
            temp        = dom.ODEModel.from_data(all_data, ref) 
            parsed[ref] = temp
        elif dtype == 'time_response_data':
            trd_refs[ref] = data
        elif dtype == 'generic':
            trd_refs[ref] = data
        else:
            raise ValueError(f'Unexpected data type "{dtype}"')
    
    #Instantiate data sets
    for ref, kwargs in trd_refs.items():
        #Determine the model argument
        if 'model' in kwargs:
            model_ref = kwargs['model']
            
            if model_ref in parsed:
                model = parsed[model_ref]
            else:
                msg = 'TimeResponseData {} requires a missing model : {}'
                raise ValueError(msg.format(ref, model_ref))
        else:
            model = None
        
        #Extract the other args and call the constructor and then update
        skip        = ['model']
        kwargs      = {k: v for k, v in kwargs.items() if k not in skip}
        trd         = ddt.TimeResponseData.load_files(**kwargs, model=model)
        parsed[ref] = trd 
        
        if model is None:
            trd.ref = ref
        
    return parsed_result(all_data, parsed)