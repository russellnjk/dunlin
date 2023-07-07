import re
from numbers import Number
from typing  import Sequence, Union

import dunlin.utils as ut


    
    
def dot(x          : Union[dict, list, str, Number], 
        child_name : str, 
        rename     : dict, 
        delete     : set,
        recurse    : list[bool]
        ) -> Union[str, list[str], dict]:
    
    pattern = '\.{0,1}[a-zA-Z_][\w_]*'
    def repl(x):
        name = x[0]
        if name[0] == '.':
            return name
        
        elif name in ut.reserved:
            return name
        
        elif name in rename:
            return rename[name]
        
        else:
            return child_name + '.' +name
    
    sub = lambda s: re.sub(pattern, repl, s)
    
    if type(x) == dict:
        args = child_name, rename, delete, recurse[1:]
        if recurse[0]:
            return {dot(k, *args): dot(v, *args) for k, v in x.items() if k not in delete}
        else:
            return {k: dot(v, *args) for k, v in x.items() if k not in delete}
        
    elif type(x) == list:
        if recurse[0]:
            result = []
            for i in x:
                if i in delete:
                    continue
                else:
                    i_ = sub(i)
                
                result.append(i_)
            
            return result
        else:
            return list(x)
    
    elif type(x) == str:
        
        return sub(x)
    
    elif isinstance(x, Number):
        return x
    
    else:
        msg = f'Ecountered unexpected type {type(x)} when flattening model.'
        msg = f'The item has value: {x}'
        raise TypeError(msg)

def flatten(all_data        : dict, 
            required_fields : dict[str, callable], 
            parent_ref      : str, 
            hierarchy       : Sequence = ()
            ) -> dict:
   
    parent_data = all_data[parent_ref]
    submodels   = parent_data.get('submodels', {})
    hierarchy   = [*hierarchy, parent_ref]
    
    flattened = {}
    
    for child_name, child_config in submodels.items():
        #Extract child config
        child_ref    = child_config['ref']
        delete       = child_config.get('delete', [])
        rename       = child_config.get('rename', {})
        
        #Check hierarchy is not circular
        if child_ref in hierarchy:
            raise CircularHierarchy(*hierarchy, child_ref)
        elif child_ref not in all_data:
            raise MissingModel(child_ref)
        
        #Get child data. This must be done recursively as the child
        #may contain submodels
        child_data = flatten(all_data, required_fields, child_ref, hierarchy)
        
        #Rename and delete
        submodel_data = rename_delete(child_name, child_data, rename, delete, required_fields) 
        
        #Update the flattened data
        for key in submodel_data:
            flattened.setdefault(key, {}).update(submodel_data[key])
    
    #Update flattened with the parent data. Overwriting is expected.
    for key, value in parent_data.items():
        if key == 'submodels':
            continue
        elif key in flattened:
            flattened[key].update(value)
        else:
            flattened[key] = value
    
    return flattened
        
def rename_delete(child_name      : str, 
                  child_data      : dict, 
                  rename          : dict, 
                  delete          : set, 
                  required_fields : dict[str, list[bool]]
                  ) -> dict:
    
    delete_ = set(delete)
    
    #Check that delete and rename are mutually exclusive
    overlap = delete_.intersection(rename)
    if overlap:
        msg = f'Overlap between delete and rename: {overlap} '
        raise ValueError(msg)
    
    #Iterate through fields
    #Delete and rename
    submodel_data = {}
    
    for field, recurse in required_fields.items():
        if type(field) == str:
            old_value = child_data.get(field, None)
        else:
            current = child_data
            for i in field:
                current = current.get(i, {})
                
            old_value = current
            
        if not old_value:
            continue
        
        new_value = dot(old_value, child_name, rename, delete_, recurse)
        
        if type(field) == str:
            submodel_data[field] = new_value
        else:
            current = submodel_data
            for i in field[:-1]:
                current = current.setdefault(i, {})
            
            current[field[-1]] = new_value
        
    return submodel_data

class MissingModel(Exception):
    pass

class CircularHierarchy(Exception):
    def __init__(self, *model_refs):
        joined = ' -> '.join(model_refs)
        super().__init__(joined)
        