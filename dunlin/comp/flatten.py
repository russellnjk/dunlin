import re
from numbers import Number
from typing  import Sequence, Union

import dunlin.utils as ut


    
    
def dot(x          : Union[dict, list, str, Number], 
        child_name : str, 
        rename     : dict, 
        delete     : set[str],
        recurse    : list[bool]
        ) -> Union[list, dict]:
    '''
    This function recursively performs rename/delete operations on the 
    argument `x`. As the names imply, the `rename` and `delete` arguments 
    specify the information related to renaming and deletion. When a list or 
    dictionary is encountered in `x`, the function is called recursively with 
    that list/dictionary. 
    
    The behaviour of the function at each level is controlled by the `recurse` 
    argument. At each level, only the first value in `recurse` is used; the 
    remainder (i.e. `recurse[1:]`) is passed to the next level of recursion.
    
    Assume that `x` is a dictionary with multiple layers. If the first value of 
    `recurse `is `True`, the rename/delete operations are applied to the 
    first layer. Otherwise the first layer remains the same. If the second value 
    of `recurse` is `True`, the rename/delete operations are also applied to 
    the second layer. You get the idea. This also means that the length of 
    `recurse` must equal the depth of the deepest branch of `x`. 
    
    Lists are parsed in a similar way but they are treated as values in 
    dictionaries; each element in the list is passed into the next recursion 
    level with `recurse[1:]`. In other words `x=["a", "b", "c"]` requires 
    `recurse` to have a length of 2, similar to `x={"t": "a", "u": "b", "v": "c"}`. 
    However, the first value no longer matters since the list has no keys. 
    
    It is not possible to use `recurse` in a way to create unique behaviour for 
    each branch of the dictionary. Introducing such a feature would complicate
    the algorithm unecessarily. Downstream development should work with data 
    formats that only require uniform rules for level of nesting.
    
    From the front-end, x can only be a dictionary or list. However, during 
    recursion, strings and numbers will also be passed in. These two types of 
    data will always be renamed.
    
    Parameters
    ----------
    x : Union[dict, list]
        The container containing child data which has keys and values that 
        require renaming/deletion before integrating with the parent model.
    child_name : str
        Needed to generate detailed error messages.
    rename : dict
        Maps a name in the data (if found) to a new name during substitution.
    delete : set
        A set of names not to include when flattening.
    recurse : list[bool]
        Determines the levels of recurions where renaming/deletion are to be 
        carried out.

    Returns
    -------
    (Union[list, dict])
        A dictionary or list with the elements renamed/deleted.

    '''
    
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
        args   = child_name, rename, delete, recurse[1:]
        result = []
        for i in x:
            if not isinstance(i, (Number, str, list, tuple, dict)):
                msg  = 'Encountered an unexpected type when flattening model.'
                msg += f' Received: {x}'
                raise TypeError(msg)    
            else:
                i_ = dot(i, *args)
            
            result.append(i_)
         
        return result
    
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
        