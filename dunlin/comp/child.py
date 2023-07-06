import re
import dunlin.utils as ut

def wrap_merge(key):
    def outer(function):
        def inner(parent_data, *args, **kwargs):
            result       = function(*args, **kwargs)
            parent_datum = parent_data.get(key, {})
            
            result = {**result, **parent_datum}
            return result
        inner.key = key
        return inner
    return outer

def make_child_item(x, child_name, rename=None, delete=None, max_depth=10):
    if ut.isnum(x):
        #Numbers and string corresponding to numbers are returned as-is
        return x
    elif type(x) != str:
        raise TypeError(f'Unexpected type: {type(x).__name__}')
    
    string  = x
    rename = {} if rename is None else rename
    delete = () if delete is None else delete
    pattern = '[a-zA-Z_][a-zA-Z0-9_.]*' 
    
    def repl(match):
        variable = match[0]
        if variable in ut.reserved:
            return variable
        elif variable in rename:
            new_variable = rename[variable]
        elif variable in delete:
            msg = f'Encountered a name marked for deletion: {variable}'
            raise NameError(msg)
        else:
            new_variable = child_name + '.' + variable
        
        return new_variable
    
    result  = re.sub(pattern, repl, string.strip())
    
    return result