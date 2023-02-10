from typing import Sequence

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
        
        #Get child data. Recurse if required
        child_data = flatten(all_data, required_fields, child_ref, hierarchy)
        
        #Delete and replace
        submodel_data = delete_rename(child_name, child_data, rename, delete, required_fields) 

        for key in submodel_data:
            flattened.setdefault(key, {}).update(submodel_data[key])
    
    #Merge and overwrite
    for key, value in parent_data.items():
        if key == 'submodels':
            continue
        elif key in flattened:
            flattened[key].update(value)
        else:
            flattened[key] = value
    
    return flattened
        
def delete_rename(child_name      : str, 
                  child_data      : dict, 
                  rename          : callable, 
                  delete          : Sequence, 
                  required_fields : dict[str, callable]
                  ) -> dict:
    #Check that delete and rename are mutually exclusive
    overlap = set(delete).intersection(rename)
    if overlap:
        msg = f'Overlap between delete and rename: {overlap} '
        raise ValueError(msg)
    
    #Iterate through fields
    #Delete and rename
    submodel_data = {}
    args          = child_name, rename, delete
    for field, rename_func in required_fields.items():
        if field not in child_data:
            continue
        
        submodel_data[field] = {}
        for key, value in child_data[field].items():
            #Delete
            if key in delete:
                continue
            #Rename
            else:
                try:
                    new_key, new_value = rename_func(key, value, *args)
                except Exception as e:
                    msg = f'Error in renaming data in {field} {key}.'
                    msg = f'{msg}\n{e.args[0]}'
                    raise type(e)(msg)
                    
                submodel_data[field][new_key] = new_value
    return submodel_data

class MissingModel(Exception):
    pass

class CircularHierarchy(Exception):
    def __init__(self, *model_refs):
        joined = ' -> '.join(model_refs)
        super().__init__(joined)
        