from dunlin.comp.flatten import flatten

def flatten_model(all_data        : dict, 
                  ref             : str, 
                  required_fields : dict[str, list[bool]]
                  ) -> dict:
    
    #Flatten the model
    flattened = flatten(all_data, required_fields, ref)
    
    flattened['ref'] = ref
    return flattened
