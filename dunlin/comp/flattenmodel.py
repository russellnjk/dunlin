from dunlin.comp.flatten import flatten
from dunlin.comp.ode import required_fields as ode_required_fields


def flatten_ode(all_data, ref):
    #Flatten the model
    flattened = flatten(all_data, ode_required_fields, ref)
    
    flattened['ref'] = ref
    return flattened
