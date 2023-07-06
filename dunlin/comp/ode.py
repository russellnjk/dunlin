import dunlin.utils as ut
from dunlin.comp.child   import make_child_item
    
def rename_x(x_name, x_data, child_name, rename, delete):
    new_name = make_child_item(x_name, child_name, rename, delete)
    return new_name, x_data

def rename_p(p_name, p_data, child_name, rename, delete):
    new_name = make_child_item(p_name, child_name, rename, delete)
    return new_name, p_data

def rename_func(func_name, func_data, child_name, rename, delete):
    new_name = make_child_item(func_name, child_name, rename, delete)
    f        = lambda v: make_child_item(v, child_name, rename, delete)
    new_data = [f(v) for v in func_data]
    return new_name, new_data

def rename_vrb(vrb_name, vrb_data, child_name, rename, delete):
    new_name = make_child_item(vrb_name, child_name, rename, delete)
    new_data = make_child_item(vrb_data, child_name, rename, delete)
    return new_name, new_data

def rename_rt(rt_name, rt_data, child_name, rename, delete):
    new_name = make_child_item(rt_name, child_name, rename, delete)
    new_data = make_child_item(rt_data, child_name, rename, delete)
    return new_name, new_data

def rename_rxn(rxn_name, rxn_data, child_name, rename, delete):
    new_name = make_child_item(rxn_name, child_name, rename, delete)
    
    f = lambda v: make_child_item(v, child_name, rename, delete)
    if ut.isdictlike(rxn_data):
        new_data = {k: f(v) for k, v in rxn_data.items()}
    elif ut.islistlike(rxn_data):
        new_data = [f(v) for v in rxn_data]
    
    return new_name, new_data

def rename_ev(ev_name, ev_data, child_name, rename, delete):
    new_name = make_child_item(ev_name, child_name, rename, delete)
    
    f = lambda v: make_child_item(v, child_name, rename, delete)
    g = lambda v: [f(i) for i in v] if ut.islistlike(v) else f(v)
    if ut.isdictlike(ev_data):
        new_data = {k: g(v) for k, v in ev_data.items()}
    elif ut.islistlike(ev_data):
        new_data = [g(v) for v in ev_data]
    
    return new_name, new_data
    
required_fields = {'states'     : rename_x,
                   'parameters' : rename_p,
                   'functions'  : rename_func,
                   'variables'  : rename_vrb,
                   'rates'      : rename_rt,
                   'reactions'  : rename_rxn,
                   'events'     : rename_ev,
                   }

