def asn(lhs, *rhs, indent=0):
    if len(rhs) == 1:
        if type(lhs) == str:
            left = lhs
        else:
            left = ', '.join(lhs)
        return '\t'*indent + f'{left} = {rhs[0]}'
    elif len(rhs) == 2 or len(rhs) == 4:
        raise ValueError('')
    elif len(rhs) == 3:
        right = f'{rhs[0]} if {rhs[1]} else {rhs[2]}'
        return asn(lhs, right, indent=indent)
    else:
        right = f'{rhs[-3]} if {rhs[-2]} else {rhs[-1]}'
        rhs_  = rhs[:-3]
        return asn(lhs, *rhs_, right, indent=indent)

def def_func(name, *args):
    signature = ', '.join(args)
    return f'def {name}({signature}):'