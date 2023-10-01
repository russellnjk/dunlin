import re

import dunlin.standardfile.dunl.readstring    as rst
import dunlin.standardfile.dunl.readshorthand as rsh

def read_element(element, interpolators=None):
    try:
        interpolated = interpolate(element, interpolators)
        strings      = rsh.read_shorthand(interpolated)
        result       = {} 
        
        for string in strings:
            data   = rst.read_string(string)
            result = {**result, **data} 
            
    except Exception as e:
        arg   = e.args[0]
        arg   = f'Error trying to parse element:\n{element}\n{arg}'
        error = type(e)(arg)
        
        raise error
        
    return result

###############################################################################
#Interpolation
############################################################################### 
quotes = '\'"'
   
def interpolate(element, interpolators):
    global quotes
    result          = ''
    i0              = 0
    quote           = []
    in_interpolator = False
    
    for i, char in enumerate(element):
        if char == '`' and not quote:
            
            chunk = element[i0:i]
            
            if in_interpolator:
                if chunk in interpolators:
                    chunk = interpolators[chunk]
                else:
                    raise SyntaxError(f'Undefined interpolator {chunk}')
                
            in_interpolator  = not in_interpolator
            
            result          += chunk

            i0 = i + 1

        elif char in quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)
            
    chunk = element[i0:]
    result += chunk
    
    return result
