import re

import dunlin.standardfile.dunl.readstring    as rst
import dunlin.standardfile.dunl.readshorthand as rsh
import dunlin.standardfile.dunl.delim         as dm

def read_element(element, interpolators=None):
    interpolated = interpolate(element, interpolators)
    substituted  = rsh.read_shorthand(interpolated)
    result       = {} 
    
    for s in substituted:
        data   = rst.read_string(s)
        result = {**result, **data} 

    return result

###############################################################################
#Interpolation
###############################################################################    
def interpolate(element, interpolators):
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

        elif char in dm.quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)
            
    chunk = element[i0:]
    result += chunk
    
    return result