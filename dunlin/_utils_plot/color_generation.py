import seaborn           as sns

###############################################################################
#Globals
###############################################################################
#Refer for details: https://xkcd.com/color/rgb/
colors = sns.colors.xkcd_rgb

###############################################################################
#Colors for Scenarios
###############################################################################
def make_color_scenarios(scenarios, base_colors=None, palette_type='pastel'):
    global colors
    base_colors = len(scenarios) if base_colors is None else base_colors
    base_colors = _read_base_colors(base_colors, palette_type)
    if len(scenarios) != len(base_colors):
        raise BaseColorError(f'Not enough base colors for base scenarios: {scenarios}. Need at least {len(scenarios)}')
    
    palette = dict(zip(scenarios, base_colors))
    
    return palette
    
def make_dark_scenarios(scenarios, base_colors, reverse=False, palette_type=None):
    base_colors = len(scenarios) if base_colors is None else base_colors
    return _scenarios(sns.dark_palette, scenarios, base_colors, reverse, palette_type)

def make_light_scenarios(scenarios, base_colors, reverse=False, palette_type=None):
    base_colors = len(scenarios) if base_colors is None else base_colors
    return _scenarios(sns.light_palette, scenarios, base_colors, reverse, palette_type)

def _scenarios(func, scenarios, base_colors, reverse=False, palette_type=None):
    global colors
    base_colors = _read_base_colors(base_colors, palette_type)
    bases       = {}
    grads       = {}
    n           = 0
    
    for c in scenarios:
        if type(c) == tuple:
            base = c[:len(c)-1]
            grad = c[-1]
        elif type(c) == str or isnum(c):
            base = ()
            grad = c
        else:
            raise TypeError(f'Unexpected type of scenario "{c}" of type {type(c)}')

        if base in bases:
            base_color = bases[base]
        else:
            if hasattr(base_colors, 'items'):
                try:
                    base_color = base_colors[base]
                except KeyError:
                    raise BaseColorError(f'No base color for base scenario: {base}.')
            else:
                base_color  = base_colors[n]
                n          += 1 
                if n > len(base_colors):
                    raise BaseColorError(f'Not enough base colors for base scenarios: {bases}. Need at least {len(bases)}')
            
            bases[base] = base_color
        
        grads.setdefault(base, []).append(grad)

    palette = {}
    for base, base_color in bases.items():
        base_color = base_color
        colors_    = func(base_color, len(grads[base])+2,reverse=reverse)
        colors_    = colors_[1:len(colors_)-1]
        
        for grad, color in zip(grads[base], colors_):
            key          = (*base, grad) if base else grad
            palette[key] = color
    
    return palette

def _read_base_colors(base_colors, palette_type='pastel'):
    if type(base_colors) == int:
        base_colors = sns.color_palette(palette_type, base_colors)
    elif hasattr(base_colors, '__iter__') and type(base_colors) != str:
        base_colors = [colors[color] if type(color) == str else color for color in base_colors]
    else:
        raise BaseColorTypeError(base_colors)
    
    return base_colors
                    
def isnum(x):
    try:
        float(x)
        return True
    except:
        return False
       
class BaseColorError(Exception):
    pass

class BaseColorTypeError(Exception):
    def __init__(self, t):
        msg = f'Base colors must be a positive integer or a list/tuple of colors. Received: {t}'
        super().__init__(msg)
