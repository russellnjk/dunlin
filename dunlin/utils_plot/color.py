import seaborn  as sns
from numbers import Number
from typing  import Any

###############################################################################
#Globals
###############################################################################
#Refer for details: https://xkcd.com/color/rgb/
xkcd_colors = sns.colors.xkcd_rgb

def get_color(color: str|list[Number]) -> str|list[Number]:
    return sns.colors.xkcd_rgb.get(color, color)

def get_colors(colors: list[str|list[Number]]) -> list[str|list[Number]]:
    return [get_color(color) for color in colors]

# def get_color(color        : str|list[Number]|list[list[Number]],
#               allow_nested : bool = True
#               ) -> str|list[Number]|list[list[Number]]:
    
#     #For xkcd colors and hexcodes
#     if check_type(color):
#         if type(color) == str:
#             return sns.colors.xkcd_rgb.get(color, color)
#         else:
#             return list(color)
#     elif type(color)in {list, tuple}:
#         if all(check_type(x) for x in color):
#             return [get_color(x) for x in color]
    
#     #If the function has not returned, raise an exception
#     msg  = 'Invalid color format. The allowed formats are: '
#     msg += 'a string, a list of numbers or a list of the first two formats. '
#     msg += f'Received {color}.'
#     raise ValueError(msg)
        
# def check_type(x: Any) -> bool:
#     '''Checks if x is a string or list of numbers. Does not check if the values 
#     are valid.
#     '''
#     if type(x) == str:
#         return True
#     elif all(isinstance(i, Number) for i in x):
#         return True
#     else:
#         return False