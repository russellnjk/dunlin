#Import front end functions
from .standardfile     import *
from .ode              import *
from .load             import load_file
from .optimize         import *
from .utils_plot       import (xkcd_colors,
                               get_color,
                               get_colors
                               )

#Import datastructures for advanced use
from . import datastructures