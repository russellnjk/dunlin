import sys
import os
from os.path  import dirname, join

#Add path
_dir = dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)
    
#Import front end
# from .standardfile     import *
from .model            import (read_file, 
                               make_models, 
                               Model
                               )
from .simulate         import (plot_simresults, 
                               simulate_model, 
                               simulate_and_plot_model, 
                               SimResult
                               )
from .optimize         import (fit_model,
                               plot_dataset,
                               simulate_and_plot,
                               OptResult
                               )
from ._utils_plot      import (figure, 
                               gridspec, 
                               colors, 
                               scilimit, 
                               save_figs,
                               colors,
                               make_color_scenarios,
                               make_dark_scenarios,
                               make_light_scenarios
                               )

styles = {}

_utils_plot_path = join(_dir, '_utils_plot')
for file in os.listdir(_utils_plot_path):
    try:
        name, ext = file.split('.')
        if ext == 'mplstyle':
            styles[name] = join(_utils_plot_path, file)
    except:
        pass