import sys
import os
from os.path  import dirname, join

#Add path
_dir = dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)
    
#New files
from .model            import read_file, make_models, Model
from ._utils_plot.plot import figure, gridspec, colors, scilimit, save_figs

styles = {}

_utils_plot_path = join(_dir, '_utils_plot')
for file in os.listdir(_utils_plot_path):
    try:
        name, ext = file.split('.')
        if ext == 'mplstyle':
            styles[name] = join(_utils_plot_path, file)
    except:
        pass