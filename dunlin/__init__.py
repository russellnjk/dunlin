import sys
import os
from os.path  import dirname, join

#Add path
_dir = dirname(__file__)
sys.path.insert(0, _dir)

from .model_handler    import *
from ._utils_plot.axes import colors, palette_types

styles = {}

_utils_plot_path = join(_dir, '_utils_plot')
for file in os.listdir(_utils_plot_path):
    try:
        name, ext = file.split('.')
        if ext == 'mplstyle':
            styles[name] = join(_utils_plot_path, file)
    except:
        pass