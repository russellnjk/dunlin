import sys
from os       import getcwd, listdir
from os.path  import abspath, dirname, join

#Add path
_dir = dirname(dirname(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
