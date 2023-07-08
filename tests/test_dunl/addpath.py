import sys
from os       import getcwd, listdir
from os.path  import abspath, dirname, join

#Add path
_dir = dirname(dirname(dirname(__file__)))
_dir = join(_dir, 'dunlin', 'standardfile', 'dunl')
if _dir not in sys.path:
    sys.path.insert(0, _dir)
