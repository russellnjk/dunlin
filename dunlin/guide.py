import os
import shutil
from pathlib import Path

def retrieve(s='cases'):

    src = Path(__file__).parent.parent / s
    dst = os.getcwd()
    shutil.copytree(src, dst)
    