from pathlib     import Path

from .dun_raw import DunRawCode

###############################################################################
#Master Parsing Algorithm
###############################################################################
raw_code_parsers = {'.dun': DunRawCode.read_file}

def read_file(*filenames):
    global raw_code_parsers
    
    if not filenames:
        return None
    
    data = []
    for filename in filenames:
        suffix = Path(filename).suffix
        
        #Get the raw code class
        rc    = raw_code_parsers[suffix](filename)
        data += rc.to_data()
    
    return data

###############################################################################
#Master Writing Algorithm
###############################################################################
def data2dun():
    pass

def data2sbml():
    pass

def data2yaml():
    pass