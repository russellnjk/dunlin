import numpy as np
import re

import dunlin.utils as ut

class MassTransfer:
    def __init__(self, spatial_data):
        self.compartments = spatial_data['compartments']
        
        
        # self.dmnt2states = 
