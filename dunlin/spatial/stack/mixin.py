import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


from .eventstack import EventStack 
from dunlin.datastructures import SpatialModelData

class Imager(EventStack):
    def __init__(self, spatial_data: SpatialModelData):
        
        super().__init__(spatial_data)
        
        
        
    def to_image(self,) -> np.ndarray:
        pass
