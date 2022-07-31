import dunlin.utils.paramcontrol as upc

class BaseGeometry():
    def __init__(self, name, ordinal, domain_type):
        self.name        = name
        self.ordinal     = ordinal
        self.domain_type = domain_type
        

class AnalyticGeometry(BaseGeometry):
    _function_type = ['layered']
    
    def __init__(self, name, ordinal, domain_type, math, function_type='layered'):
        super().__init__(name, ordinal, domain_type)
        pass
    
class SampledFieldGeometry(BaseGeometry):    
    def __init__(self, name, ordinal, domain_type, value, bounds=None ):
        super().__init__(name, ordinal, domain_type)
        
        if value is not None and bounds is not None:
            raise ValueError('value and bounds cannot be used together.')
            
        if value is not None:
            self.value  = value
            self.bounds = None
        else:
            self.value  = None
            if hasattr(bounds, 'items'):
                self.bounds = bounds['lb'], bounds['ub']
            else:
                lb, ub      = bounds
                self.bounds = lb, ub

class CSGeometry(BaseGeometry):
    def __init__(self, name, ordinal, domain_type, node):
        super().__init__(name, ordinal, domain_type)

class CSGPrimitive():
    @upc.accept_vals(shape=['sphere', 'cube', 'cylinder', 'cone', 'circle', 'square'])
    def __init__(self, shape):
        self.shape = shape
    
class CSGSetOperator():
    '''
    | for union.
    & for intersection.
    – for difference
    ^ for symmetric difference
    '''
    pass

'''
``geometry
coordinate_components : []
domain_types          : []
domains               : []
adjacent_domains      : []
sampled_fields        : []
definitions           : [
    hemisphere : [
        definition  : csgeometry,
        domain_type : nucleus,
        ordinal     : 1,
        composition : intersection[sphere, translation : [cube, 1, 0, 0] ]
        ]
    vesicle : [
        definition  : csgeometry,
        domain_type : vesicle,
        ordinal     : 1,
        composition : translation: [ 
            rotation : [
                scale : [sphere, 0.075833, 0.080451, 0.029583], 
                2.4391, 1.5373, 0.58404 
                ], 
            5.439, 5.88, 1.078
            ]
        ]
    sphere0 : [
        definition  : analytic,
        domain_type : vacuole,
        composition : 8*(x-1)**2+8*(y-1)**2+8*(z-1)**2 < 1
        ]
    ]

```definitions
definitions           : [
    hemisphere : [
        definition  : csgeometry,
        domain_type : nucleus,
        ordinal     : 1,
        composition : intersection[sphere, translation : [cube, 1, 0, 0] ]
        ]
    vesicle : [
        definition  : csgeometry,
        domain_type : vesicle,
        ordinal     : 1,
        composition : translation: [ 
            rotation : [
                scale : [sphere, 0.075833, 0.080451, 0.029583], 
                2.4391, 1.5373, 0.58404 
                ], 
            5.439, 5.88, 1.078
            ]
        ]
    sphere0 : [
        definition  : analytic,
        domain_type : vacuole,
        composition : 8*(x-1)**2+8*(y-1)**2+8*(z-1)**2 < 1
        ]
    ]
'''

'''

a : [b : c, d: [e: f, g: [h: i, j: k]], l: m]

in new .dun

`a
b : c

``d
e : f

```g
h : i
j : k

`a
l : m


'''