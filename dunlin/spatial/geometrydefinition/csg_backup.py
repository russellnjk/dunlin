import numpy         as np
import scipy.spatial as spl
import warnings
from numba             import njit
from numba.core.errors import NumbaPerformanceWarning
from typing            import Sequence, Union

from .csgbase import Fillable, Squarelike, Circular, CSGObject
from .sparse import make_sparse

###############################################################################
#Primitives
###############################################################################
class Square(Squarelike, Fillable):
    def __init__(self, _points=None) -> None:
        if _points is None:
            points = np.array([[-1, -1], 
                               [-1,  1], 
                               [ 1,  1], 
                               [ 1, -1],
                               ], dtype=np.float64)
        else:
            points = np.array(_points, dtype=np.float64)
            
            if points.shape != (4, 2):
                msg = 'Wrong shape for square. Expected array shape (4, 2).'
                msg = f'{msg} Received {points.shape}'
                raise ValueError(msg)
        
        super().__init__(points)
        
    @property
    def ndims(self) -> int:
        return 2
    
class Cube(Squarelike, Fillable):
    def __init__(self, _points=None) -> None:
        if _points is None:
            points = np.array([[-1, -1, -1],
                               [-1,  1, -1],
                               [ 1,  1, -1],
                               [ 1, -1, -1],
                               [-1, -1,  1],
                               [-1,  1,  1],
                               [ 1,  1,  1],
                               [ 1, -1,  1],
                               ], dtype=np.float64)
        else:
            points = np.array(_points, dtype=np.float64)
            
            if points.shape != (8, 3):
                msg = 'Wrong shape for cube. Expected array shape (8, 3).'
                msg = f'{msg} Received {points.shape}'
                raise ValueError(msg)
        
        super().__init__(points)
        
    @property
    def ndims(self) -> int:
        return 3
    
class Circle(Circular, Fillable):
    def __init__(self, _center=None, _radii=None, _orientation=None) -> None:
        radii       = [1, 1] if _radii       is None else _radii
        center      = [0, 0] if _center      is None else _center
        orientation = 0      if _orientation is None else _orientation
        
        super().__init__(center, radii, orientation, _ndims=2)

class Sphere(Circular, Fillable):
    def __init__(self, _center=None, _radii=None, _orientation=None) -> None:
        radii       = [1, 1, 1]    if _radii       is None else _radii
        center      = [0, 0, 0]    if _center      is None else _center
        orientation = [0, 0, 0, 0] if _orientation is None else _orientation
        
        super().__init__(center, radii, orientation, _ndims=3)

###############################################################################
#Composite
###############################################################################
class Composite(CSGObject, Fillable):
    _allowed = 'union', 'intersection', 'difference'
    
    def __init__(self, op, *shapes) -> None:
        
        #Check op and number of shapes
        if op not in self._allowed:
            msg = f'Expected one of {self._allowed}. Received {repr(op)}'
            raise ValueError(msg)
        
        if len(shapes) < 2:
            raise ValueError(f'Expected at least 2 shapes. Received {len(shapes)}.')
        elif op == 'difference' and len(shapes) != 2:
            raise ValueError(f'Expected 2 shapes. Received {len(shapes)}.')
        
        #Preprocess shapes and 
        shapes_ = []
        ndims   = set()
        for shape in shapes:
            if not isinstance(shape, CSGObject):
                msg = f'Expected a CSGObject. Received {type(shape).__name__}'
                raise TypeError(msg)
            
            shapes_.append(shape)
            ndims.add(shape.ndims)
            
        #Save operations and shapes
        self._op     = op
        self._shapes = tuple(shapes_)
        self.atol    = 1e-12
        
        #Check ndims and save as attribute
        if len(ndims) > 1:
            raise ValueError('Inconsistent dimensions.')
        self._ndims = tuple(ndims)[0]
    
    @property
    def ndims(self) -> int:
        return self._ndims
    
    def contains_points(self, points, _atol=None) -> np.ndarray:
        atol = self.atol if _atol is None else _atol
        shapes = self._shapes
        
        if self._op == 'union':
            is_inside = shapes[0].contains_points(points, atol)
            
            for shape in shapes[1:]:
                is_inside = is_inside | shape.contains_points(points, atol)
        
        elif self._op == 'intersection':
            is_inside = shapes[0].contains_points(points, atol)
            
            for shape in shapes[1:]:
                is_inside = is_inside & shape.contains_points(points, atol)
                
        else:
            in_a = shapes[0].contains_points(points, atol)
            in_b = shapes[1].contains_points(points, atol)
            
            is_inside = in_a & ~in_b
        
        return is_inside
    
    def transform(self, transformation: str, *args) -> None:
        for shape in self._shapes:
            shape(transformation, *args)
    
    def make_grid(self, step=0.1) -> np.ndarray:
        shapes = self._shapes 
        
        if self._op == 'difference':
            points = {tuple(point) for point in shapes[0].make_grid(step)}
            
        else:
            points = {tuple(point) for shape in shapes 
                      for point in shape.make_grid(step)
                     }
            
        points = np.array(list(points), dtype=np.float64)
        atol   = step/2
        _      = make_sparse(np.array([[0.1, 0.1], [0.12, 0.12]]), atol)#Warmup
        grid   = make_sparse(points, atol)
        grid   = np.array(grid, dtype=np.float64)
        
        return grid
        
###############################################################################
#Parse Definition
###############################################################################
primitives = {'square': Square, 'circle': Circle, 
              'cube': Cube, 'sphere': Sphere
              }    
operators = ['union', 'intersection', 'difference']

def parse_node(node):
    if type(node) == str:
        if node in primitives:
            new_shape = primitives[node]()
            return new_shape
        else:
            raise ValueError('No primitive {node}.')
            
    elif type(node) == list:
        if node[0] in operators:
            #Expect Composite(op, shapes)
            op, *shapes_ = node
            shapes       = [parse_node(s) for s in shapes_]
            new_shape    = Composite(op, *shapes)
            
            return new_shape
        
        else:
            #Expect piping i.e. [shape, *transformations]
            new_shape = parse_node(node[0])
            for transformation in node[1:]:
                if type(transformation) == dict:
                    new_shape = new_shape(**transformation)
                elif type(transformation) in [list, tuple]:
                    new_shape = new_shape(*transformation)
                else:
                    msg = 'Expected list, tuple or dict.'
                    msg = f'{msg} Received {type(transformation).__name__}'
                    raise TypeError(msg)
            return new_shape
        
    else:
        msg = f'Node must be a str, list. Received {type(node).__name__}.'
        raise TypeError(msg)
