import numpy         as np
from numba             import njit
# from numba.core.errors import NumbaPerformanceWarning
# from typing            import Sequence, Union

from .csgbase import PointCloud, Circular, Squarelike, CSGObject, Primitive

###############################################################################
#Primitives
###############################################################################
class Square(Squarelike, PointCloud):
    def __init__(self, _center=None, _axes=None, _orientation=None):
        center      = [0, 0] if _center      is None else _center
        axes        = [1, 1] if _axes        is None else _axes
        orientation = 0      if _orientation is None else _orientation
        
        super().__init__(center, axes, orientation, 2)

class Cube(Squarelike, PointCloud):
    def __init__(self, _center=None, _axes=None, _orientation=None):
        center      = [0, 0, 0]    if _center      is None else _center
        axes        = [1, 1, 1]    if _axes        is None else _axes
        orientation = [0, 0, 0, 0] if _orientation is None else _orientation
        
        super().__init__(center, axes, orientation, 3)

class Circle(Circular, PointCloud):
    def __init__(self, _center=None, _radii=None, _orientation=None) -> None:
        radii       = [1, 1] if _radii       is None else _radii
        center      = [0, 0] if _center      is None else _center
        orientation = 0      if _orientation is None else _orientation
        
        super().__init__(center, radii, orientation, _ndims=2)

class Sphere(Circular, PointCloud):
    def __init__(self, _center=None, _radii=None, _orientation=None) -> None:
        radii       = [1, 1, 1]    if _radii       is None else _radii
        center      = [0, 0, 0]    if _center      is None else _center
        orientation = [0, 0, 0, 0] if _orientation is None else _orientation
        
        super().__init__(center, radii, orientation, _ndims=3)

class Cylinder(Primitive, PointCloud):
    def __init__(self, _center=None, _radii=None, _orientation=None) -> None:
        radii       = [1, 1, 1]    if _radii       is None else _radii
        center      = [0, 0, 0]    if _center      is None else _center
        orientation = [0, 0, 0, 0] if _orientation is None else _orientation
    
        super().__init__(center, radii, orientation, _ndims=3)
    
    @staticmethod
    @njit
    def _contains_points(points, center, axes, rtol):
        upper   = center + axes*(1+rtol)
        lower   = center - axes*(1+rtol)
        radii   = axes[:2]*(1+rtol)
        center_ = center[:2]
        result  = np.zeros(len(points), dtype=np.bool_)
        
        for i, p in enumerate(points):
            if np.any(p > upper):
                continue
            elif np.any(p < lower):
                continue
            
            p_        = p[:2]
            result[i] = np.sum( ((p_-center_)/radii)**2 ) <= 1
            
        return result
    
###############################################################################
#Composite
###############################################################################
class Composite(CSGObject, PointCloud):
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
        
        #Check ndims and save as attribute
        if len(ndims) > 1:
            raise ValueError('Inconsistent dimensions.')
            
        #Save operations and shapes
        self._op     = op
        self._shapes = tuple(shapes_)
        self._ndims  = tuple(ndims)[0]
        
        #For lazy evaluation
        self._center  = None
        self._centers = None
        self._vectors = None
    
    @property
    def ndims(self) -> int:
        return self._ndims
    
    @property
    def shapes(self):
        return self._shapes
    
    @property
    def op(self):
        return self._op
    
    @property
    def centers(self):
        if self._centers is None:
            shapes   = self._shapes
            _centers = np.array([shape.center for shape in shapes])
            
            self._centers = _centers
        
        return self._centers
    
    @property
    def center(self):
        if self._center is None:
            centers = self.centers
            _center = np.mean(centers, axis=0)
            
            self._center  = _center
            
        return self._center
    
    @property
    def vectors(self):
        if self._vectors is None:
            center  = self.center 
            centers = self.centers
            vectors = centers-center
            
            self._vectors = vectors
        
        return self._vectors
    
    def __iter__(self):
        return iter(self._shapes)
    
    def __getitem__(self, index):
        return self._shapes[index]
    
    ###########################################################################
    #Spatial Methods
    ###########################################################################
    def contains_points(self, points, _rtol=None) -> np.ndarray:
        shapes = self._shapes
        rtol   = self.rtol if _rtol is None else _rtol
        
        if self._op == 'union':
            is_inside = shapes[0].contains_points(points, rtol)
            
            for shape in shapes[1:]:
                is_inside = is_inside | shape.contains_points(points, rtol)
        
        elif self._op == 'intersection':
            is_inside = shapes[0].contains_points(points, rtol)
            
            for shape in shapes[1:]:
                is_inside = is_inside & shape.contains_points(points, rtol)
                
        else:
            in_a = shapes[0].contains_points(points, rtol)
            in_b = shapes[1].contains_points(points, rtol)
            
            is_inside = in_a & ~in_b
        
        return is_inside
    
    def transform(self, transformation: str, *args) -> None:
        op     = self._op
        shapes = self._shapes
        
        if transformation in ['translate', 'rotate']:
            new_shapes    = [shape(transformation, *args) for shape in shapes]
            new_composite = type(self)(op, *new_shapes)
                
        else:
            vectors       = self.vectors
            new_vectors   = vectors*args
            displacements = new_vectors - vectors
            new_shapes    = []
            
            for displacement, shape in zip(displacements, shapes):
                new_shape = shape('scale', *args)
                new_shape = new_shape('translate', *displacement)
                
                new_shapes.append(new_shape)
            
            new_composite = type(self)(op, *new_shapes)
        
        return new_composite
    
    def make_grid(self, step=0.1) -> np.ndarray:
        shapes = self._shapes 
        points = set()
        
        if self._op == 'difference':
            points = {tuple(point) for point in shapes[0].make_grid(step)}
        else:
            points = {tuple(point) for shape in shapes 
                      for point in shape.make_grid(step)
                     }
        
        points = np.array(list(points), dtype=np.float64)
        return points
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        op = self._op
        if op =='intersection':
            j = ' & '
        elif op == 'union':
            j = ' | '
        else:
            j = ' - '
            
        a = j.join( [type(s).__name__ for s in self] )
        s = f'{type(self).__name__}'
        s = f'{s}({a})'
        
        return s
    
        
