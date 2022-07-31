import numpy         as np
import scipy.spatial as spl
import warnings
from abc               import ABC, abstractmethod
from numba             import njit
from numba.core.errors import NumbaPerformanceWarning
from numbers           import Number
from typing            import Union

from .rotation import (rotate2D, rotate3D, add_rotations)

###############################################################################
#CSG Base Classes
###############################################################################
class CSGObject(ABC):
    
    def __call__(self, transformation, *args, **kwargs):
        if args and kwargs:
            msg = 'Detected a mix of positional and keyword arguments.'
            msg = f'{msg} Arguments must be all positional or all keywords.'
            msg = f'{msg}\nargs: {args}\nkwargs: {kwargs}'
            
            raise ValueError(msg)

        if kwargs:
            preprocess = self.parse_transformation_args
            if transformation in ['scale', 'translate']:
                if self.ndims == 2:
                    new_args = preprocess(['x' 'y'], kwargs)
                else:
                    new_args = preprocess(['x' 'y', 'z'], kwargs)
            elif transformation == 'rotate':
                if self.ndims == 2:
                    new_args = preprocess(['radians', 'x' 'y'], kwargs)
                else:
                    new_args = preprocess(['radians', 'x' 'y', 'z'], kwargs)
            else:
                raise ValueError(f'Unexpected transformation {transformation}.')
        else:

            if transformation in ['scale', 'translate']:

                if len(args) == self.ndims:
                    new_args = args
                else:
                    msg = f'Expected {self.ndims} arguments. Received {args}'
                    raise ValueError(msg)
                
            elif transformation == 'rotate':
                if len(args) == self.ndims + 1:
                    new_args = args
                elif self.ndims == 2 and len(args) == 1:
                    new_args = args
                else:
                    msg = f'Expected {self.ndims} arguments. Received {args}'
                    raise ValueError(msg)
            else:
                raise ValueError(f'Unexpected transformation {transformation}.')
                
        return self.transform(transformation, *new_args)
    
    @staticmethod
    def parse_transformation_args(expected_keys, dct):
        set_keys   = set(expected_keys)
        set_dct    = set(dct)
        missing    = set_keys.difference(set_dct)
        unexpected = set_dct.difference(set_keys)
        msg        = ' '
        if missing:
            msg += f'Missing keys: {missing}.'
        if unexpected:
            msg += f'Unexpected keys: {unexpected}'
        if msg:
            raise ValueError(msg)
        
        return {k: dct[k] for k in expected_keys}
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return f'{type(self).__name__}'
        
    ###########################################################################
    #Abstract Methods
    ###########################################################################
    @abstractmethod
    def contains_points(self):
        ...
    
    @abstractmethod
    def transform(self, transformation, *args):
        ...
    
    @property
    @abstractmethod
    def ndims(self) -> int:
        ...
    
class Squarelike(CSGObject):
    atol = 1e-12
    
    def __init__(self, corners):
        hull = spl.ConvexHull(corners)
        eqns = hull.equations.astype(np.float64)
        
        self._corners = corners
        self._hull    = hull
        self._offset  = eqns[:,-1].astype(np.float64)
        self._normal  = eqns[:,:-1].astype(np.float64)
        
    ###########################################################################
    #Safe Access
    ###########################################################################
    @property
    def corners(self) -> np.ndarray:
        return self._corners
    
    @property
    def normal(self) -> np.ndarray:
        return self._normal
    
    @property
    def offset(self):
        return self._offset
    
    @property
    def hull(self):
        return self._hull
    
    ###########################################################################
    #Spatial Methods
    ###########################################################################
    @staticmethod
    @njit
    def _contains_points(points, normal, offset, atol=1e-12) -> bool:
        result = np.array([np.all(np.dot(normal, point) + offset <= atol) for point in points])
        
        return result
    
    def contains_points(self, points: np.ndarray, _atol=None) -> bool:
        atol   = self.atol if _atol is None else _atol
        normal = self._normal
        offset = self._offset
        points = np.array(points, dtype=np.float64)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
            result = self._contains_points(points, normal, offset, atol)
            
        return result
    
    def transform(self, transformation, *args):
        points = self._corners
        
        if transformation == 'scale':
            new_points = points*args
        elif transformation == 'translate':
            new_points = points+args
        elif transformation == 'rotate':
            if self.ndims == 2:
                new_points = rotate2D(points, *args)
            else:
                new_points = rotate3D(points, *args)
        else:
            msg = 'transformation must be scale, translate or rotate.'
            msg = f'{msg} Received {transformation}'
            raise ValueError(msg)
        
        return type(self)(new_points)
    
    @staticmethod
    def _interpolate(corner0, corner1, points, step):
        vector     = corner1 - corner0
        new_points = []
        intervals  = int(np.sum(vector**2)**0.5/step)
        for point in points:
            for i in range(1, intervals):
                new_point = point + vector*i/intervals
                
                new_points.append(new_point)
        
        
        points.extend(new_points)
        return points
    
    def make_grid(self, step: float=0.1) -> np.ndarray:
        interpolate = self._interpolate
        
        corners = self._corners
        points  = [corners[0]]
        
        if self.ndims == 2:
            interpolate(corners[0], corners[1], points, step)
            interpolate(corners[1], corners[2], points, step)
        elif self.ndims == 3:
            interpolate(corners[0], corners[1], points, step)
            interpolate(corners[1], corners[2], points, step)
            interpolate(corners[0], corners[4], points, step)
        
        #Remove duplicates
        grid = {tuple(p) for p in points}
        grid = np.array(list(grid), np.float64)
        return grid
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        a = tuple(self._corners)
        s = super().__str__()
        s = f'{s}{a}'
        
        return s
    
class Circular(CSGObject):
    def __init__(self, center, radii, orientation, _ndims: int):
        
        #Validate radii
        if len(radii) != _ndims:
            msg = f'Exactly {_ndims} radii must be provided. Received {radii}'
            raise ValueError(msg)
        
        for r in radii:
            if r <= 0:
                msg = f'Radius must be positive. Received {radii}.'
                raise ValueError(msg)
        
        #Validate center
        if len(center) != len(radii):
            msg = f'Expected {len(radii)} coordinates. Received {center}.'
            raise ValueError(msg)
        
        #Validate orientation
        if len(radii) == 2:
            if isinstance(orientation, Number):
                while orientation < 0:
                    orientation += 2*np.pi
                while orientation > 2*np.pi:
                    orientation -= 2*np.pi
            else:
                msg = f'Expected orientation to be a number. Received {orientation}.'
                raise ValueError(msg)
        elif len(radii) == 3:
            if len(orientation) == 4:
                o = orientation[0]
                
                while o < 0:
                    o += 2*np.pi
                while o > 2*np.pi:
                    o -= 2*np.pi

                orientation = [o, *orientation[1:]] 
            else:
                msg = f'Expected orientation of length 4. Received {orientation}.'
                raise ValueError(msg)
        
        #Save the attributes
        self._center = np.array(center, dtype=np.float64)
        self._radii  = np.array(radii, dtype=np.float64)
        
        if len(radii) == 2:
            self._orientation = orientation
        else:
            self._orientation = np.array(orientation, dtype=np.float64)
    
    ###########################################################################
    #Safe Accessors
    ###########################################################################
    @property
    def center(self):
        return self._center
    
    @property
    def radii(self):
        return self._radii
    
    @property
    def orientation(self):
        return self._orientation
    
    @property
    def ndims(self):
        return len(self._radii)
    
    ###########################################################################
    #Spatial Methods
    ###########################################################################
    @staticmethod
    @njit
    def _contains_points(points, center, radii):
        result = np.array([np.sum( ((p-center)/radii)**2 ) <= 1 for p in points])
        return result
        
    def contains_points(self, points: np.ndarray, _atol=None) -> bool:
        center      = self._center
        radii       = self._radii
        orientation = self._orientation
        points      = np.array(points, dtype=np.float64)
        
        if self.ndims == 2:
            if orientation:
                radians = -orientation
                x, y    = center
                points_ = rotate2D(points, radians, x, y)
            else:
                points_ = points
        else:
            if any(orientation):
                radians = -orientation[0]
                x, y, z = orientation[1:]
                points_ = rotate3D(points, radians, x, y, z)
            else:
                points_ = points
        
        result      = self._contains_points(points_, center, radii)
        return result
    
    def transform(self, transformation, *args):
        center      = self._center
        radii       = self._radii
        orientation = self._orientation
        
        if transformation == 'scale':
            new_center      = center
            new_radii       = radii*args
            new_orientation = orientation
        elif transformation == 'translate':
            new_center      = center+args
            new_radii       = radii
            new_orientation = orientation
        elif transformation == 'rotate':
            if self.ndims == 2:
                pt0        = center
                pt1        = center + [np.cos(orientation), np.sin(orientation)]
                points     = np.array([pt1, pt0])
                new_points = rotate2D(points, *args)
                new_vector = new_points[1] - new_points[0]
    
                new_center      = rotate2D([center], *args)[0]
                new_radii       = radii
                new_orientation = np.arctan2(*new_vector)

            else:
                new_center      = center
                new_radii       = radii
                new_orientation = add_rotations(orientation, args)
                
        else:
            msg = 'transformation must be scale, translate or rotate.'
            msg = f'{msg} Received {transformation}'
            raise ValueError(msg)
        
        
        return type(self)(new_center, new_radii, new_orientation)
    
    def make_grid(self, step:float=0.1) -> np.ndarray:
        center      = self._center
        radii       = self._radii
        orientation = self._orientation
        
        #Get linspace args
        min_vals = center - radii
        max_vals = center + radii 
        
        intervals = [int(i) + 1 for i in (max_vals - min_vals)/step]
        
        itr  = zip(min_vals, max_vals, intervals)
        axes = [np.linspace(*args) for args in itr]
        grid = np.meshgrid(*axes, sparse=False)
        grid = np.stack([a.flatten() for a in grid], axis=1)
        
        #Rotate grid into the correct frame of reference
        if self.ndims == 2 and orientation:
            radians = orientation
            x, y    = center
            grid    = rotate2D(grid, radians, x, y)
            
        elif self.ndims == 3 and any(orientation):
            grid = rotate3D(grid, *orientation)
        
        #Remove duplicates
        
        grid = {tuple(p) for p in grid}
        grid = np.array(list(grid))
        return grid
    
    ###########################################################################
    #Representation
    ###########################################################################
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        orientation = self._orientation
        
        if isinstance(orientation, Number):
            o = orientation 
        else:
            o = tuple(orientation)
        
        d = tuple(self._radii)
        a = f'radii= {d}, orientation= {o}'
        s = super().__str__()
        s = f'{s}({a})'
        
        return s

###############################################################################
#Mixins 
###############################################################################
class Fillable(ABC):
    ###########################################################################
    #Point Filling
    ###########################################################################
    def fill(self, grid: np.ndarray=None, 
             exterior: bool=False, 
             step: float=0.1
             ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        
        grid = self.make_grid(step) if grid is None else grid
        
        is_inside = self.contains_points(grid)
        interior  = grid[is_inside]
        
        if exterior:
            exterior = grid[~is_inside]
            
            return interior, exterior
        else:
            return interior
    
    
    ###########################################################################
    #Plotting
    ###########################################################################
    def scatter_2D(self, ax, grid=None, step=0.1, interior=True, exterior=False, 
                  interior_args=None, exterior_args=None
                  ) -> list:
        if self.ndims != 2:
            raise ValueError('Shape is not 2-D.')
        
        interior_, exterior_ = self.fill(grid, exterior=True, step=step)
        
        interior_args = {} if interior_args is None else interior_args
        exterior_args = {} if exterior_args is None else exterior_args
        
        result = []
        
        if interior:
            temp = ax.scatter(interior_[:,0], interior_[:,1], **interior_args)
            result.append(temp)
        if exterior:
            temp = ax.scatter(exterior_[:,0], exterior_[:,1], **exterior_args)
            result.append(temp)
            
        return result
    
    def scatter_3D(self, ax, grid=None, step=0.1, interior=True, exterior=False, 
                  interior_args=None, exterior_args=None
                  ) -> list:
        if self.ndims != 3:
            raise ValueError('Shape is not 3-D.')
        
        interior_, exterior_ = self.fill(grid, exterior=True, step=step)
        
        interior_args = {} if interior_args is None else interior_args
        exterior_args = {} if exterior_args is None else exterior_args
        
        result = []
        
        if interior:
            temp = ax.scatter(interior_[:,0], interior_[:,1], interior_[:,2], **interior_args)
            result.append(temp)
        if exterior:
            temp = ax.scatter(exterior_[:,0], exterior_[:,1], exterior_[:,2], **exterior_args)
            result.append(temp)
            
        return result
    
    ###########################################################################
    #Abstract Methods
    ###########################################################################
    @property
    @abstractmethod
    def make_grid(self):
        ...
        