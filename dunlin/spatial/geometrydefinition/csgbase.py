import numpy         as np
import scipy.spatial as spl
import warnings
from abc               import ABC, abstractmethod, abstractproperty
from numba             import njit
from numba.core.errors import NumbaPerformanceWarning
from numbers           import Number
from typing            import Union

from .rotation import (rotate2D, rotate3D, add_rotations)

###############################################################################
#CSG Base Classes
###############################################################################
class CSGObject(ABC):
    rtol = 1e-6
    
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
    #Abstract Methods
    ###########################################################################
    @abstractproperty
    def ndims(self):
        ...
    
class Primitive(CSGObject):
    def __init__(self, center, axes, orientation, _ndims):
        #Validate axes
        if len(axes) != _ndims:
            msg = f'Exactly {_ndims} axes must be provided. Received {axes}'
            raise ValueError(msg)
        
        for r in axes:
            if r <= 0:
                msg = f'Lengths must be positive. Received {axes}.'
                raise ValueError(msg)
        
        #Validate center
        if len(center) != len(axes):
            msg = f'Expected {len(axes)} coordinates. Received {center}.'
            raise ValueError(msg)
        
        #Validate orientation
        if len(axes) == 2:
            if isinstance(orientation, Number):
                while orientation < 0:
                    orientation += 2*np.pi
                while orientation >= 2*np.pi:
                    orientation -= 2*np.pi
            else:
                msg = f'Expected orientation to be a number. Received {orientation}.'
                raise ValueError(msg)
        elif len(axes) == 3:
            if len(orientation) == 4:
                o = orientation[0]
                
                while o < 0:
                    o += 2*np.pi
                while o > 2*np.pi:
                    o -= 2*np.pi

                orientation = np.array([o, *orientation[1:]], dtype=np.float64) 
            else:
                msg = f'Expected orientation of length 4. Received {orientation}.'
                raise ValueError(msg)
        
        #Save the attributes
        self._center      = np.array(center, dtype=np.float64)
        self._axes        = np.array(axes, dtype=np.float64)
        self._orientation = orientation
        self._ndims       = _ndims

    @property
    def center(self):
        return self._center
    
    @property
    def axes(self):
        return self._axes
    
    @property
    def orientation(self):
        return self._orientation
    
    @property
    def ndims(self) -> int:
        return self._ndims
    
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
        
        d = tuple(self._axes)
        a = f'axes= {d}, orientation= {o}'
        s = f'{type(self).__name__}'
        s = f'{s}({a})'
        
        return s
        
    def contains_points(self, points, _rtol=None):
        center      = self._center
        axes        = self._axes
        orientation = self._orientation
        ndims       = self._ndims
        points      = np.array(points, dtype=np.float64)
        rtol        = self.rtol if _rtol is None else _rtol
        
        if len(points.shape) != 2:
            msg = 'points argument must be a 2-D array.'
            msg = f'{msg} Received {points.shape}'
            raise ValueError(msg)
        elif points.shape[1] != self.ndims:
            msg = f'Expected {self.ndims} columns i.e. shape of (n, {self.ndims}). '
            msg = f'{msg} Received {points.shape}.'
            raise ValueError(msg)
        
        #Rotate in object's frame of reference
        if ndims == 2 and orientation:
            x, y    = center
            radians = -orientation
            points_ = rotate2D(points, radians, x, y)
        elif ndims == 3 and any(orientation):
            radians = -orientation[0]
            x, y, z = orientation[1:]
            points_ = rotate3D(points, radians, x, y, z)
        else:
            points_ = points
        
        return self._contains_points(points_, center, axes, rtol)
    
    def transform(self, transformation, *args):
        center      = self._center
        axes        = self._axes
        orientation = self._orientation
        
        if transformation == 'scale':
            new_center      = center
            new_axes        = axes*args
            new_orientation = orientation
        elif transformation == 'translate':
            new_center      = center+args
            new_axes        = axes
            new_orientation = orientation
        elif transformation == 'rotate':
            if self.ndims == 2:
                pt0        = center
                pt1        = center + [np.cos(orientation), np.sin(orientation)]
                points     = [pt0, pt1]
                new_points = rotate2D(points, *args)
                new_vector = new_points[1] - new_points[0]
                new_center      = rotate2D([center], *args)[0]
                new_axes        = axes
                new_orientation = np.arctan2(new_vector[1], new_vector[0])

            else:
                new_center      = center
                new_axes        = axes
                new_orientation = add_rotations(orientation, args)
                
        else:
            msg = 'transformation must be scale, translate or rotate.'
            msg = f'{msg} Received {transformation}'
            raise ValueError(msg)
        
        return type(self)(new_center, new_axes, new_orientation)
    
    def make_grid(self, step:float=0.1) -> np.ndarray:
        center     = self._center
        axes       = self._axes
        orientation = self._orientation
        
        #Get linspace args
        min_vals = center - axes
        max_vals = center + axes 
        
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
    #Abstract Methods
    ###########################################################################
    @staticmethod
    @abstractmethod
    def _contains_points(points, center, radii, rtol):
        ...

class Circular(Primitive):
    @staticmethod
    @njit
    def _contains_points(points, center, radii, rtol):
        radii  = radii*(1+rtol)
        result = np.array([np.sum( ((p-center)/radii)**2 ) <= 1 for p in points])
        return result

class Squarelike(Primitive):
    @staticmethod
    @njit
    def _contains_points(points, center, axes, rtol):
        upper  = center + axes*(1+rtol)
        lower  = center - axes*(1+rtol)
        result = np.array([np.all( (p <= upper) & (p >= lower) ) for p in points])
        
        return result
    
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
    @abstractmethod
    def make_grid(self, step: float=0.1):
        ...
        
    