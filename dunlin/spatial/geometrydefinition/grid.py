import numpy             as np
from numbers import Number
from typing  import Iterable, Sequence

import dunlin.utils_plot     as upp

###############################################################################
#Classes
###############################################################################
class BaseGrid:
    #To be set by the subclasses
    _ndims: int
    _grid : np.ndarray
    _graph: dict[tuple, list]
    _spans: np.ndarray
    _step : Number
    
    #For plotting
    _scatter_args = dict(s=100)
    _line_args    = dict(color='black', marker='None', linewidth=1)
    
    def __init__(self, name: str) -> None:
        self.name = name
    
    @property
    def ndims(self) -> int:
        return self._ndims
    
    @property
    def graph(self) -> dict:
        return self._graph
    
    @property
    def points(self) -> np.ndarray: 
        return self._points
    
    def items(self) -> Iterable:
        return self._graph.items()
    
    def keys(self) -> Iterable:
        return self._graph.keys()
    
    def values(self) -> Iterable:
        return self._graph.values()
    
    def _make_point(self, point) -> tuple:
        point_ = tuple(point)
        ndims  = self._ndims
        
        if len(point_) != ndims:
            msg  = f'Expected a point with {ndims} coordinates. '
            msg += f'Received point {point}' 
            raise ValueError(msg)
        return point_
    
    def __iter__(self):
        return iter(self._graph)
    
    def __getitem__(self, point):
        point = self._make_point(point)
        
        return self._graph[point]
    
    def __str__(self):
        c = type(self).__name__
        s = ', '.join([str(list(i)) for i in self.spans])
        d = self.step
        r = f'{c}(step={d}, spans=({s}))'
        return r
    
    def __repr__(self):
        return str(self)
    
    
    def plot(self, ax, scatter_args=None, line_args=None):
        scatter_args = {} if scatter_args is None else scatter_args
        line_args    = {} if line_args    is None else line_args
        
        result = {'scatter': self.plot_points(ax, **scatter_args),
                  'edges'  : self.plot_edges( ax, **line_args   )
                  }
        
        return result
    
    def plot_edges(self, ax, **line_args):
        #Parse plot args
        sub_args     = {'name': self.name}
        converters   = {'color': upp.get_color}
        line_args    = upp.process_kwargs(line_args, 
                                          [self.name], 
                                          self._line_args, 
                                          sub_args, 
                                          converters
                                          )
    
        #Plot on the axes
        result = []
        for point, neighbours in self.items():
            temp = []
            
            for key, neighbour in neighbours.items():   
                if key < 0:
                    continue
                temp = zip(point, neighbour)
                temp = ax.plot(*temp, **line_args)
                result.append(temp)
           
        return result
    
    def plot_points(self, ax, **scatter_args):
        #Parse plot args
        sub_args     = {'name': self.name}
        converters   = {'c': upp.get_color}
        scatter_args = upp.process_kwargs(scatter_args, 
                                          [self.name], 
                                          self._scatter_args, 
                                          sub_args, 
                                          converters
                                          )
    
        #Plot on the axes
        points = self.points
        values = points.T
        result = ax.scatter(*values, **scatter_args)
        
        return result

class RegularGrid(BaseGrid):
    @property
    def spans(self) -> np.ndarray:
        return self._spans
    
    @property
    def step(self) -> Number:
        return self._step
    
    @property
    def _plot_args(self):
        return dict(name=self.name, step=self.step, spans=self.spans)
  
class BasicGrid(RegularGrid):
    @classmethod
    def make_neighbours(cls, point, step, spans):
        neighbours = {}
        point      = tuple(point)
        
        for axis in range(len(point)):
            plus  = point[:axis] + (point[axis] + step,) + point[axis+1:]
            minus = point[:axis] + (point[axis] - step,) + point[axis+1:]
            
            if cls._in_spans(plus, spans):
                neighbours[axis + 1] = plus
            if cls._in_spans(minus, spans):
                neighbours[-(axis + 1)] = minus
        
        return neighbours
    
    @staticmethod
    def _in_spans(point, spans):
        for coordinate, span in zip(point, spans):
            if coordinate < span[0] or coordinate > span[1]:
                return False
        return True
    
    @staticmethod
    def _check_span(span):
        if len(span) != 2:
            msg  = 'Expected a pair of values i.e. (lower_bound, upper_bound)'
            msg += f' Received {span}'
            
            raise ValueError(msg)
        elif not isinstance(span[0], Number) or not isinstance(span[1], Number):
            msg = f'Bounds must be numbers. Received {span}'
            raise ValueError(msg)
            
        elif span[0] >= span[1]:
            msg = f'Lower bound >= upper bound. Received {span}'
            raise ValueError(msg)
    
    
    
    def __init__(self, step, *spans, name=''):
        super().__init__(name)
        
        ndims = len(spans)
        axes  = []
        for span in spans:
            self._check_span(span)
            
            start = span[0]//step * step
            stop  = span[1]//step * step
            inter = int((stop-start)/step) + 1
            axes.append(np.linspace(start, stop, inter))
        
        grid = np.meshgrid(*axes)
        grid = np.stack([a.flatten() for a in grid], axis=1)
        
        graph  = {}
        # shifts = self.make_shifts(ndims, step)
        
        for point in grid:
            key        = tuple(point) 
            graph[key] = self.make_neighbours(point, step, spans)
        
        self._points  = grid
        self._graph = graph
        self._ndims = ndims
        self._spans = np.array(spans)
        self._step  = step
    
class NestedGrid(RegularGrid):
    @staticmethod
    def _check_steps(parent, child):
        if parent.step//child.step != parent.step/child.step:
            msg  = 'Parent step size is not a multiple of child step size.'
            msg += f'\nParent step size: {parent.step}'
            msg += f'\nChild step size: {child.step}'
            raise ValueError(msg)
    
    @staticmethod
    def _check_child_inside(parent, child) -> None:
        pmax = parent.spans[:, 1]
        pmin = parent.spans[:, 0]
        cmax = child.spans[:, 1]
        cmin = child.spans[:, 0]
        
        if any(pmin >= cmin) or any(pmax <= cmax):
            msg  = 'Child grid not fully contained by parent.'
            msg += f'\nParent spans: {parent}\nChild spans: {child}'
            raise ValueError(msg)
        
        pstep = parent.step
        if any(cmax//pstep != cmax/pstep) or any(cmin//pstep != cmin/pstep):
            msg  = 'Child corners do not fit parent grid.'
            msg += f'\nParent spans: {parent}\nChild spans: {child}'
            raise ValueError(msg)
        
    @staticmethod
    def _check_children_separate(child, others) -> None:
        for child_ in others:
            for axis, axis_ in zip(child.spans, child_.spans):
                if axis_[0] <= axis[0] <= axis_[1] or axis_[0] <= axis[1] <= axis_[1]:
                    msg  = 'Overlapping children.\n'
                    msg += f'{child}\n{child_}'
                    raise ValueError(msg)
    
    def __init__(self, parent, *children, name: str='') -> None:
        #Check type
        for obj in [parent, *children]:
            if not isinstance(obj, RegularGrid):
                msg  = 'Expected an instance of a RegularGrid.'
                msg += f' Received {type(obj).__name__}\n'
                msg += f'mro: {type(obj).__mro__}'
                raise TypeError(msg)
        
        #Proceed with instantiation
        super().__init__(name)
        
        ndims = parent.ndims
        graph = {k: dict(v) for k, v in parent.graph.items()}
        
        for i, child in enumerate(children):
            if child.ndims != ndims:
                msg = 'Inconsistent dimensions.'
                raise ValueError(msg)
            
            #Check steps are compatible
            self._check_steps(parent, child)
            #Check child is inside parent
            self._check_child_inside(parent, child)
            #Check that there is no overlap with other children
            self._check_children_separate(child, children[:i])
            
            for child_point, child_neighbours in child.items():
                if child_point in graph:
                    parent_neighbours = dict(parent.graph[child_point])
                    temp              = {}

                    for key in parent_neighbours:
                        if key in child_neighbours:
                            temp[key] = child_neighbours[key]
                        else:
                            temp[key] = parent_neighbours[key]
                    
                    graph[child_point] = temp
                       
                else:
                    graph[child_point] = dict(child_neighbours)
                
        #Make the new grid
        self._ndims = ndims
        self._spans = parent.spans
        self._graph = graph
        self._points  = np.array(list(graph))
        self._step  = parent.step 

###############################################################################
#Instantiation from Config Dicts
###############################################################################
def make_grids_from_config(grid_config: dict, use_cache=True) -> dict[str, NestedGrid]:
    basic_grids  = make_basic_grids(grid_config)
    nested_grids = {}
    cache        = {} if use_cache else None
    
    for name in basic_grids:
        grid = merge_basic_grids(basic_grids, 
                                 grid_config, 
                                 name,
                                 cache
                                 )
        
        nested_grids[name] = grid
    
    return nested_grids

def make_basic_grids(grid_config: dict) -> dict[str, BasicGrid]:
    basic_grids = {}
    
    for name, config in grid_config.items():
        args = config['config']
        
        try:
            if hasattr(args, 'items'):
                grid = BasicGrid(**args)
            else:
                grid = BasicGrid(*args)
        except Exception as e:
            s = f'Error in instantiating BasicGrid {name}.\n'
            a = e.args[0]
            n = s + a
            
            raise type(e)(n)
            
        basic_grids[name] = grid
    
    return basic_grids
    

def merge_basic_grids(basic_grids: dict[str, BasicGrid], 
                      grid_config: dict, 
                      parent_name: str, 
                      cache: dict=None, 
                      _hierarchy: Sequence=()
                      ) -> NestedGrid:
    #Set up cache and return if possible
    cache = {} if cache is None else cache
    if parent_name in cache:
        return cache[parent_name]
    
    #Prepare to extract child grids
    parent_grid    = basic_grids[parent_name]
    children_names = grid_config[parent_name].get('children', [])
    child_grids    = []
    
    #Iterate and extract child grids
    for child_name in children_names:
        #Check the hierarchy
        if child_name in _hierarchy:
            raise CircularHierarchy(*_hierarchy, child_name)
        
        #Make child grid and append
        if child_name in cache:
            child_grid = cache[child_name]
        else:
            child_grid = basic_grids[child_name]
            
            if grid_config[child_name].get('children'):
                child_grid = merge_basic_grids(basic_grids, 
                                               grid_config, 
                                               child_name,
                                               cache,
                                               _hierarchy + (child_name, )
                                               )
            
        child_grids.append(child_grid)
    
    #Instantiate and update cache
    nested_grid        = NestedGrid(parent_grid, *child_grids, name=parent_name)
    cache[parent_name] = nested_grid 
    
    return nested_grid

class CircularHierarchy(Exception):
    def __init__(self, *hierarchy):
        s   = ' -> '.join([str(i) for i in hierarchy])
        msg = f'Circular hierarchy: {s}'
        
        super().__init__(msg)