import numpy             as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numbers            import Number
from typing             import Iterable, Sequence

import dunlin.utils_plot     as upp
    
###############################################################################
#Classes
###############################################################################
class BaseGrid:
    #For plotting
    _voxel_centers = dict(s=100, marker='s', c='steel')
    _voxel_links   = dict(color='grey', marker='None', linewidth=1)
    _vertices      = dict(s=10, c='cobalt')
    _edges         = dict(color='black', marker='None', linewidth=1)
    
    def __init__(self, 
                 name, 
                 step, 
                 spans, 
                 ndims, 
                 voxels,  
                 sizes,
                 vertices
                 ) -> None:
        self.name     = name
        self.step     = step
        self.spans    = spans
        self.ndims    = ndims
        self.voxels   = voxels
        self.sizes    = sizes
        self.vertices = vertices
    
    def contains(self, point) -> bool:
        spans = self.spans
        lb    = spans[:, 0]
        ub    = spans[:, 1]
        point = np.array(point)
        
        result = np.all((lb <= point) & (ub >= point))
        
        return result
    
    def __str__(self) -> str:
        c = type(self).__name__
        s = ', '.join([str(list(i)) for i in self.spans])
        d = self.step
        r = f'{c}(step={d}, spans=({s}))'
        return r
    
    def __repr__(self) -> str:
        return str(self)
    
    
    def plot(self, 
             ax, 
             voxel_center_args=None, 
             voxel_link_args=None,
             vertex_args=None,
             edge_args=None,
             to_plot = ('voxel_centers', 'voxel_links', 'vertices', 'edges')
             ):
        
        voxel_center_args = {} if voxel_center_args is None else voxel_center_args
        voxel_link_args   = {} if voxel_link_args   is None else voxel_link_args
        vertex_args       = {} if vertex_args       is None else vertex_args
        edge_args         = {} if edge_args         is None else edge_args
        
        result = {}
        if 'voxel_centers' in to_plot:
            result['voxel_centers'] = self.plot_voxel_centers(ax, **voxel_center_args)
        
        if 'voxel_links' in to_plot:
            result['voxel_links'] = self.plot_voxel_links(ax, **voxel_link_args)
            
        if 'vertices' in to_plot:
            result['vertices'] = self.plot_vertices(ax, **vertex_args)
        
        if 'edges' in to_plot:
            result['edges'] = self.plot_edges(ax, **edge_args)
        
        return result
        
    def plot_voxel_centers(self, ax, **scatter_args):
        #Parse plot args
        sub_args     = {'name': self.name}
        converters   = {'c': upp.get_color}
        scatter_args = upp.process_kwargs(scatter_args, 
                                          [self.name], 
                                          self._voxel_centers, 
                                          sub_args, 
                                          converters
                                          )
        
        #Plot on the axes
        result    = []
        base_size = scatter_args['s']
        for point in self.voxels:
            rel_size          = self.sizes[point]/self.step
            scatter_args['s'] = base_size*rel_size
            values            = [[i] for i in point]
            
            temp = ax.scatter(*values, **scatter_args)
            result.append(temp)
            
        return result
        
        
    def plot_voxel_links(self, ax, **line_args):
        #Parse plot args
        sub_args     = {'name': self.name}
        converters   = {'color': upp.get_color}
        line_args    = upp.process_kwargs(line_args, 
                                          [self.name], 
                                          self._voxel_links, 
                                          sub_args, 
                                          converters
                                          )
    
        #Plot on the axes
        result = []
        seen   = set()
        for point, neighbours in self.voxels.items():
            temp = []
            
            for shift, shift_neighbours in neighbours.items():   
                for neighbour in shift_neighbours:
                    pair = frozenset([point, neighbour])
                    
                    if pair in seen:
                        continue
                    seen.add(pair)
                    temp = zip(point, neighbour)
                    temp = ax.plot(*temp, **line_args)
                    result.append(temp)
           
        return result
        
    def plot_edges(self, ax, **line_args):
        #Parse plot args
        sub_args     = {'name': self.name}
        converters   = {'color': upp.get_color}
        line_args    = upp.process_kwargs(line_args, 
                                          [self.name], 
                                          self._edges, 
                                          sub_args, 
                                          converters
                                          )
    
        #Plot on the axes
        result = []
        seen   = set()
        for point, neighbours in self.vertices.items():
            temp = []
            
            for shift, neighbour in neighbours.items():                  
                pair = frozenset([point, neighbour])
                if pair in seen:
                    continue
                
                seen.add(pair)
                temp = zip(point, neighbour)
                temp = ax.plot(*temp, **line_args)
                result.append(temp)
           
        return result
    
    def plot_vertices(self, ax, **scatter_args):
        #Parse plot args
        sub_args     = {'name': self.name}
        converters   = {'c': upp.get_color}
        scatter_args = upp.process_kwargs(scatter_args, 
                                          [self.name], 
                                          self._vertices, 
                                          sub_args, 
                                          converters
                                          )
        
        #Plot on the axes
        points = np.array(list(self.vertices))
        values = points.T
        result = ax.scatter(*values, **scatter_args)
        
        return result


class RegularGrid(BaseGrid):
    @staticmethod
    def _check_span_format(span):
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
    
    
    def __init__(self, step: float, *spans, name: str=''):
        ndims   = len(spans)
        axes    = []
        shifts  = {}
        vshifts = {}
        
        for i, span in enumerate(spans, start=1):
            #Generate the arguments for numpy's meshgrid
            self._check_span_format(span)
            
            start = int(span[0]/step) * step
            stop  = int(span[1]/step) * step
            inter = int((stop-start)/step)*2 + 1
            axis  = np.linspace(start, stop, inter)
            axes.append(axis)
            
            #Get the coordinates of the neighbours
            plus  = dict(zip(axis[1::2], axis[3::2]))
            minus = dict(zip(axis[3::2], axis[1::2]))
            
            shifts[ i] = plus
            shifts[-i] = minus
            
            plus  = dict(zip(axis[0::2], axis[2::2]))
            minus = dict(zip(axis[2::2], axis[0::2]))
            
            vshifts[ i] = plus
            vshifts[-i] = minus
            
        #Generate the grid with meshgrid
        grid          = np.meshgrid(*axes)
        voxels        = {}
        slices        = (slice(1, None, 2), )*ndims
        voxel_centers = np.stack([a[slices].flatten().astype(np.float64) for a in grid], axis=1)
        sizes         = {}
        
        for point in voxel_centers:
            #Use tuples instead of numpy arrays for hashing
            key        = tuple(point) 
            neighbours = {}
            
            for shift, adjacent in shifts.items():
                idx        = abs(shift) - 1
                value      = key[idx]
                next_value = adjacent.get(value)
                if next_value is None:
                    continue
                
                neighbour = key[:idx] + (next_value, ) + key[idx+1:]

                neighbours[shift] = [neighbour]         
                    
            #Update voxels
            voxels[key] = neighbours
            sizes[key]  = step
        
        #Determine vertices
        slices   = (slice(0, None, 2), )*ndims
        points   = np.stack([a[slices].flatten() for a in grid], axis=1) 
        vertices = {}
        
        for point in points:
            key        = tuple(point) 
            neighbours = {}
            
            for shift, adjacent in vshifts.items():
                idx        = abs(shift) - 1
                value      = key[idx]
                if value in adjacent:
                    next_value = adjacent[value]
                    neighbour  = key[:idx] + (next_value, ) + key[idx+1:]
                
                    neighbours[shift] = neighbour
                
            vertices[key] = neighbours
        
        super().__init__(name, step, np.array(spans), ndims, voxels, sizes, vertices)
    
    def voxelize(self, point) -> tuple:
        '''Map a point to a voxel.
        '''
        step = self.step
        temp = [np.floor(i/step)*step + 0.5*step for i in point]
        
        return tuple(temp)
    
    def shift_point(self, point, shift) -> tuple:
        step = self.step
        idx  = abs(shift) - 1
        new  = point[idx] + step if shift > 0 else point[idx] - step
        temp = tuple(point[:idx]) + (new,) + tuple(point[idx+1:])
        
        return temp
    
class NestedGrid(BaseGrid):
    @staticmethod
    def _check_steps(parent, child):
        if parent.step % child.step:
            msg  = 'Parent step size must be a multiple of child step size.'
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
            if not isinstance(obj, BaseGrid):
                msg  = 'Expected an instance of a BaseGrid.'
                msg += f' Received {type(obj).__name__}\n'
                msg += f'mro: {type(obj).__mro__}'
                raise TypeError(msg)
        
        #Determine voxels
        #Extract voxels from parent that are not in children
        voxels   = {}
        sizes    = {}
        cache    = set()
        
        def contains(parent_point):
            if parent_point in cache:
                return True
            for child in children:
                if child.contains(parent_point):
                    cache.add(parent_point)
                    return True
            return False
        
        for parent_point, parent_neighbours in parent.voxels.items():
            if contains(parent_point):
                continue
            else:
                temp = {}
                
                for shift, parent_shift_neighbours in parent_neighbours.items():
                    temp.setdefault(shift, [])
                    
                    for parent_neighbour in parent_shift_neighbours:
                        if not contains(parent_neighbour):
                            temp[shift].append(parent_neighbour)
                
                voxels[parent_point] = temp
                sizes[ parent_point] = parent.sizes[parent_point]  
        
        #Iterate through children and update
        ndims         = parent.ndims
        parent_voxels = parent.voxels
        is_boundary   = lambda neighbours: len(neighbours) < ndims*2
        shifts        = [i for i in range(-ndims, ndims+1) if i != 0]
        
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
            
            for child_voxel, child_neighbours in child.voxels.items():
                
                if is_boundary(child_neighbours):
                    
                    temp = {}
                    for shift in shifts:
                        if shift in child_neighbours:
                            temp[shift] = child_neighbours[shift]
                        else:
                            #Find the nearest touching parent voxel
                            shifted           = parent.shift_point(child_voxel, shift)
                            parent_voxel      = parent.voxelize(shifted)
                            parent_neighbours = parent_voxels[parent_voxel]
                            temp[shift]       = [parent_voxel]
                            
                            #Update the parent voxel as well
                            voxels[parent_voxel][-shift] += [child_voxel]
                        
                    voxels[child_voxel] = temp

                else:
                    voxels[child_voxel] = dict(child_neighbours)
                
                sizes[child_voxel] = child.sizes[child_voxel]
        
        #Determine vertices
        vertices = {k: dict(v) for k, v in parent.vertices.items()}
        
        for i, child in enumerate(children):
            for child_point, child_neighbours in child.vertices.items():
                if child_point in vertices:
                    parent_neighbours = dict(parent.vertices[child_point])
                    temp              = {}

                    for key in parent_neighbours:
                        if key in child_neighbours:
                            temp[key] = child_neighbours[key]
                        else:
                            temp[key] = parent_neighbours[key]
                    
                    vertices[child_point] = temp
                       
                else:
                    vertices[child_point] = dict(child_neighbours)
        
        super().__init__(name, 
                         parent.step, 
                         parent.spans, 
                         ndims, 
                         voxels,
                         sizes,
                         vertices
                         )
        self._grids = [parent, *children]
    
    def voxelize(self, point):
        parent, *children = self._grids
        
        for grid in children:
            if grid.contains(point):
                return grid.voxelize(point)
            
        return parent.voxelize(point)
    
    def shift_point(self, point, shift):
        parent, *children = self._grids
        
        for grid in children:
            if grid.contains(point):
                return grid.shift_point(point, shift)
            
        return parent.shift_point(point, shift)

###############################################################################
#Instantiation from Config Dicts
###############################################################################
def make_grids_from_config(grid_config: dict, use_cache=True) -> dict[str, NestedGrid]:
    regular_grids  = make_regular_grids(grid_config)
    nested_grids = {}
    cache        = {} if use_cache else None
    
    for name in regular_grids:
        grid = merge_regular_grids(regular_grids, 
                                 grid_config, 
                                 name,
                                 cache
                                 )
        
        nested_grids[name] = grid
    
    return nested_grids

def make_regular_grids(grid_config: dict) -> dict[str, RegularGrid]:
    regular_grids = {}
    
    for name, config in grid_config.items():
        args = config['config']
        
        try:
            if hasattr(args, 'items'):
                grid = RegularGrid(**args)
            else:
                grid = RegularGrid(*args)
        except Exception as e:
            s = f'Error in instantiating regularGrid {name}.\n'
            a = e.args[0]
            n = s + a
            
            raise type(e)(n)
            
        regular_grids[name] = grid
    
    return regular_grids
    

def merge_regular_grids(regular_grids: dict[str, RegularGrid], 
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
    parent_grid    = regular_grids[parent_name]
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
            child_grid = regular_grids[child_name]
            
            if grid_config[child_name].get('children'):
                child_grid = merge_regular_grids(regular_grids, 
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