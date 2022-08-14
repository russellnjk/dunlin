import numpy as np
from numbers import Number
from typing import Union

import dunlin.utils_plot as upp
from .grid import BasicGrid, NestedGrid

class ShapeStack:
    #For plotting
    _scatter_args = dict(s=100, label='{name}')
    _line_args    = dict(color='black', marker='None', linewidth=1)
    
    def __init__(self, grid: Union[BasicGrid, NestedGrid], *shapes) -> None:
        self._grid = grid
        
        points     = grid.points
        shape_nums = np.zeros(len(points), dtype=np.int32) 
        
        for i, shape in enumerate(shapes, start=1):
            is_inside             = shape.contains_points(points)
            shape_nums[is_inside] = i
        
        points_      = list(grid.graph)
        point2shape  = dict(zip(points_, shape_nums))
        shape2points = {}
        edges        = {}
        graph        = grid.graph
        
        for point, shape_num in point2shape.items():
            shape2points.setdefault(shape_num, []).append(point)
            
            for neighbour in graph[point].values():
                shape_num_ = point2shape[neighbour]
                

                edge = tuple(sorted([shape_num, shape_num_]))
                pair = tuple(sorted([point, neighbour]))
                
                lst = edges.setdefault(edge, [])
                if pair not in lst:
                    lst.append(pair)
                
        self._point2shape  = point2shape
        self._shape2points = shape2points
        self._edges        = edges
        self._names        = [getattr(obj, 'name', '') for obj in [grid, *shapes]]
        
    @property
    def point2shape(self) -> dict[tuple, int]:
        return self._point2shape
    
    @property
    def shape2points(self) -> dict[int, tuple]:
        return self._shape2points
    
    @property
    def edges(self) -> dict[tuple[int, int], tuple[tuple, tuple]]:
        return self._edges
    
    @property
    def names(self) -> dict[int, str]:
        return self._names
    
    def __getitem__(self, shape_num) -> np.ndarray:
        points = np.array(self.shape2points[shape_num])
        return points
    
    def plot_lines(self, ax, edge=None, **line_args) -> list:
        #Check the format of edge
        if edge == None:
            edge = list(self.edges)
            return [self.plot_lines(ax, i, **line_args) for i in edge]
        if isinstance(edge, Number):
            edge = edge, edge
        elif len(edge) == 2 and all([isinstance(i, Number) for i in edge]):
            edge = tuple(sorted(edge))
        elif [len(i) == 2 and all([isinstance(ii, Number) for ii in i]) for i in edge]:
            return [self.plot_lines(ax, i, **line_args) for i in edge]
        else:
            msg  = 'Expected a number, pair numbers or a list of pairs of numbers.'
            msg += f' Received {edge}'
            raise ValueError(msg)
            
        #Parse plot args
        names        = self.names[edge[0]], self.names[edge[1]]
        sub_args     = {'edge': edge, 'name': names}
        converters   = {'color': upp.get_color}
        line_args    = upp.process_kwargs(line_args, 
                                          [edge], 
                                          self._line_args, 
                                          sub_args, 
                                          converters
                                          )
        
        #Plot on the axes
        if edge not in self.edges:
            return []
        
        pairs = self.edges[edge]
        pairs = np.array(pairs)
        
        result = []
        for pair in pairs:
            pair = pair.T
            temp = ax.plot(*pair, **line_args)
            result.append(temp)
            
        return result
    
    def plot_shape(self, ax, shape_num=None, **scatter_args) -> list:
        #Check the format of shape_num
        if shape_num is None:
            shape_num = list(self.shape2points)
            return [self.plot_shape(ax, i, **scatter_args) for i in shape_num]
        elif isinstance(shape_num, Number):
            pass
        elif all([isinstance(i, Number) for i in shape_num]): 
            return [self.plot_shape(ax, i, **scatter_args) for i in shape_num]
        else:
            msg = f'Expected a number or list of numbers. Received {shape_num}'
            raise ValueError(msg)
            
        #Parse plot args
        sub_args     = {'shape_num': shape_num, 'name': self.names[shape_num]}
        converters   = {'c': upp.get_color}
        scatter_args = upp.process_kwargs(scatter_args, 
                                          [shape_num], 
                                          self._scatter_args, 
                                          sub_args, 
                                          converters
                                          )
    
        #Plot on the axes
        points = self[shape_num]
        values = points.T
        result = ax.scatter(*values, **scatter_args)
        
        return result
    