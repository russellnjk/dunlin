import numpy as np
from numbers import Number
from typing import Union

import dunlin.utils      as ut
import dunlin.utils_plot as upp
from .grid                                  import BasicGrid, NestedGrid
from .csgnode                               import parse_node 
from dunlin.spatial.geometrydefinition.grid import make_grids_from_config

class ShapeStack:
    #For plotting
    _scatter_args = dict(s=100, label='{name}')
    _line_args    = dict(color=lambda edge, name: 'gray' if edge[0] == edge[1] else 'black', 
                         marker='None', 
                         linewidth=1
                         )
                
    @classmethod
    def from_geometry_data(cls, geometry_data, _full=False):
        gdefs   = geometry_data['geometry_definitions']
        shapes  = {}
        
        for gdef_name, gdef in gdefs.items():
            if gdef.definition == 'csg':
                node  = gdef.node
                shape = parse_node(node, gdef_name)
                order = gdef.order
                
                shapes[order] = shape
            else:
                raise NotImplementedError(f'No implementation for {gdef.definition} yet.')

        #Sort the shapes
        shapes = [shapes[i] for i in sorted(shapes)]

        #Make the grid
        grid_config  = geometry_data['grid_config'].to_data()
        nested_grids = make_grids_from_config(grid_config)
        main_grid    = next(iter(nested_grids.values()))

        #Stack the shapes
        stk = cls(main_grid, *shapes)
        
        if _full:
            return stk, nested_grids
        return stk
    
    def __init__(self, grid: Union[BasicGrid, NestedGrid], *shapes) -> None:
        points     = grid.points
        shape_nums = np.zeros(len(points), dtype=np.int32) 
        name2num   = {}
        num2name   = {}
        
        for i, shape in enumerate(shapes, start=1):
            is_inside             = shape.contains_points(points)
            shape_nums[is_inside] = i
            
            if hasattr(shape, 'name'):
                name = shape.name
                if type(name) != str and name is not None:
                    t    = f'{type(name).__name__}'
                    msg  = 'Shape names must be string. '
                    msg += f'Shape {shape} at index {i-1} is of type {t}.'
                    raise ValueError(msg)
                    
                if name in name2num:
                    msg = f'Encountered more than one shape with name {name}.'
                    raise ValueError(msg)
                elif name is not None:
                    name2num[name] = i
                    num2name[i]    = name
                    
        points_      = list(grid.graph)
        point2shape  = dict(zip(points_, shape_nums))
        shape2points = {}
        edges        = {}
        graph        = grid.graph
        
        #Iterate to get edges
        for point, shape_num in point2shape.items():
            shape2points.setdefault(shape_num, set()).add(point)
            
            for neighbour in graph[point].values():
                shape_num_ = point2shape[neighbour]
                
                edge = tuple(sorted([shape_num, shape_num_]))
                pair = tuple(sorted([point, neighbour]))
                
                st = edges.setdefault(edge, set())
                st.add(pair)
        
        self._grid         = grid
        self.shapes        = shapes
        self._point2shape  = point2shape
        self._shape2points = shape2points
        self._edges        = edges
        self._names        = [getattr(obj, 'name', '') for obj in [grid, *shapes]]
        self.name2num      = name2num
        self.num2name      = num2name
    
    @property
    def graph(self) -> dict:
        return self._grid.graph
    
    @property
    def grid(self) -> np.ndarray:
        return self._grid
    
    @property
    def points(self) -> np.ndarray:
        return self._grid._points

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
    
    def get_points(self, shape_num):
        if type(shape_num) == str:
            shape_num = self.names2num[shape_num]
        
        #Convert frozenset to array
        points = self.shape2points[shape_num]
        points = np.array(tuple(points))
        return points
    
    def get_edges(self, shape_num0, shape_num1):
        if type(shape_num0) == str:
            shape_num0 = self.name2num[shape_num0]
        if type(shape_num1) == str:
            shape_num1 = self.name2num[shape_num1]
        
        # key = frozenset([shape_num0, shape_num1])
        key = tuple(sorted([shape_num0, shape_num1]))
        
        #Convert frozenset to array
        edges = self.edges[key]
        edges = np.array(tuple(edges))
        return edges
    
    def get_shape_num(self, point):
        
        shape_num = self.point2shape[point]
        
        if shape_num == 0 :
            return None
        else:
            return shape_num - 1
    
    def get_shape(self, point):
        shape_num = self.get_shape_num(point)
        
        if shape_num is None:
            return None
        else:
            return self.shapes[shape_num]
    
    def __getitem__(self, key) -> np.ndarray:
        if ut.islistlike(key):
            return self.get_edges(*key)
        else:
            return self.get_points(key)
    
    
    def plot_edges(self, ax, edge=None, **line_args) -> list:
        #Check the format of edge
        if edge == None:
            edge = self.edges
            return [self.plot_edges(ax, i, **line_args) for i in edge]
        if isinstance(edge, Number):
            edge = edge, edge
        elif len(edge) == 2 and all([isinstance(i, Number) for i in edge]):
            edge = tuple(sorted(edge))
        elif [len(edge) == 2 and all([isinstance(ii, Number) for ii in i]) for i in edge]:
            return [self.plot_edges(ax, i, **line_args) for i in edge]
        else:
            msg  = 'Expected a number, pair numbers or a list of pairs of numbers.'
            msg += f' Received {edge}'
            raise ValueError(msg)
            
        #Parse plot args
        names        = [self.num2name.get(i, None) for i in edge]
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
        
        pairs = self[edge]
        
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
    