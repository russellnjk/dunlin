from typing import Literal

import dunlin.utils             as ut
from .bases               import DataValue, DataDict
from .coordinatecomponent import CoordinateComponentDict
from .domaintype          import DomainTypeDict

class GeometryDefinition(DataValue):
    def __init__(self,
                 all_names: set(),
                 coordinate_components: CoordinateComponentDict,
                 domain_types: DomainTypeDict,
                 name: str,
                 definition: Literal['csg', 'analytic', 'sampledfield'],
                 domain_type: str,
                 order: int,
                 **kwargs
                 ) -> None:
        
        #Check domain_type
        if domain_type not in domain_types:
            msg  = f'domain_type {domain_type} not found in domain types. '
            msg += f'Available domain types: {list(domain_types.keys())}.'
            raise ValueError(msg)
        
        #Check order
        if not ut.isint(order):
            msg = f'Expected a integer for order. Received {order}'
            raise ValueError(msg)
        elif order < 0:
            msg = f'Expected a zero or positive integer for order. Received {order}'
            raise ValueError(msg)
        
        #Check definition
        if definition not in ['csg', 'analytic', 'sampledfield']:
            raise ValueError(f'Invalid definition {definition}')
        
        #Implement the definition
        ndims = coordinate_components.ndims
        if definition == 'csg':
            args  = {'node': self.define_csg(kwargs, ndims)}
        else:
            raise NotImplementedError(f'{definition} no implemented yet.')
        
        #Call the parent constructor
        super().__init__(all_names,
                         name, 
                         order=order, 
                         definition=definition, 
                         domain_type=domain_type,
                         _args=kwargs,
                         **args
                         )
    @classmethod
    def define_csg(cls, kwargs, ndims):
        if list(kwargs) != ['node']:
            msg  = 'Expected exactly one definition-specific argument: node. '
            msg += f'Received: {kwargs.keys()}'
            raise ValueError(msg)
        
        raw_node = kwargs['node']
        
        if type(raw_node) == str:
            primitives = {2: ['square', 'circle'],
                          3: ['cube', 'sphere', 'cylinder', 'cone']
                          }
            primitives = primitives[ndims]
            if raw_node in primitives:
                node = [raw_node]
            else:
                msg = f'No primitive {raw_node}.'
                raise ValueError(msg)
            
        else:    
            node = cls.parse_node(kwargs['node'], ndims)

        
        return node
        
    @classmethod
    def parse_node(cls, raw_node, ndims):
        primitives = {2: ['square', 'circle'],
                      3: ['cube', 'sphere', 'cylinder', 'cone']
                      }
        primitives = primitives[ndims]
        operations = ['union', 'intersection', 'difference']
        transforms = ['scale', 'translate', 'rotate']
        combined   = primitives + operations + transforms
        first      = None
        node       = []
        
        for i, item in enumerate(raw_node):
            #Check item type
            if type(item) != str and not ut.isnum(item) and not ut.islistlike(item):
                msg = f'Expected a str or list-like item. Received {type(item).__name__}'
                raise TypeError(msg)
            
            #Check for errors in first item
            if i == 0: 
                if item not in combined:
                    msg  = f'Expected first element to be one of {combined}. '
                    msg += f'Received {item}.'
                    raise ValueError(msg)
                elif item == 'difference' and len(raw_node) != 3:
                    msg  = 'Difference operator can only accept two arguments. '
                    msg += f'Received {raw_node}'
                    raise ValueError(msg)
                elif item in ['scale', 'translate'] and len(raw_node) != ndims + 1:
                    msg  = 'Expected {ndims} arguments for transformation {item}. '
                    msg += f'Received {len(item)-1} arguments.'
                    raise ValueError(msg)
                elif item in ['rotate'] and len(raw_node) != ndims + 2:
                    msg  = 'Expected {ndims+1} arguments for transformation {item}. '
                    msg += f'Received {len(item)-1} arguments.'
                    raise ValueError(msg)
                else:
                    first = item
                    node.append(item)
                    continue
            
            #Check for errors in subsequent node
            if item in transforms:
                msg = f'Unexpected transform at position {i} in {raw_node}.'
                raise ValueError(msg)
            elif first not in operations and type(item) == str:
                msg = f'Unexpected argument at position {i} in {raw_node}.'
                raise ValueError(msg)
            elif first in transforms and not ut.isnum(item):
                msg = f'Expected a number at position {i} in {raw_node}.'
                raise ValueError(msg)
            elif first == 'scale' and item <= 0:
                msg = f'Scaling must be be positive. Received {item} in {raw_node}.'
                raise ValueError(msg)
            
            #Proceed with recursion if required
            if type(item) == str:
                temp = item
            elif ut.isnum(item):
                temp = float(item)
            else:
                temp = cls.parse_node(item, ndims)
            node.append(temp)
        
        return node
    
    def to_data(self) -> dict:
        definition = self.definition
        
        dct = {'definition'  : definition,
               'domain_type' : self.domain_type,
               'order'       : self.order,
               }
        
        dct.update(self._args)
        
        return dct
    
class GeometryDefinitionDict(DataDict):
    itype = GeometryDefinition
    
    def __init__(self, 
                 all_names: set, 
                 coordinate_components: CoordinateComponentDict,
                 domain_types: DomainTypeDict, 
                 mapping: dict
                 ) -> None:
            
        super().__init__(all_names, mapping, coordinate_components, domain_types)
        
        #Check order (ordinal) and sort
        seen       = set()
        order2name = {}
        for name, gdef in self.items():
            order = gdef.order
            
            if order in seen:
                raise ValueError(f'Repeat of order {order} in geometry definition.')
            
            seen.add(order)
            order2name[order] = name
        
        _data      = self._data
        seen       = sorted(seen)
        self._data = {order2name[order]: _data[order2name[order]] for order in seen}
        
