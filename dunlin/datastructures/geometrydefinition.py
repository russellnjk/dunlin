from numbers import Number
from typing import Any, Literal, Union

import dunlin.utils             as ut
from .bases               import DataValue, DataDict
from .coordinatecomponent import CoordinateComponentDict
from .compartment         import CompartmentDict
from .domain              import DomainDict


class GeometryDefinition(DataValue):
    def __init__(self,
                 all_names                : set(),
                 coordinate_components    : CoordinateComponentDict,
                 compartments             : CompartmentDict,
                 order2geometrydefinition : dict,
                 name                     : str,
                 geometry_type            : Literal['csg', 'analytic', 'sampledfield', 'parametric'],
                 compartment              : str,
                 order                    : int,
                 definition               : Any
                 ) -> None:
        
        #Check compartment
        if compartment not in compartments:
            msg  = f'Compartment {compartment} not found in domain types. '
            msg += f'Available compartments: {list(compartments.keys())}.'
            raise ValueError(msg)
        
        #Check order
        if not ut.isint(order):
            msg = f'Expected a integer for order. Received {order}'
            raise ValueError(msg)
        elif order < 0:
            msg = f'Expected a zero or positive integer for order. Received {order}'
            raise ValueError(msg)
        elif order in order2geometrydefinition:
            other = order2geometrydefinition[order]
            msg   = f'Geometry definitions {name} and {other} share order: {order}.'
            raise ValueError(msg)
            
        #Check geometry_type
        allowed = 'csg', 'analytic', 'sampledfield'
        if geometry_type not in allowed:
            temp = ', '.join(allowed)
            msg  = f'Invalid geometry_type {geometry_type}. Allowed values are: {temp}.'
            raise ValueError(msg)
        
        #Parse the definition
        method = getattr(self, '_parse_' + geometry_type, None)
        
        if not method:
            msg  = f'Geometry type {geometry_type} not implemented yet. '
            msg += 'This should change in future versions.'
            raise NotImplementedError(msg)
        
        definition_copy = method(name, definition, coordinate_components)
        
        #Call the parent constructor
        super().__init__(all_names,
                         name, 
                         order         = order, 
                         compartment   = compartment,
                         definition    = definition_copy, 
                         geometry_type = geometry_type
                         )
        
        order2geometrydefinition[order] = name
    
    @classmethod
    def _parse_csg(cls, 
                   name                  : str,
                   definition            : Union[list, str], 
                   coordinate_components : CoordinateComponentDict
                   ) -> Union[list, str]:
        
        #Determine which primitives, operators and transformations are allowed
        ndims           = coordinate_components.ndims
        transformations = {'translate', 'scale', 'rotate'}
        operators       = {'union', 'intersection', 'difference'}
        if ndims == 2:
            primitives = {'square', 'circle'}
        else:
            primitives = {'cube', 'sphere', 'cylinder', 'cone'}
        
        #Parse and if necessary, recurse
        if type(definition) == str:
            #Case 1: Primitive
            if definition in primitives:
                return definition
            else:
                temp = ', '.join(primitives)
                msg  = f'Error in parsing definition of {name}. '
                msg += f'The following definition is invalid: \n{definition}\n'
                msg += 'Expected a CSG primitive but received {definition}. '
                msg += f'The allowed values are: {temp}.'
                raise ValueError(msg)
                
        elif type(definition) == list:
            if not list:
                msg  = f'Error in parsing definition of {name}. '
                msg += f'The following definition is invalid: \n{definition}\n'
                msg += 'Empty lists are not valid.'
                raise ValueError(msg)
                
            #Case 2: Operator
            elif definition[0] in operators:
                operator = definition[0]
                
                if len(definition) < 3:
                    msg  = f'Error in parsing definition of {name}. '
                    msg += f'The following definition is invalid: \n{definition}\n'
                    msg += 'Operators should be lists with length of at least 3. '
                    msg += 'The format should be [<operator>, <item1>, <item2>...].'
                    raise ValueError(msg)
                    
                elif operator == 'difference' and len(definition) != 3:
                    msg  = f'Error in parsing definition of {name}. '
                    msg += f'The following definition is invalid: \n{definition}\n'
                    msg += 'Difference operators should be lists with length of exactly 3. '
                    msg += 'The format should be [<operator>, <item1>, <item2>].'
                    raise ValueError(msg)
                
                temp = [cls._parse_csg(name, x, coordinate_components) for x in definition[1:]]
                
                return [operator, *temp]
            
            #Case 3: Transformation
            elif definition[0] in transformations:
                transformation = definition[0]
                length         = len(definition)
                
                if transformation == 'rotate' and length != ndims + 3: 
                    msg  = f'Error in parsing definition of {name}. '
                    msg += f'The following definition is invalid: \n{definition}\n'
                    msg += 'Rotations should be lists with length of at ndims + 3. '
                    msg += 'The format should be [<operator>, <angle>, <x>, <y>, <z if applicable>, <item>].'
                    raise ValueError(msg)
                elif transformation != 'rotate' and length != ndims + 2:
                    msg  = f'Error in parsing definition of {name}. '
                    msg += f'The following definition is invalid: \n{definition}\n'
                    msg += 'Rotations should be lists with length of at ndims + 2. '
                    msg += 'The format should be [<operator>, <x>, <y>, <z if applicable>, <item>].'
                    raise ValueError(msg)
                
                numbers = definition[1:-1]
                if any([not isinstance(x, Number) for x in numbers]):
                    msg  = f'Error in parsing definition of {name}. '
                    msg += f'The following definition is invalid: \n{definition}\n'
                    msg += 'All values following the item to be transformed must be numbers. '
                    raise ValueError(msg)
                    
                temp = cls._parse_csg(name, definition[-1], coordinate_components) 
                
                return [transformation, *numbers, temp]
                
            else:
                msg  = f'Error in parsing definition of {name}. '
                msg += f'The following definition is invalid: \n{definition}\n'
                msg += f'Invalid operator/transformation {definition[0]}.'
                raise ValueError(msg)
        else:
            msg  = f'Error in parsing definition of {name}. '
            msg += f'The following definition is invalid: \n{definition}\n'
            msg += 'CSG definitions can only contain lists and strings. '
            msg += f'Encountered {type(definition)}'
            raise ValueError(msg)
        
    def to_dict(self) -> dict:
        dct = {'geometry_type' : self.geometry_type,
               'compartment'   : self.compartment,
               'order'         : self.order,
               'definition'    : self.definition,
               }
        
        dct = {self.name: dct}
        return dct
    
class GeometryDefinitionDict(DataDict):
    itype = GeometryDefinition
    
    def __init__(self, 
                 all_names             : set, 
                 coordinate_components : CoordinateComponentDict,
                 compartments          : CompartmentDict, 
                 mapping               : dict
                 ):
        
        order2geometrydefinition = {}
        
        super().__init__(all_names, 
                         mapping, 
                         coordinate_components, 
                         compartments, 
                         order2geometrydefinition
                         )
        
        #Sort the geometry definitions
        sorted_geometry_definitions = [order2geometrydefinition[k] for k in sorted(order2geometrydefinition)]
        
        #Save the attributes
        self.order2geometrydefinition    = order2geometrydefinition
        self.sorted_geometry_definitions = sorted_geometry_definitions
        
        
        