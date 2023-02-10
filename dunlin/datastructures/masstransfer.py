import pandas as pd
from numbers import Number
from typing  import Union

import dunlin.utils as ut
from dunlin.datastructures.bases import GenericItem, NamespaceDict
from .stateparam                 import ParameterDict, StateDict
from .coordinatecomponent        import CoordinateComponentDict
from .rate                       import RateDict

class MassTransfer(GenericItem):
    def __init__(self,  
                 ext_namespace        : set,
                 coordinate_components: CoordinateComponentDict,
                 states               : StateDict,
                 parameters           : ParameterDict,
                 name                 : str,
                 species              : str,
                 *parameter_names    
                 ) -> None:
       
        #Check arguments correspond to state/param names
        if species not in states:
            msg = f'State {species} is not in model states.'
            raise ValueError(msg)
        
        
        for i in parameter_names:
            if ut.isnum(i):
                continue
            elif i not in parameters:
                msg = f'Coefficient {i} is not in parameters.'
                raise ValueError(msg)
        
        #Determine the parameter for each axis
        if ut.isnum(parameter_names):
            string = str(parameter_names)
            values = dict.fromkeys(coordinate_components.axes, string)
        elif type(parameter_names) == str:
            values = dict.fromkeys(coordinate_components.axes, parameter_names)
        elif not ut.islistlike(parameter_names):
            msg  = 'Expected a string or a list-like container of strings. '
            msg += f'Received {parameter_names} of type {type(parameter_names).__name__}.'
            raise ValueError(msg)
        elif len(parameter_names) == 1:
            if ut.isnum(parameter_names[0]):
                string = str(parameter_names[0])
            else:
                string = parameter_names[0]
            values = dict.fromkeys(coordinate_components.axes, string)
        elif len(parameter_names) == coordinate_components.ndims:
            strings = [str(i) if ut.isnum(i) else i for i in parameter_names]
            values  = dict(zip(coordinate_components.axes, strings))
        else:
            ndims = coordinate_components.ndims
            msg   = 'Expected either a single parameter for all axes or one parameter '
            msg  += f'for each axis. Received {parameter_names} for {ndims} dimensions.'
            raise ValueError(msg)
        
        #Call the parent constructor
        ext_namespace.add(name)
        self.name          = name
        self.species       = species
        self._coefficients = values
        self.namespace     = tuple([species, *parameter_names])
        self._axis_num     = dict(enumerate(coordinate_components.axes, start=1))
        
        #Freeze
        self.freeze()
    
    @property
    def coefficients(self) -> dict:
        return self._coefficients
    
    def __getitem__(self, axis):
        if type(axis) == str:
            return self._coefficients[axis]
        else:
            #Use absolute values as mass transfer does not vary with axis direction
            axis = abs(axis)
            axis = self._axis_num[abs(axis)]
            return self._coefficients[axis]
    
    def to_data(self) -> Union[Number, list[Number]]:
        lst = [*self.coefficients.values()]
        
        if len(lst) == 1:
            return lst[0]
        
        u = lst[0]
        same = True
        for i in lst[1:]:
            if i != u:
                same = False
        
        if same:
            return lst[0]
        else:
            return lst
        
class MassTransferDict(NamespaceDict):
    itype: type
    
    def __init__(self, 
                 ext_namespace         : set,
                 coordinate_components : CoordinateComponentDict,
                 rates                 : RateDict,
                 states                : StateDict,
                 parameters            : ParameterDict,
                 mapping               : Union[dict, pd.DataFrame],
                 ) -> None:
        
        super().__init__(ext_namespace, 
                         mapping, 
                         coordinate_components, 
                         states, 
                         parameters
                         )
        
        namespace = set()
        seen      = set()
        for mt_species, mt in self.items():
            if mt_species in seen:
                msg = f'Redefinition of mass transfer coefficients for {mt_species}.'
                raise ValueError(msg)
            else:
                seen.add(mt_species)
                
            namespace.update(mt.namespace)
        
        # missing = set(states.names).difference(seen)
        # if rates:
        #     missing = missing.difference(rates.states)
            
        
        # if missing:
        #     msg = f'Missing mass transfer data for one or more states: {missing}'
        #     raise ValueError(msg)
        
        self.namespace = tuple(namespace)
        self.freeze()
    
    def find(self, state, axis):
        return None
    
class Advection(MassTransfer):
    def __init__(self, 
                 ext_namespace         : set,
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 parameters            : ParameterDict,
                 species               : str,
                 *parameter_names    
                 ) -> None:
        name = ut.adv(species)
        
        super().__init__(ext_namespace, 
                         coordinate_components, 
                         states, 
                         parameters, 
                         name, 
                         species, 
                         *parameter_names
                         )

class AdvectionDict(MassTransferDict):
    itype = Advection

class Diffusion(MassTransfer):
    def __init__(self, 
                 ext_namespace         : set,
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 parameters            : ParameterDict,
                 species               : str,
                 *parameter_names    
                 ) -> None:
        name = ut.dfn(species)
        
        super().__init__(ext_namespace, 
                         coordinate_components, 
                         states, 
                         parameters, 
                         name, 
                         species, 
                         *parameter_names
                         )

class DiffusionDict(MassTransferDict):
    itype = Diffusion

    
    