from numbers import Number
from typing  import Literal

import dunlin.utils                    as ut
from dunlin.datastructures.bases import DataDict, DataValue
from .stateparam                 import StateDict
from .coordinatecomponent        import CoordinateComponentDict


'''
These classes differ significantly from their SBML counterparts.

In SBML, compartments represent physical locations of chemical species.
Meanwhile, domain types represent regions in space where a particular species exists 
with a particular initial value is allowed to exist.
    
SBML defines a one-to-one mapping between species and compartments, and a 
many-to-one mapping between compartments and domain types. This is 
confusing because the geometry is defined entirely be domain types. The 
result is that the compartments cannot be mapped to a specific region of 
the geometry. Instead, the region specified by a domain type will contain 
all species from associated compartments throughout its entirety. Thus, 
species from different compartments would exist in the same space if they 
were assigned to the same domain type. This contradicts the purpose of 
having compartments in the first place.

This contradiction can be solved if the domain type could be subdivided 
into subregions. SBML compartments define an attribute called the unit size. There 
are two ways to interpret the unit size:
    1. The unit sizes sum to one. Each domain_type represents a fraction 
    of domain type although where exactly each domain_type exists is 
    left undefined in the model. This happens when the compartment has the 
    same number of dimensions as the domain type.
    2. The unit size represents a conversion between dimensions 
    e.g. ratio of 3D volume to 2D area. This happens when the compartment has 
    a different number of dimensions from the domain type.

The first case is problematic as there is no way to know exactly exactly 
where in physical space each compartment is located within a domain type. 
Also, SBML does not require that the unit sizes sum to 1.

The second case is also problematic as the interpretation is ambiguous unless 
the number of dimensions of the compartment is one fewer than that of the 
domain type. For example, a 2-D compartment represents the surface of a 
3-D domain type. However, SBML lacks guidance on how to interpret surface-bound 
states. Furthermore, such simulations often have complex requirements that 
depend on the nature of the problem e.g. steric effects, transmembrane structures 
etc. And once again, SBML does not require that the unit sizes for this 
application sum to one.

The domains and adjacent domains in SBML adds to the confusion. This is 
because 
    1. Domains are expected to be unconnected but SBML does not enforce it.
    2. Adjacent domains can naturually arise from the geometry definitions 
    but SBML does not state if the list of adjacent domains serves as 
    a check or how to handles discrepancies.
    
The resulting framework seems not only unecesarily confusing. In addition, 
the broadly defined range of inputs and interpretations also make the 
it prone to erroneous input that could potentially be flagged out even 
before instantiation. User-friendliness is also wanting.

I have therefore made changes as follows:
    1. Compartments are removed. When a group of states occupies a region 
    of space with the same number of dimensions as the model, they are 
    grouped under a Dunlin domain type. This corresponds to the case where 
    the compartment has the same number of dimensions as the domain type 
    
    Dunlin does not explicitly account for case where the compartment has one 
    fewer dimension than the domain type. When the user wants to simulate a 
    membrane bound protein, they could define a reaction that takes place at 
    a surface between two domains. The rate of the reaction could include a 
    dummy state that exists on the other side of the membrane and has a value 
    of 1. This causes the reaction to be defined only at the surface. The 
    membrane bound version of the protein would have no diffusion or advection 
    coefficients. More complicated scenarios require extensions beyond what 
    SBML specifies (according to my current understanding).
    
    3. States now have a one-to-one mapping with domain types or surfaces. 
    All states of the same domain type will exist in the same region of space. 
    All states of the same surface will exist in the same region of space.

    4. Dunlin domains are now interpreted as instances of a Dunlin domain type. 
    Domains of the same domain type cannot touch. An exception is raised otherwise.
    
    4. Dunlin surfaces are instances of a Dunlin surface type. Unexpected, 
    missing or erroneous surfaces raise an exception. They replace the SBML
    adjacent domains.

The model definition requires the user to specify domain types, domains, 
surface types and surfaces. In principle, numerical integration does not 
require knowledge of domains and surfaces. However, such information can 
be useful if one needs to extract information from a particular subregion of 
space.

'''
###############################################################################
#Domain/Surface Types
###############################################################################
class ContainerType(DataValue):
    itype = ''
    
    def __init__(self,
                 all_names       : set,
                 all_states      : StateDict,
                 state2container : dict[str, str],
                 name            : str,
                 states          : list[str] = None,
                 **kwargs
                 ):
        
        #Check the states
        if states:
            for state in states:
                if state not in all_states:
                    msg = f'{self.itype} {name} contains an undefined state: {state}.'
                    raise NameError(msg)
                elif state in state2container:
                    msg = f'State {state} was assigned to at least two {self.itype}:'
                    msg = f'{msg} {name} and {state2container[state]}.'
                    raise ValueError(msg)
                
                state2container[state] = name
        else:
            states = []
                
        #Call the parent constructor
        super().__init__(all_names, 
                         name, 
                         states = frozenset(states),
                         **kwargs
                         )
    
class DomainType(ContainerType):
    itype = 'domain_type'
    
    def __init__(self,
                 all_names             : set,
                 coordinate_components : CoordinateComponentDict,
                 all_states            : StateDict,
                 state2domain_type     : dict[str, str],
                 domain2domain_type    : dict[str, str],
                 internal_points       : dict[tuple, str],
                 name                  : str,
                 states                : list[str]               = None,
                 domains               : dict[str, list[Number]] = None
                 ):
        
        domain2internal_point = {}
        
        if domains:
            for domain, internal_point in domains.items():
                #Check name
                if not ut.is_valid_name(name):
                    msg = f'Invalid name provided for domain: {name}'
                    raise ValueError(msg)
                elif name in all_names:
                    msg = f'Repeated definition of {domain}.'
                    raise NameError(msg)
                    
                #Parse and check internal point
                spans = list(coordinate_components.spans.values())
                ndims = coordinate_components.ndims
                
                if any([not isinstance(i, Number) for i in internal_point]):
                    msg  = f'Error in {type(self).__name__} {name}.'
                    msg += 'Internal point can ony contain numbers.'
                    raise TypeError(msg)
                
                
                if len(internal_point) != coordinate_components.ndims:
                    msg  = f'Error in {type(self).__name__} {name}.'
                    msg += f' Expected an internal point with {ndims} coordinates.'
                    msg += f' Received {internal_point}'
                    raise ValueError(msg)
                
                for i, span in zip(internal_point, spans):
                    if i <= span[0] or i >= span[1]:
                        msg  = f'Error in {type(self).__name__} {name}.'
                        msg += 'Internal point must be lie inside coordinate components.'
                        msg += ' Spans: {spans}. Received {internal_point}'
                        raise ValueError(msg)
                
                if tuple(internal_point) in internal_points:
                    msg  = f'Error in {type(self).__name__} {name}.'
                    msg += f' Repeated internal points {internal_point}.'
                    raise ValueError(msg)
                
                
                internal_points.add(tuple(internal_point))
                domain2internal_point[domain] = list(internal_point)
                domain2domain_type[domain]    = name
                
        super().__init__(all_names, 
                         all_states, 
                         state2domain_type,
                         name,
                         states,
                         domain2internal_point = domain2internal_point 
                         )
       
    def to_dict(self) -> dict:
        dct = {}
        if self.states:
            dct['states'] = list(self.states)
        if self.domain2internal_point:
            dct['domains'] = self.domain2internal_point
        if dct:
            dct = {self.name: dct}
        
        return dct
    
class SurfaceType(ContainerType):
    itype = 'surface_type'
    
    def __init__(self,
                 all_names             : set,
                 all_states            : StateDict,
                 state2surface_type    : dict[str, str],
                 domain_pair2surface   : dict[str, str],
                 surface2surface_type  : dict[str, str],
                 name                  : str,
                 states                : list[str]                  = None,
                 surfaces              : dict[str, list[str, str]]  = None
                 ):
        
        surface2domain_pair = {}
        
        if surfaces:
            for surface, domain_pair in surfaces.items():
                if surface in surface2surface_type:
                    msg  = f'Repeated definition of surface {surface}.'
                    msg += f'This surface was found in surface type {surface2surface_type[surface]}.'
                    raise NameError(msg)
                elif surface in all_names:
                    msg = f'Repeated definition of {surface}.'
                    raise NameError(msg)
                    
            
                domain_pair_ = tuple(sorted(domain_pair))
                
                if domain_pair_ in domain_pair2surface:
                    msg  = f'Repeated definition of domain pair {domain_pair}.'
                    msg += f'This pair was found in surface {domain_pair2surface[domain_pair_]}.'
                    raise NameError(msg)
                
                surface2surface_type[surface]     = name
                domain_pair2surface[domain_pair_] = surface
                surface2domain_pair[surface]      = list(domain_pair)
        
        super().__init__(all_names, 
                         all_states, 
                         state2surface_type, 
                         name,
                         states,
                         surface2domain_pair = surface2domain_pair
                         )
        
    def to_dict(self) -> dict:
        dct = {}
        if self.states:
            dct['states'] = list(self.states)
        if self.surface2domain_pair:
            dct['surfaces'] = self.surface2domain_pair
        if dct:
            dct = {self.name: dct}
        
        return dct
        

###############################################################################
#Dicts for Domain/Surface Types
###############################################################################
class DomainTypeDict(DataDict):
    itype = DomainType
    
    def __init__(self, 
                 all_names             : set, 
                 coordinate_components : CoordinateComponentDict,
                 states                : StateDict,
                 mapping               : dict
                 ):
        
        state2domain_type  = {}
        domain2domain_type = {}
        internal_points    = set()
        
        super().__init__(all_names, 
                         mapping, 
                         coordinate_components,
                         states, 
                         state2domain_type,
                         domain2domain_type,
                         internal_points
                         )
        
        #Update
        self.state2domain_type  = state2domain_type
        self.domain2domain_type = domain2domain_type
        
class SurfaceTypeDict(DataDict):
    itype = SurfaceType
    
    def __init__(self, 
                 all_names    : set, 
                 states       : StateDict,
                 domain_types : DomainTypeDict,
                 mapping      : dict
                 ) -> None:
        
        state2surface_type   = {}
        domain_pair2surface  = {}
        surface2surface_type = {}
        
        super().__init__(all_names, 
                         mapping,
                         states, 
                         state2surface_type,
                         domain_pair2surface,
                         surface2surface_type,
                         )
        
        surface_states = set(state2surface_type)
        repeated       = surface_states.intersection(domain_types.state2domain_type)
        union          = surface_states.union(domain_types.state2domain_type)
        expected       = set(states.names)
        missing        = expected.difference(union)
        
        if repeated:
            msg = f'States {repeated} appeared in both domain and surface types.'
            raise ValueError(msg)
        elif missing:
            msg = f'States {missing} not assigned to a domain or surface type.'
            raise ValueError(msg)
        
class Surface(DataValue):
    def __init__(self,
                 all_names      : set,
                 domain_types   : DomainTypeDict,
                 domain2surface : set,
                 name           : str,
                 domain0        : str,
                 domain1        : str
                 ):
        
        domain2domain_type = domain_types.domain2domain_type
        
        if type(domain0) != str:
            msg = f'Domains must be strings. Received {type(domain0)}.'
            raise TypeError(msg)
        if type(domain1) != str:
            msg = f'Domains must be strings. Received {type(domain1)}.'
            raise TypeError(msg)
        if domain0 not in domain2domain_type:
            msg = f'Unexpected domain {domain0}.'
            raise ValueError(msg)
        if domain1 not in domain2domain_type:
            msg = f'Unexpected domain {domain1}.'
            raise ValueError(msg)
        if domain0 == domain1:
            msg = f'Received repeated domains : ({domain0}, {domain1}).'
            raise ValueError(msg)
        
        tup = tuple(sorted([domain0, domain1])) 
        if tup  in domain2surface:
            msg = f'Repeated pair of domains {domain0}, {domain1}.'
            raise ValueError(msg)
            
        super().__init__(all_names,
                         name,
                         domains     = tup,
                         domains_ori = [domain0, domain1]
                         )
        
        domain2surface[tup] = name
    
    def to_dict(self) -> dict:
        dct = {self.name: list(self.domains_ori)}
        return dct
    
    def __iter__(self):
        return iter(self.domains)
    
    def __getitem__(self, idx: Literal[0, 1]) -> str:
        return self.domains[idx]

class SurfaceDict(DataDict):
    itype = Surface
    
    def __init__(self, 
                 all_names    : set,
                 domain_types : DomainTypeDict,
                 surfaces     : dict
                 ):
        domain2surface = {}
        
        super().__init__(all_names, 
                         surfaces,
                         domain_types,
                         domain2surface
                         )
        self.domain2surface = domain2surface
        
        
    