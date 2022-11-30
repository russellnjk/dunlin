import pandas as pd
from typing import Union

import dunlin.comp as cmp
from .spatialreaction    import SpatialReactionDict
from .boundarycondition import BoundaryConditionDict
from .compartment       import CompartmentDict
from .masstransfer      import AdvectionDict, DiffusionDict

from .coordinatecomponent import CoordinateComponentDict
from .gridconfig          import GridConfigDict
from .domaintype          import DomainTypeDict
from .adjacentdomain      import AdjacentDomainsDict
from .geometrydefinition  import GeometryDefinitionDict

from .modeldata    import ModelData
from .ode          import ODEModelData
from .geometrydata import GeometryData

class SpatialModelData(ODEModelData):
    def _init_geometry(self,
                       coordinate_components: dict,
                       grid_config: dict,
                       domain_types: dict,
                       adjacent_domains: dict,
                       geometry_definitions: dict
                       ):
        
        namespace = self.namespace
        
        ccds  = CoordinateComponentDict(coordinate_components)
        gcfg  = GridConfigDict(namespace, ccds, grid_config)
        dmnts = DomainTypeDict(namespace, ccds, domain_types)
        admns = AdjacentDomainsDict(namespace, ccds, dmnts, adjacent_domains)
        gdefs = GeometryDefinitionDict(namespace, ccds, dmnts, geometry_definitions)
        
        self.coordinate_components = ccds
        self.grid_config           = gcfg
        self.domain_types          = dmnts
        self.geometry_definitions  = gdefs
        self.adjacent_domains      = admns        
    
    def _init_masstransfer(self,
                           compartments: dict,
                           advection: dict = None,
                           diffusion: dict = None,
                           boundary_conditions: dict = None
                           ):
        namespace = self.namespace
        
        compartments = CompartmentDict(namespace, 
                                       self.states, 
                                       self.domain_types,
                                       compartments
                                       )
        advection    = AdvectionDict(namespace, 
                                     self.coordinate_components, 
                                     self.rates,
                                     self.states, 
                                     self.parameters, 
                                     advection
                                     )
        diffusion    = DiffusionDict(namespace, 
                                     self.coordinate_components, 
                                     self.rates,
                                     self.states, 
                                     self.parameters, 
                                     diffusion
                                     )
        
        
        self.compartments = compartments
        self.advection    = advection
        self.diffusion    = diffusion
        
        #Boundary conditions
        self.boundary_conditions = BoundaryConditionDict(namespace, 
                                                         self.coordinate_components, 
                                                         self.states,
                                                         boundary_conditions
                                                         )
        
       
    def __init__(self,
                 ref: str, 
                 states: Union[dict, pd.DataFrame], 
                 parameters: Union[dict, pd.DataFrame], 
                 compartments: dict,
                 geometry: dict,
                 functions: dict = None, 
                 variables: dict = None, 
                 reactions: dict = None, 
                 rates: dict = None, 
                 events: dict = None, 
                 extras: Union[dict, callable] = None,
                 units: dict = None,
                 advection: dict = None,
                 diffusion: dict = None,
                 boundary_conditions: dict = None
                 ) -> None:
        
        #Call parent constructor
        super().__init__(ref,
                         states,
                         parameters,
                         functions,
                         variables,
                         reactions,
                         rates,
                         events,
                         extras,
                         units,
                         _reactions=SpatialReactionDict
                         )
        
        #Unfreeze to allow modification
        self.unfreeze()
        self.namespace = set(self.namespace)
        
        #Create geometry data structures
        self._init_geometry(**geometry)
        
        #Extend the model with spatial-based items
        self._init_masstransfer(compartments, advection, diffusion, boundary_conditions)
            
        #Map the reactions to the compartments
        self.reaction_compartments = None
        
        #Freeze
        self.namespace = frozenset(self.namespace)
        self.freeze()
        
    def to_data(self, recurse=True) -> dict:
        keys = ['states', 
                'parameters', 
                'functions',
                'variables',
                'reactions',
                'rates',
                'events',
                'units',
                'advection',
                'diffusion',
                'boundary_conditions'
                ]
        data0 = self._to_data(keys, recurse)
        
        keys = ['coordinate_components',
                'grid_config',
                'domain_types',
                'adjacent_domains',
                'geometry_definitions',
                ]
        
        data1 = self._to_data(keys, recurse)
        
        data0[self.ref]['geometry'] = data1[self.ref]
        
        return data0
    
        
        