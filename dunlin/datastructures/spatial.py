import pandas as pd
from typing import Union

import dunlin.comp as cmp

from .boundarycondition   import BoundaryConditionDict
from .compartment         import CompartmentDict
from .masstransfer        import AdvectionDict, DiffusionDict

from .coordinatecomponent import CoordinateComponentDict
from .gridconfig          import GridConfig
from .domain              import DomainDict
from .adjacentdomain      import AdjacentDomainDict
from .geometrydefinition  import GeometryDefinitionDict

from .ode          import ODEModelData

class SpatialModelData(ODEModelData):
    '''
    For hierarchical models, the following data from the child model is extracted 
    into the parent model and flattened:
        1. Same as ODEModelData
            1. States
            2. Parameters
            3. Funcions
            4. Variables
            5. Reactions
            6. Rates
            7. Events
        2. Specific to SpatialModelData
            1. Compartments
            2. Advection
            3. Diffusion
            4. Boundary conditions
    
    Geometry-related data is skipped because of the parent geometry can be 
    very different from that of the child or may require child geometries to 
    be merged resulting in very difficult interpretation. Therefore, the user 
    must specify all geometry in the parent model instead of having it 
    automatically extracted from the child model.
    
    Peripheral information such as metadata and int_args are skipped as is the 
    case for ODEModelData.
        
    '''
    @classmethod
    def from_all_data(cls, all_data, ref):
        required_fields = {'states'              : [True, False, False],
                           'parameters'          : [True, False, False],
                           'functions'           : [True, False],
                           'variables'           : [True, True],
                           'reactions'           : [True, True, True],
                           'rates'               : [True, True],
                           'events'              : [True, False, True],
                           'compartments'        : [True, True],
                           'advection'           : [True, True],
                           'diffusion'           : [True, True],
                           'boundary_conditions' : [True, False, False, False],
                           }
        
        flattened  = cmp.flatten_model(all_data, ref, required_fields)
        
        return cls(**flattened)
    
    def __init__(self,
                 ref                   : str, 
                 states                : Union[dict, pd.DataFrame], 
                 parameters            : Union[dict, pd.DataFrame], 
                 compartments          : dict,
                 coordinate_components : dict,
                 grid_config           : dict,
                 domains               : dict,
                 adjacent_domains      : dict,
                 geometry_definitions  : dict,
                 functions             : dict = None, 
                 variables             : dict = None, 
                 reactions             : dict = None, 
                 rates                 : dict = None, 
                 events                : dict = None, 
                 units                 : dict = None,
                 advection             : dict = None,
                 diffusion             : dict = None,
                 boundary_conditions   : dict = None,
                 meta                  : dict = None,
                 int_args              : dict = None,
                 sim_args              : dict = None,
                 opt_args              : dict = None
                 ):
        
        #Call the parent constructor to specify which attributes are exportable
        #This allows the input to be reconstructed assuming no modification
        #after instantiation
        super()._set_exportable_attributes(['ref', 
                                            'states', 
                                            'parameters', 
                                            'compartments',
                                            'coordinate_components',
                                            'grid_config',
                                            'domains',
                                            'adjacent_domains',
                                            'geometry_definitions',
                                            'functions', 
                                            'variables',
                                            'reactions',
                                            'rates',
                                            'events',
                                            'units',
                                            'meta',
                                            'int_args',
                                            'sim_args',
                                            'opt_args'
                                            ]
                                           )  
        
        #Set up the core data structures
        all_names = self._init_core(ref,
                                    states,
                                    parameters,
                                    functions,
                                    variables,
                                    reactions,
                                    rates,
                                    events,
                                    units
                                    )
        
        self.coordinate_components = CoordinateComponentDict(coordinate_components)
        self.grid_config           = GridConfig(all_names,
                                                None,
                                                self.coordinate_components,
                                                **grid_config
                                                )
        self.compartments          = CompartmentDict(all_names,
                                                     self.states,
                                                     compartments
                                                     )
        self.domains               = DomainDict(all_names, 
                                                self.coordinate_components, 
                                                self.compartments, 
                                                domains
                                                )
        self.adjacent_domains      = AdjacentDomainDict(all_names, 
                                                        self.coordinate_components, 
                                                        self.domains, 
                                                        adjacent_domains
                                                        )
        self.geometry_definitions  = GeometryDefinitionDict(all_names, 
                                                            self.coordinate_components, 
                                                            self.compartments, 
                                                            geometry_definitions
                                                            )
        self.advection             = AdvectionDict(all_names, 
                                                   self.coordinate_components, 
                                                   self.states, 
                                                   self.parameters, 
                                                   advection
                                                   ) if advection else None
        self.diffusion             = DiffusionDict(all_names, 
                                                   self.coordinate_components, 
                                                   self.states, 
                                                   self.parameters, 
                                                   diffusion
                                                   ) if diffusion else None
        self.boundary_conditions   = BoundaryConditionDict(all_names, 
                                                           self.coordinate_components, 
                                                           self.states, parameters, 
                                                           boundary_conditions
                                                           ) if boundary_conditions else None
        
        #Assign attributes
        self.meta         = self.deep_copy('meta', meta)
        self.int_args     = self.deep_copy('int_args', int_args)
        self.sim_args     = self.deep_copy('sim_args', sim_args)
        self.opt_args     = self.deep_copy('opt_args', opt_args)
      
        