import pandas as pd
from typing import Union

import dunlin.comp as cmp
# from .reaction            import ReactionDict
from .variable            import VariableDict
from .function            import FunctionDict
from .rate                import RateDict
from .extra               import ExtraDict
from .event               import EventDict 
from .unit                import UnitsDict

from .stateparam          import StateDict, ParameterDict
from .spatialreaction     import SpatialReactionDict
from .boundarycondition   import BoundaryConditionDict
from .compartment         import CompartmentDict
from .masstransfer        import AdvectionDict, DiffusionDict

from .coordinatecomponent import CoordinateComponentDict
from .gridconfig          import GridConfigDict
from .domaintype          import DomainTypeDict
from .adjacentdomain      import AdjacentDomainDict
from .geometrydefinition  import GeometryDefinitionDict

from .modeldata    import ModelData
# from .ode          import ODEModelData
# from .geometrydata import GeometryData

class SpatialModelData(ModelData):
    @classmethod
    def from_all_data(cls, all_data, ref):
        flattened  = cmp.flatten_ode(all_data, ref)
        
        return cls(**flattened)
    
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
                 extra: Union[dict, callable] = None,
                 units: dict = None,
                 advection: dict = None,
                 diffusion: dict = None,
                 boundary_conditions: dict = None
                 ) -> None:
        
        #kwargs are ignored
        #Set up the data structures
        namespace = set()
        
        #Instantiate things that are same as ODEModel
        xs, ps, funcs, vrbs, rts, evs, extra, units = self._init_basic(namespace, 
                                                                       states, 
                                                                       parameters, 
                                                                       compartments
                                                                       )
        
        #Instantiate geometry data structures
        ccds, gcfg, dmnts, admns, gdefs, gunits = self._init_geometry(namespace,
                                                                      **geometry
                                                                      )
        
        #Instantiate mass transfer structures
        rxns, cmpts, advs, dfns, bcs = self._init_masstransfer(namespace, 
                                                               xs, 
                                                               ps, 
                                                               rts, 
                                                               dmnts, 
                                                               ccds, 
                                                               compartments, 
                                                               reactions, 
                                                               advection, 
                                                               diffusion, 
                                                               boundary_conditions
                                                               )
        
        #Assign attributes
        self.ref        = ref
        self.states     = xs
        self.parameters = ps
        self.functions  = funcs
        self.variables  = vrbs
        self.rates      = rts
        self.events     = evs
        self.extra      = extra
        self.units      = units
        
        self.coordinate_components = ccds
        self.grid_config           = gcfg
        self.domain_types          = dmnts
        self.geometry_definitions  = gdefs
        self.adjacent_domains      = admns  
        self.geometry_units        = gunits
        
        self.reactions           = rxns
        self.compartments        = cmpts
        self.advection           = advs
        self.diffusion           = dfns
        self.boundary_conditions = bcs
        
        #Check no overlap between rates and reactions
        if self.rates and self.reactions:
            rate_xs  = set(self.rates.states)
            rxn_xs   = self.reactions.states
            repeated = rate_xs.intersection(rxn_xs)
        
            if repeated:
                msg = f'The following states appear in both a reaction and rate: {repeated}.'
                raise ValueError(msg)
       
        #Map the reactions to the compartments
        self.reaction_compartments = None
        
        #Freeze
        self.namespace = frozenset(namespace)
        self.freeze()
        
    @staticmethod
    def _init_basic(namespace: set,
                    states: Union[dict, pd.DataFrame], 
                    parameters: Union[dict, pd.DataFrame], 
                    compartments: dict,
                    functions: dict = None, 
                    variables: dict = None, 
                    rates: dict = None, 
                    events: dict = None, 
                    extra: Union[dict, callable] = None,
                    units: dict = None,
                    ) -> tuple:
        
        xs     = StateDict(namespace, states)
        ps     = ParameterDict(namespace, parameters)
        funcs  = FunctionDict(namespace, functions)
        vrbs   = VariableDict(namespace, variables)
        rts    = RateDict(namespace, rates)
        evs    = EventDict(namespace, events)
        extra  = ExtraDict(namespace, extra)
        units  = UnitsDict(namespace, units)
        
        return xs, ps, funcs, vrbs, rts, evs, extra, units
    
    @staticmethod
    def _init_geometry(namespace: set, 
                       coordinate_components: dict,
                       grid_config: dict,
                       domain_types: dict,
                       adjacent_domains: dict,
                       geometry_definitions: dict,
                       units: dict = None
                       ) -> tuple:
        
        ccds   = CoordinateComponentDict(coordinate_components)
        gcfg   = GridConfigDict(namespace, ccds, grid_config)
        dmnts  = DomainTypeDict(namespace, ccds, domain_types)
        admns  = AdjacentDomainDict(namespace, ccds, dmnts, adjacent_domains)
        gdefs  = GeometryDefinitionDict(namespace, ccds, dmnts, geometry_definitions)
        gunits = UnitsDict(namespace, units)
        
        return ccds, gcfg, dmnts, admns, gdefs, gunits
        
    @staticmethod
    def _init_masstransfer(namespace: set,
                           states: StateDict,
                           parameters: ParameterDict,
                           rates: RateDict,
                           domain_types: DomainTypeDict,
                           coordinate_components: CoordinateComponentDict,
                           compartments: dict,
                           reactions: dict,
                           advection: dict = None,
                           diffusion: dict = None,
                           boundary_conditions: dict = None,
                           ) -> tuple:
        
        
        cmpts = CompartmentDict(namespace, 
                                states, 
                                domain_types,
                                compartments
                                )
        advs    = AdvectionDict(namespace, 
                                coordinate_components,                                  
                                rates,
                                states, 
                                parameters, 
                                advection
                                )
        dfns    = DiffusionDict(namespace, 
                                     coordinate_components, 
                                     rates,
                                     states, 
                                     parameters, 
                                     diffusion
                                     )

        bcs = BoundaryConditionDict(namespace, 
                                    coordinate_components, 
                                    states,
                                    boundary_conditions
                                    )
        
        rxns   = SpatialReactionDict(namespace, 
                                     cmpts,
                                     reactions
                                     )
        
        return rxns, cmpts, advs, dfns, bcs
        
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
                'geometry_units'
                ]
        
        data1 = self._to_data(keys, recurse)
        
        data0[self.ref]['geometry'] = data1[self.ref]
        
        return data0
    
        
        