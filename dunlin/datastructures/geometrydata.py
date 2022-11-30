import dunlin.utils             as ut
from .coordinatecomponent import CoordinateComponentDict
from .gridconfig          import GridConfigDict
from .domaintype          import DomainTypeDict
from .adjacentdomain      import AdjacentDomainsDict
from .geometrydefinition  import GeometryDefinitionDict
from .boundarycondition   import BoundaryConditionDict 
from .unit                import UnitsDict    
from .modeldata  import ModelData

class GeometryData(ModelData):
    '''Represents geometry for spatial modeling. 
    '''

    ###########################################################################
    #Instantiation
    ###########################################################################
    # @classmethod
    # def from_all_data(cls, all_data, ref, **kwargs):
    #     keys = ['coordinate_components',
    #             'grid_config',
    #             'domain_types',
    #             'adjacent_domains',
    #             'geometry_definitions',
    #             'boundary_conditions',
    #             'units'
    #             ]
    #     args = {}
    #     temp = all_data[ref]
    #     for key in keys:
    #         if key in temp:
    #             args[key] = temp[key]
        
    #     geometry_data = cls(ref, **args, **kwargs)
    #     return geometry_data
        
    def __init__(self, 
                 ref: str,
                 coordinate_components: dict,
                 grid_config: dict,
                 domain_types: dict,
                 adjacent_domains: dict,
                 geometry_definitions: dict,
                 boundary_conditions: dict=None,
                 **kwargs
                 ):
        
        #Set up the data structures
        namespace = set()
        
        ccds  = CoordinateComponentDict(coordinate_components)
        gcfg  = GridConfigDict(namespace, ccds, grid_config)
        dmnts = DomainTypeDict(namespace, ccds, domain_types)
        admns = AdjacentDomainsDict(namespace, ccds, dmnts, adjacent_domains)
        gdefs = GeometryDefinitionDict(namespace, ccds, dmnts, geometry_definitions)
        
        self.ref                   = ref
        self.coordinate_components = ccds
        self.grid_config           = gcfg
        self.domain_types          = dmnts
        self.geometry_definitions  = gdefs
        
        if adjacent_domains:
            self.adjacent_domains = admns
        else:
            self.adjacent_domains = None
        
        #Freeze the namespace and save it
        self.namespace = frozenset(namespace)
        
        #Collate region_types
        mapping = {}
        mapping.update(dict.fromkeys(gcfg,  'grid_config'))
        mapping.update(dict.fromkeys(dmnts, 'domain_types'))
        mapping.update(dict.fromkeys(gdefs, 'geometry_definitions'))
        
        self.region_types = mapping
        
        #Freeze the attributes
        self.freeze()
        
    def to_data(self, recurse=True) -> dict:
        keys = ['coordinate_components',
                'grid_config',
                'domain_types',
                'adjacent_domains',
                'geometry_definitions',
                'boundary_conditions',
                ]
        return self._to_data(keys, recurse)
        