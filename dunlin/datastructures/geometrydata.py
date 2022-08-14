import dunlin.utils             as ut
import dunlin.standardfile.dunl as sfd
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
    def __init__(self, 
                 ref: str,
                 coordinate_components: dict,
                 grid_config: dict,
                 domain_types: dict,
                 adjacent_domains: dict,
                 geometry_definitions: dict,
                 boundary_conditions: dict=None,
                 units: dict=None,
                 **kwargs
                 ):
        
        #Set up the data structures
        namespace = set()
        
        ccds  = CoordinateComponentDict(coordinate_components)
        gcfg  = GridConfigDict(namespace, ccds, grid_config)
        dmnts = DomainTypeDict(namespace, ccds, domain_types)
        admns = AdjacentDomainsDict(namespace, ccds, dmnts, adjacent_domains)
        gdefs = GeometryDefinitionDict(namespace, ccds, dmnts, geometry_definitions)
        
        geometry_data = dict(ref                   = ref,
                             coordinate_components = ccds,
                             grid_config           = gcfg,
                             domain_types          = dmnts,
                             adjacent_domains      = admns,
                             geometry_definitions  = gdefs,
                             )
        
        if boundary_conditions:
            bcs   = BoundaryConditionDict(namespace, dmnts, boundary_conditions)
            geometry_data['boundary_conditions'] = bcs
        
        if units:
            geometry_data['units'] = UnitsDict(namespace, units)
        
        #Freeze the namespace
        geometry_data['namespace'] = frozenset(namespace)
        
        #Map in the namespaces that correspond to regions in space
        mapping = {}
        mapping.update(dict.fromkeys(gcfg,  'grid_config'))
        mapping.update(dict.fromkeys(dmnts, 'domain_types'))
        mapping.update(dict.fromkeys(gdefs, 'geometry_definitions'))
        
        geometry_data['region_types'] = mapping
        
        #Add in the remaining kwargs as-is
        geometry_data.update(**kwargs)
        
        #Call the parent constructor
        super().__init__(geometry_data)
    
    def to_data(self) -> dict:
        keys = ['ref', 
                'coordinate_components',
                'grid_config',
                'domain_types',
                'adjacent_domains',
                'geometry_definitions',
                'boundary_conditions',
                'units'
                ]
        
        return super().to_data(keys)
    