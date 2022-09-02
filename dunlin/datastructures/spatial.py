from .boundarycondition import BoundaryConditionDict
from .compartment  import CompartmentDict
from .masstransfer import AdvectionDict, DiffusionDict


from .modeldata  import ModelData


from .ode import ODEModelData
from .geometrydata import GeometryData

class SpatialModelData(ModelData):
    @classmethod
    def from_all_data(cls, all_data, mref, gref, **kwargs):
        model_data    = ODEModelData.from_all_data(all_data, mref)
        geometry_data = GeometryData.from_all_data(all_data, gref)
        
        
        keys = ['compartments', 'advection', 'diffusion', 'boundary_conditions']
        temp = all_data[mref]
        
        for key in keys:
            if key in temp:
                model_data[key] = temp[key]
        
        spatial_data  = cls(model_data, geometry_data, **kwargs)
        
        return spatial_data
        
    def __init__(self,
                 model_data    : ODEModelData,   
                 geometry_data : GeometryData,
                 **kwargs
                 ) -> None:
        
        #Check no overlaps in namespaces
        model_namespace    = model_data['namespace']
        geometry_namespace = geometry_data['namespace']
        
        repeated = model_namespace.intersection(geometry_namespace)
        if repeated:
            msg = f'Redefinition of {repeated} when combining model data and geometry data.'
            raise ValueError(msg)

        namespace = set(model_namespace | geometry_namespace)
        
        #Extend the model with spatial-based items
        for key in ['compartments', 'advection', 'diffusion']:
            if key not in model_data:
                msg = f'Model data is missing {key} information.'
                raise ValueError(msg)
                
        compartments = CompartmentDict(namespace, 
                                       model_data['states'], 
                                       geometry_data['domain_types'],
                                       model_data['compartments']
                                       )
        advection    = AdvectionDict(namespace, 
                                     geometry_data['coordinate_components'], 
                                     model_data['rates'],
                                     model_data['states'], 
                                     model_data['parameters'], 
                                     model_data['advection']
                                     )
        diffusion    = DiffusionDict(namespace, 
                                     geometry_data['coordinate_components'], 
                                     model_data['rates'],
                                     model_data['states'], 
                                     model_data['parameters'], 
                                     model_data['diffusion']
                                     )
        
        model_data['compartments'] = compartments
        model_data['advection']    = advection
        model_data['diffusion']    = diffusion
        
        if 'boundary_conditions' in model_data:
            ccds  = geometry_data['coordinate_components']
            bcs   = BoundaryConditionDict(namespace, 
                                          ccds, 
                                          model_data['states'],
                                          model_data['boundary_conditions']
                                          )
            
            model_data['boundary_conditions'] = bcs
        
        
            
        #Mep the reactions to the compartments
        model_data['reaction_compartments'] = None
        
        #Extract the ref
        ref = model_data['ref'], geometry_data['ref']
        
        super().__init__(ref       = ref,
                         model     = model_data, 
                         geometry  = geometry_data, 
                         namespace = namespace,
                         )
        
        self.update(kwargs)

    def to_data(self, recurse=True) -> dict:
        #Extract model information and export 
        _extend  = ['compartments', 'advection', 'diffusion']
        model    = self['model'].to_data(recurse, _extend=_extend)
        geometry = self['geometry'].to_data(recurse)

        #Merge    
        mref, gref = self['ref']
        dct = {mref : model[mref],
               gref : geometry[gref]
               }
        return dct
   


        
        