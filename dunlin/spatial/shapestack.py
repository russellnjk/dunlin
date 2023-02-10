from .grid.stack         import Stack
from .grid.grid          import make_grids_from_config
from .geometrydefinition import make_shapes

from dunlin.datastructures import SpatialModelData
                                  
class ShapeStack(Stack):
    def __init__(self, spatial_data: SpatialModelData):
        #Generate shapes and grids
        shapes    = make_shapes(spatial_data['geometry_definitions'])
        grids     = make_grids_from_config(spatial_data['grid_config'])
        main_grid = next(iter(grids.values()))
        
        #Save attributes
        self.spatial_data = spatial_data
        self.grids        = grids
        
        #Create auxillary mappings
        temp = self._map_shape_2_domain(spatial_data, *shapes)
        
        self.shape2domain      = temp[0]
        self.domain2shape      = temp[1]
        self.shape2domain_type = temp[2]
        self.domain_type2shape = temp[3]
        
        #Master mapping
        self.n_elements  = 0
        self.voxel2all   = {}
        self.element2all = {} 
        super().__init__(main_grid, *shapes)
        
    @staticmethod
    def _map_shape_2_domain(spatial_data, *shapes) -> tuple:
        domain_types       = spatial_data.domain_types
        shape2domain       = {}
        domain2shape       = {}
        shape2domain_type  = {}
        domain_type2shape  = {}
        
        for shape in shapes:
            shape_name  = shape.name
            domain_type = shape.domain_type
            domains     = domain_types[domain_type]
            
            found = False
            for domain, internal_point in domains.items():
                if internal_point in shape:
                    found = True
                    break
            
            if found:
                if domain in domain2shape:
                    msg = f'Domain {domain} appears to belong to more than one shape.'
                else:
                    shape2domain[shape_name]      = domain
                    domain2shape[domain]          = shape_name
                    shape2domain_type[shape_name] = domain_type
                    domain_type2shape.setdefault(domain_type, set()).add(shape_name)
            else:
                msg = f'Could not assign shape {shape.name} to a domain.'
                raise ValueError(msg)
        
        
        return shape2domain, domain2shape, shape2domain_type, domain_type2shape 
    
    @staticmethod
    def _get_state_boundary_conditions(state            : str, 
                                       voxel_boundaries : list[int], 
                                       spatial_data     : SpatialModelData
                                       ) -> dict:
        
        boundary_conditions = spatial_data.boundary_conditions
        x_bcs = {}
        for boundary in voxel_boundaries:
            bc = boundary_conditions.find(state = state,
                                          axis  = boundary
                                          )
            
            if bc:
                x_bcs[boundary] = {'condition'      : bc.condition,
                                   'condition_type' : bc.condition_type
                                   }
        
        return x_bcs

    @staticmethod
    def _get_advection_term(voxel            : tuple,
                            state            : str, 
                            voxel_boundaries : list[int],
                            voxel_edges      : dict[int, dict],
                            shifts           : list[int],
                            spatial_data     : SpatialModelData
                            ) -> dict:
        
        adv      = spatial_data.advection.get(state)
        adv_term = {}
        
        if adv:
            for shift in shifts:
                if shift in voxel_boundaries:
                    continue
                elif shift in voxel_edges:
                    continue
                adv_term[shift] = {'coeff' : adv[shift]}

        return adv_term
    
    @staticmethod
    def _get_diffusion_term(voxel            : tuple,
                            state            : str, 
                            voxel_boundaries : list[int],
                            voxel_edges      : dict[int, dict],
                            shifts           : list[int],
                            spatial_data     : SpatialModelData
                            ) -> dict:
        
        dfn      = spatial_data.diffusion.get(state)
        dfn_term = {}
        
        if dfn:
            for shift in shifts:
                if shift in voxel_boundaries:
                    continue
                elif shift in voxel_edges:
                    continue
                dfn_term[shift] = {'coeff' : dfn[shift]}

        return dfn_term
    
    @staticmethod
    def _get_state_diffusion(state            : str, 
                             voxel_boundaries : list[int],
                             voxel_edges, 
                             spatial_data     : SpatialModelData
                             ) -> dict:
        
        ndims = spatial_data.coordinate_components.ndims
        dfn   = spatial_data.diffusion.get(state)
        
        if dfn:
            x_dfn = {n: 0 if n in voxel_boundaries else dfn[n] for n in range(1, ndims+1)}
        else:
            x_dfn = {n: 0 for n in range(1, ndims+1)}
        
        return x_dfn
    
    def _add_voxel(self, voxel: tuple) -> None:
        super()._add_voxel(voxel)
        
        spatial_data = self.spatial_data
        compartments = spatial_data.compartments
        
        #Extract information
        shape_name       = self.voxel2shape[voxel]
        domain_type      = self.shapes[shape_name].domain_type
        domain           = self.shape2domain[shape_name]
        compartment      = compartments.domain_type2compartment[domain_type]
        states           = compartments.domain_type2state[domain_type]
        voxel_boundaries = self.boundaries.get(voxel, [])
        voxel_edges      = self.voxel2edge.get(voxel, {})
        size             = self.sizes[voxel]
        
        template = {'shape'       : shape_name,
                    'domain_type' : domain_type,
                    'compartment' : compartment,
                    'domain'      : domain,
                    'states'      : {},
                    'boundaries'  : {},
                    'edges'       : {},
                    'elements'    : {}
                    }
      
        for x in states:
            #Determine index of associated array element
            idx = self.n_elements
            
            #Find boundary conditions
            x_bcs = self._get_state_boundary_conditions(x, 
                                                        voxel_boundaries, 
                                                        spatial_data
                                                        )
            
            #Find advection
            x_adv = self._get_state_advection(x, 
                                              voxel_boundaries, 
                                              voxel_edges,
                                              spatial_data
                                              )
            
            
            #Find diffusion
            x_dfn = self._get_state_diffusion(x, 
                                              voxel_boundaries, 
                                              voxel_edges,
                                              spatial_data
                                              )
                
            #Update self.element2all
            self.element2all[idx] = {'shape'       : shape_name,
                                     'domain_type' : domain_type,
                                     'compartment' : compartment,
                                     'domain'      : domain,
                                     'state'       : x,
                                     'boundaries'  : x_bcs,
                                     'advection'   : x_adv,
                                     'diffusion'   : x_dfn,
                                     'size'        : size,
                                     'voxel'       : voxel
                                     }
            
            #Update the template for voxel2all
            template['boundaries'][x] = x_bcs
            template['states'][x]     = idx 
            template['elements'][idx] = x
            
            #Increase the value of n_elements
            self.n_elements += 1
            
        #Save the information
        self.voxel2all[voxel] = template
    
    # @staticmethod
    # def _make_equations(stack, masstransfer):
    #     voxels      = stack.voxels
    #     voxel2shape = stack.voxel2shape
    #     shape_dict  = stack.shape_dict
        
    #     domain_type2state = masstransfer.domain_type2state
        
        
    #     for voxel_id, voxel in enumerate(voxels):
    #         shape_name = voxel2shape[voxel]
    #         shape      = shape_dict[shape_name]
    #         dmnt       = shape.domain_type
            
    #         bulk_rxn_dct = masstransfer.make_bulk_reaction_dict(voxel_id, dmnt)
    #         rt_dct       = masstransfer.make_rate_dict(voxel_id, dmnt)
            
            
    #         if _at_boundary:
    #             bndy_rxn_dct = masstransfer.make_boundary_reaction_dict(voxel_id, dmnt)
    #             bc_dct       = masstransfer.make_boundary_condition_dict(voxel_id, )
    #         else:
    #             adv_dct = masstransfer.make_advection_dict
    #             dfn_dct = masstransfer.make_diffusion_dict
                