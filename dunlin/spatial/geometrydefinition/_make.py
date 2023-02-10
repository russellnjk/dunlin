from .csgnode import parse_node

def make_shapes(geometry_definitions):
    #Make shapes
    gdefs   = geometry_definitions
    shapes  = {}
    
    for gdef_name, gdef in gdefs.items():
        definition = gdef.definition if hasattr(gdef, 'definition') else gdef['definition'] 
        
        if definition == 'csg':
            node  = gdef.node        if hasattr(gdef, 'node'       ) else gdef['node']
            order = gdef.order       if hasattr(gdef, 'order'      ) else gdef['order']
            dmnt  = gdef.domain_type if hasattr(gdef, 'domain_type') else gdef['domain_type']
            
            shape             = parse_node(node, gdef_name)
            shape.domain_type = dmnt
            
            shapes[order] = shape
        else:
            raise NotImplementedError(f'No implementation for {gdef.definition} yet.')

    #Sort the shapes
    shapes = [shapes[i] for i in sorted(shapes)]
    
    return shapes