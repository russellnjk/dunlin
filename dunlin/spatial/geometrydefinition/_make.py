from .csg import Square, Circle, Cube, Sphere, Composite

def make_shapes(geometry_definitions) -> tuple:
    #Make shapes
    gdefs   = geometry_definitions
    shapes  = {}
    
    for gdef_name, gdef in gdefs.items():
        try:
            geometry_type = gdef.geometry_type 
        except:
            try: 
                gdef['geometry_type'] 
            except:
                msg = f'Could not determine the geometry type of {gdef}.'
                raise ValueError(msg)
                
        if geometry_type == 'csg':
            node  = gdef.definition  if hasattr(gdef, 'definition' ) else gdef['definition']
            order = gdef.order       if hasattr(gdef, 'order'      ) else gdef['order']
            dmnt  = gdef.domain_type if hasattr(gdef, 'domain_type') else gdef['domain_type']
            
            shape             = parse_csg(node, gdef_name)
            shape.domain_type = dmnt
            
            shapes[order] = shape
        else:
            raise NotImplementedError(f'No implementation for {gdef.geometry_type} yet.')

    #Sort the shapes
    shapes = [shapes[i] for i in sorted(shapes)]
    
    return shapes


###############################################################################
#Parse Definition
###############################################################################
primitives      = {'square' : Square, 
                   'circle' : Circle, 
                   'cube'   : Cube, 
                   'sphere' : Sphere
                   }    
operators       = {'union', 'intersection', 'difference'}
transformations = {'translate', 'rotate', 'scale'}


def parse_csg(node, name=None):
    if type(node) == str:
        if node in primitives:
            new_shape = primitives[node]()
            return new_shape
        else:
            raise ValueError(f'No primitive {node}.')
            
    elif type(node) == list:
        if node[0] in operators:
            #Use the form Composite(op, *shapes)
            op, *shapes_ = node
            shapes       = [parse_csg(s) for s in shapes_]
            new_shape    = Composite(op, *shapes)
            
            new_shape.name = name
            return new_shape
        
        elif node[0] in transformations:
            *transformation_args, shape = node
           
            new_shape = parse_csg(shape)
            new_shape = new_shape(*transformation_args)
           
            new_shape.name = name
            return new_shape
           
    else:
        msg = f'Node must be a str or list. Received {type(node).__name__}.'
        raise TypeError(msg)
