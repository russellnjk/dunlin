from .csg import Square, Circle, Cube, Sphere, Composite

###############################################################################
#Parse Definition
###############################################################################
primitives      = {'square': Square, 
                   'circle': Circle, 
                   'cube': Cube, 
                   'sphere': Sphere
                   }    
operators       = {'union', 'intersection', 'difference'}
transformations = {'translate', 'rotate', 'scale'}


def parse_node(node, name=None):
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
            shapes       = [parse_node(s) for s in shapes_]
            new_shape    = Composite(op, *shapes)
            
            new_shape.name = name
            return new_shape
        
        elif node[0] in transformations:
            *transformation_args, shape = node
           
            new_shape = parse_node(shape)
            new_shape = new_shape(*transformation_args)
           
            new_shape.name = name
            return new_shape
           
    else:
        msg = f'Node must be a str or list. Received {type(node).__name__}.'
        raise TypeError(msg)
