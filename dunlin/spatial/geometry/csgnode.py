from .csg import Square, Circle, Cube, Sphere, Composite

###############################################################################
#Parse Definition
###############################################################################
primitives = {'square': Square, 'circle': Circle, 
              'cube': Cube, 'sphere': Sphere
              }    
operators = ['union', 'intersection', 'difference']

def parse_node(node, name=None, domain_type=None):
    if type(node) == str:
        if node in primitives:
            new_shape = primitives[node]()
            return new_shape
        else:
            raise ValueError('No primitive {node}.')
            
    elif type(node) == list:
        if node[0] in operators:
            #Expect Composite(op, shapes)
            op, *shapes_ = node
            shapes       = [parse_node(s) for s in shapes_]
            new_shape    = Composite(op, *shapes)
            
            new_shape.name        = name
            new_shape.domain_type = domain_type
            return new_shape
        
        else:
            #Expect piping i.e. [shape, *transformations]
            new_shape = parse_node(node[0])
            for transformation in node[1:]:
                if type(transformation) == dict:
                    new_shape = new_shape(**transformation)
                elif type(transformation) in [list, tuple]:
                    new_shape = new_shape(*transformation)
                else:
                    msg = 'Expected list, tuple or dict.'
                    msg = f'{msg} Received {type(transformation).__name__}'
                    raise TypeError(msg)
            
            new_shape.name        = name
            new_shape.domain_type = domain_type
            return new_shape
        
    else:
        msg = f'Node must be a str, list. Received {type(node).__name__}.'
        raise TypeError(msg)
