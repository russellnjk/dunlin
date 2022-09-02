def add_coordinate(name, coordinates):
    s = [str(i).replace('.', '_') for i in coordinates]
    s = '__'.join(s)
    return name + '__' + s