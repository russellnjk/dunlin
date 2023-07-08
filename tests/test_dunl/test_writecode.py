import addpath
from writecode    import write_dunl_code
from writeelement import write_dict

dct = {'a': {0: 1},
       'b': {2: [3, 4, 5],
             6: {7: 8,
                 9:10
                 }
             }
       }

r = write_dunl_code(dct)
print(r)

assert r == ';a\n0 : 1\n\n;b\n2 : [3, 4, 5]\n\n;b;6\n7 : 8\n\n9 : 10'

class X:
    def to_dunl_elements(self, **kwargs) -> str:
        return write_dict({2: 3})

dct = {'a': {0: 1},
        'b': X()
        }

r = write_dunl_code(dct)
print(r)
assert r == ';a\n0 : 1\n\n;b\n2 : 3'
