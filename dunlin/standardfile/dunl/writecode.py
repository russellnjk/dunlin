from typing import Callable

from . import writeelement as we

###############################################################################
#Code Generation
###############################################################################
def write_dunl_code(dct      : dict, 
                    n_format : Callable  = str,
                    _dir     : list[str] = (),
                    ) -> str:
        
    chunks = []
    for key, value in dct.items():
        if hasattr(value, 'to_dunl_elements'):
            directory     = [*_dir, key]
            directory_code = write_directory(directory)
            
            body  = value.to_dunl_elements(n_format=n_format)
            chunk = f'{directory_code}\n{body}'
            
        elif isinstance(value, dict):
            directory     = [*_dir, key]
            directory_code = write_directory(directory)
            
            body  = write_dunl_code(value, n_format, directory)
            chunk = f'{directory_code}\n{body}'
        
        elif not _dir:
            raise ValueError('Insufficient nesting.')
            
        else:
            chunk = we.write_dict({key: value}, 
                                  multiline_dict = False, 
                                  n_format       = n_format
                                  )
        
        chunks.append(chunk)
        
    code = '\n\n'.join(chunks)
    return code
    
def write_directory(directory: list):
    return ''.join([f';{x}' for x in directory])