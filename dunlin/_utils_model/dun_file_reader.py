import numpy as np
import re
from   pathlib import Path

###############################################################################
#Non-Standard Imports
###############################################################################
try:
    import dunlin._utils_model.dun_element_reader as der
    from  .base_error  import DunlinBaseError
    from  .custom_eval import safe_eval as eval
except Exception as e:
    if Path.cwd() == Path(__file__).parent:
        import dun_element_reader as der
        from  base_error  import DunlinBaseError
        from  custom_eval  import safe_eval  as eval
    else:
        raise e
    
###############################################################################
#Front-End Functions
###############################################################################
def read_file(filename, _parse=True):
    
    with open(filename, 'r') as file:
        sections = get_sections(file, filename)
        if _parse:
            return parse_sections(sections)
        else:
            return sections

def get_sections(main_code, filename=''):
    def open_file(filename):
        if str(filename)[-4:] == '.dun':
            with open(filename, 'r') as file:
                sections, imports = split_sections(file)
        else:
            raise NotImplementedError()
        return sections, imports 
            
    files             = []
    sections, imports = split_sections(main_code)
    section_names     = set(sections)
    
    while imports:
        new_imports = []
        for f in imports:
            abs_p = Path(f)
            rel_p = Path(filename).parent / f
            
            if rel_p.exists():
                p = rel_p
            elif abs_p.exists() and abs_p.is_absolute():
                p = abs_p
            else:
                raise DunlinConfigError.import_path(f'{rel_p}\nor\n{abs_p}')
            
            if p in files or p == filename:
                continue
            else:
                files.append(p)

            sections_, imports_ = open_file(p)
            
            if section_names.intersection(sections_):
                raise DunlinConfigError.repeat('one or more models during import.')
                
            sections = {**sections_, **sections}
            new_imports.extend(imports_)
        
        imports = new_imports
        
    return sections
    
def split_sections(code):
    sections             = {}
    curr_section         = None
    curr_section_name    = None
    curr_subsection_name = None
    chunks               = None
    imports              = []
    to_iter              = code.split('\n') if type(code) == str else code
    
    for line in to_iter:
        line_ = line.strip()
        
        if not line_:
            continue
        
        elif line[:6] == 'import':
            if curr_section is not None:
                raise DunlinConfigError.import_position()
            
            to_import = parse_import(line)
            imports.append(to_import)
            
        elif line_[0] == ';':
            continue
        
        elif line[:3] == '```':
            raise DunlinConfigError.depth('max')
            
        elif line[:2] == '``':
            new_subsection_name = line_[2:len(line_)].strip()
            arg                 = der.parsers.get(new_subsection_name, [None, None])[1]
            if not arg:
                raise DunlinConfigError.subsection_type(new_subsection_name)
            else:
                new_subsection_name = arg
                
            if new_subsection_name == curr_subsection_name:
                raise DunlinConfigError.repeat(new_subsection_name)
        
            #Update
            chunks               = curr_section.setdefault(new_subsection_name, [])
            curr_subsection_name = new_subsection_name
            
        elif line[0] == '`':
            new_section_name = line_[1:len(line_)].strip()
            if new_section_name == curr_section_name:
                raise DunlinConfigError.repeat(new_section_name)
                
            #Update
            curr_section      = sections.setdefault(new_section_name, {'model_key': new_section_name})
            curr_section_name = new_section_name
            
        elif line[0].isspace():
            chunks[-1] = chunks[-1] + line
            
        else:
            if chunks is None:
                raise ValueError(f'This line does not belong to any subsection: {line_}')
            chunks.append(line)
            
    return sections, imports

def parse_import(import_statement):
    try:
        _, to_import = import_statement.split('import ')
        to_import    = to_import.strip()
    except:
        raise DunlinConfigError()
    
    return to_import

def parse_sections(sections):
    dun_data = {}
    for section_name, section_code in sections.items():
        if section_name[0] == '_':
            continue

        dun_data[section_name] = {}
        dun_data[section_name] = parse_section(section_code)
    return dun_data

def parse_section(section):
    parsed_section = {}
    
    for subsection_type, elements in section.items():
        if subsection_type[0] == '_':
            continue
        elif subsection_type == 'model_key':
            parsed_section[subsection_type] = elements
            continue
        
        func, arg           = der.parsers[subsection_type]
        parsed_section[arg] = {}
        
        for element in elements:
            parsed = func(element)
            parsed_section[arg].update(parsed)
      
    return parsed_section

##############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinConfigError(DunlinBaseError):
    @classmethod
    def import_position(cls):
        return cls.raise_template('Import statement not placed at top of file.', 0)
    
    @classmethod
    def import_path(cls, path):
        return cls.raise_template(f'Could not find a file path(s):\n{path}\nUse an absolute path if the file(s) to be imported is not in the same folder as the main file.', 1)
    
    @classmethod
    def import_format(cls, path):
        return cls.raise_template(f'Cannot read file due to unsupported extension: {path}.', 2)
    
    @classmethod
    def repeat(cls, key):
        return cls.raise_template(f'Repeated definition of {key}', 10)
    
    @classmethod
    def subsection_type(cls, subsection_type):
        return cls.raise_template(f'No parser for this subsection type: {subsection_type}', 11)
    
if __name__ == '__main__':
    #Test section splitting
    dun_data = read_file('dun_test_files/M1.dun', False)
    answer   = {'M1': {'states'   : ['x0 : [1, 1]\n', 'x1 : [0, 0]\n', 'x2 : [0, 0]\n', 'x3 : [0, 0]\n'], 
                       'params'   : ['p0 : [0.01, 0.01]\n', 'p1 : [0.01, 0.01]\n', 'p2 : [0.01, 0.01]\n', 'p3 : [0.01, 0.01]\n', 'k2 : [0.5, 0.5]\n'], 
                       'rxns'     : ['r0 : x0 > x1, p0\n', 'r1 : x1 > x2, p1, MM(p2, x2, k2)\n'], 
                       'vrbs'     : ['sat : x2/(x2+k2)\n'], 
                       'rts'      : ['x3 : x0*p3\n'], 
                       'funcs'    : ['MM(v, x, k): v*x/(x+k)\n'], 
                       'events'   : ['e0: [trigger: x0 < 0.2, assignment: x0 = 1]\n'], 
                       'exvs'     : ["sat =\n\n\treturn {'x': t, 'y': sat}\n"], 
                       'int_args' : ['method = LSODA\n']
                       }
                }
    
    assert 'M1' in dun_data
    for key, value in answer['M1'].items():
        assert len(dun_data['M1'][key]) == len(value)
        
        for a, b in zip(dun_data['M1'][key], value):
            a.strip() == b.strip()
    
    #Test section splitting with imports
    dun_data = read_file('dun_test_files/M2.dun', False)
    answer   = {'M1': {'states'    : ['x0 : [1, 1]\n', 'x1 : [0, 0]\n', 'x2 : [0, 0]\n', 'x3 : [0, 0]\n'], 
                       'params'    : ['p0 : [0.01, 0.01]\n', 'p1 : [0.01, 0.01]\n', 'p2 : [0.01, 0.01]\n', 'p3 : [0.01, 0.01]\n', 'k2 : [0.5, 0.5]\n'], 
                       'rxns'      : ['r0 : [x0 > x1, p0]\n', 'r1 : [x1 > x2, p1, MM(p2, x2, k2)]\n'], 
                       'vrbs'      : ['sat : x2/(x2+k2)\n'], 
                       'rts'       : ['x3 : x0*p3\n'], 
                       'funcs'     : ['MM(v, x, k): v*x/(x+k)\n'], 
                       'events'    : ['e0: [trigger: x0 < 0.2, assignment: x0 = 1]\n'], 
                       'exvs'      : ["sat :\n\n\treturn {'x': t, 'y': sat}\n"], 
                       'int_args'  : ['method : LSODA\n']
                       }, 
                'M2': {'states'    : ['xx0 : [0]\n', 'xx1 : [0]\n'], 
                       'params'    : ['pp0 : [0.01, 0.01]\n', 'pp1 : [0.01, 0.01]\n', 'kk1 : [0.5, 0.5]\n'], 
                       'rxns'      : ['r0 : [xx0 > xx1, pp0, pp1*xx2/(xx2 + kk1)]']
                       }
                }
    assert 'M1' in dun_data
    assert 'M2' in dun_data
    for key, value in answer['M1'].items():
        assert len(dun_data['M1'][key]) == len(value)
        
        for a, b in zip(dun_data['M1'][key], value):
            assert a.strip() == b.strip()
    for key, value in answer['M2'].items():
        assert len(dun_data['M2'][key]) == len(value)
        
        for a, b in zip(dun_data['M2'][key], value):
            print(a, b)
            assert a.strip() == b.strip()
    
    #Repeated model deifinition in file
    try:
        read_file('dun_test_files/M3_error.dun', False)
    except DunlinConfigError as e:
        assert e.num == 10
    except Exception as e:
        raise e
        
    #Repeated model definition in imported file
    try:
        read_file('dun_test_files/M3_error.dun', False)
    except DunlinConfigError as e:
        assert e.num == 10
    except Exception as e:
        raise e
    
    #Repeated model definition in when combined with imported file
    try:
        read_file('dun_test_files/M4_error.dun', False)
    except DunlinConfigError as e:
        assert e.num == 10
    except Exception as e:
        raise e
    
    #Repeated subsection definition 
    try:
        read_file('dun_test_files/M5_error.dun', False)
    except DunlinConfigError as e:
        assert e.num == 10
    except Exception as e:
        raise e
    
    #Subsection without parser 
    try:
        read_file('dun_test_files/M6_error.dun', False)
    except DunlinConfigError as e:
        assert e.num == 10
    except Exception as e:
        raise e
        
    #Test with full parsing
    dun_data = read_file('dun_test_files/M1.dun', True)
    answer   = {'M1': {'states'   : {'x0': [1.0, 1.0],
                                     'x1': [0.0, 0.0],
                                     'x2': [0.0, 0.0],
                                     'x3': [0.0, 0.0]},
                       'params'   : {'p0': [0.01, 0.01],
                                     'p1': [0.01, 0.01],
                                     'p2': [0.01, 0.01],
                                     'p3': [0.01, 0.01],
                                     'k2': [0.5, 0.5]},
                       'rxns'     : {'r0': ['x0 > x1', 'p0'], 'r1': ['x1 > x2', 'p1', 'MM(p2, x2, k2)']},
                       'vrbs'     : {'sat': 'x2/(x2+k2)'},
                       'rts'      : {'x3': 'x0*p3'},
                       'funcs'    : {'MM(v, x, k)': 'v*x/(x+k)'},
                       'events'   : {'e0': {'trigger': 'x0 < 0.2', 'assignment': 'x0 = 1'}},
                       'exvs'     : {'sat': "\treturn {'x': t, 'y': sat}"},
                       'int_args' : {'method': 'LSODA'}
                       }
                }
    assert 'M1' in dun_data
    for key, value in answer['M1'].items():
        assert len(dun_data['M1'][key]) == len(value)
        
        for a, b in zip(dun_data['M1'][key], value):
            if type(a) == str and type(b) == str:
                a.strip() == b.strip()
            