from pathlib     import Path

from . import dun_element_reader as der
from . import _base_raw

###############################################################################
#Raw Code Class
###############################################################################
class DunRawCode(_base_raw.RawCode):
    '''
    This class acts as an intermediary between config files and loaded the data.
    
    Config file <-> DunRawCode <-> DunData <-> Model
    '''
    ###########################################################################
    #Instantiators
    ###########################################################################
    @classmethod
    def read_file(cls, filename):
        '''File -> RawCode
        '''
        with open(filename, 'r') as file:
            args = cls._read_lines(file)
        return cls(**args, name=filename)

    @classmethod
    def read_string(cls, string, name=''):
        '''String -> RawCode
        '''
        args = cls._read_lines(string.split('\n'))
        return cls(**args, name='')
    
    @staticmethod
    def _read_lines(lines):
        current_section    = None
        current_subsection = None
        sections           = []
        
        for line in lines:
            line_ = line.strip()
            
            if not line_ or line_[0] == ';':
                continue
            
            elif line[0] == '>':
                raise NotImplementedError()
            
            elif line[0] == '`' and line[1] != '`':
                current_section    = Section(line_[1:].strip())
                sections.append(current_section)
                
            elif line[:2] == '``' and line[2] != '`':
                if current_section is None:
                    raise SubsectionPositionError()
                
                subsection_type    = line_[2:].strip()
                current_subsection = Subsection(subsection_type)
                
                current_section.append(current_subsection)
                
            elif line[:3] == '```':
                raise InvalidMarkerError(line)
            
            elif line[0].isspace():
                current_subsection[-1] += line
            
            else:
                current_subsection.new_element(line)
        
        return {'sections': sections}
    
    def __init__(self, sections=None, name=''):
        self.sections  = []    if sections  is None else list(sections)
        self.name      = Path(name) 
        
        self._check_unique_sections(self.sections)
    
    ###########################################################################
    #Iterators/Checkers
    ###########################################################################
    def __iter__(self):
        return iter(self.sections)
    
    @staticmethod
    def _check_unique_sections(sections):
        unique       = []
        duplicates   = []
        
        for section in sections:
            name = section.name
            if name in unique:
                duplicates.append(name)
            else:
                unique.append(name)
        
        if duplicates:
            raise DuplicateModelKeysError(duplicates)
    
    ###########################################################################
    #Accessors/Modifiers
    ###########################################################################
    def __add__(self, other):
        return self._add(other)
    
    def add(self, *others, new_name=''):
        result = None
        for x in [self, *others]:
            if result is None:
                result = x
            else:
                result = result + x
        
        result.name = new_name
        return result
    
    def _add(self, other, new_name=''):
        if type(other) != type(self):
            raise TypeError(f'Attempted to add {type(self)} with {type(other)}.')
        
        new_sections = self.sections + other.sections
        
        return type(self)(new_sections, new_name)
    
    def get_model_keys(self):
        return [section['model_key'] for section in self.sections]
    
    def __contains__(self, model_key):
        return model_key in self.get_model_keys()
    
    def __len__(self):
        return len(self.sections)
    
    def __getitem__(self, idx):
        return self.sections[idx]
    
    def __setitem__(self, idx, section):
        self.sections[idx] = section
    
    def append(self, section):
        self.sections.append(section)
    
    ###########################################################################
    #Parsers/Writers
    ###########################################################################
    def to_data(self):
        return [section.to_data() for section in self]
        
    def __str__(self):
        sections = '\n\n'.join([str(section) for section in self])
        code     = sections
        
        return code
    
    def __repr__(self):
        code = str(self)
        return type(self).__name__ + f' {self.name} ' + '{\n' + code + '\n}'
    
    def to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(str(self))
    
class InvalidMarkerError(Exception):
    def __init__(self, line):
        super().__init__(f'Invalid marker encountered in: {line}')
        
class ImportPositionError(Exception):
    def __init__(self):
        super().__init__('Import statements must be the top of the file.')

class SubsectionPositionError(Exception):
    def __init__(self, subsection_name):
        super().__init__('Subsection {subsection_name} is not inside any section.')

class DuplicateModelKeysError(Exception):
    def __init__(self, model_keys):
        super().__init__(f'Detected duplicate model keys: {model_keys}')

###############################################################################
#Section Class
###############################################################################
class Section:
    ###########################################################################
    #Instantiators
    ###########################################################################
    def __init__(self, name, subsections=None, section_type='model'):
        self.section_type = section_type
        self.name         = name
        self.subsections  = [] if subsections is None else list(subsections) 
        self._check_unique_subsections()
    
    ###########################################################################
    #Iterators/Checkers
    ###########################################################################
    def __iter__(self):
        return iter(self.subsections)
    
    def _check_unique_subsections(self):
        all_types = self.get_subsection_types()
        unique    = set(all_types)
        
        if len(all_types) != len(unique):
            raise RepeatedSubsectionError(self.model_key)
    
    ###########################################################################
    #Accessors/Modifiers
    ###########################################################################
    def __getitem__(self, idx):
        return self.subsections[idx]
    
    def __setitem__(self, idx, value):
        self.subsections[idx] = value
        self._check_unique_subsections()
    
    def get_subsection_types(self):
        return [subsection.subsection_type for subsection in self]
    
    def __contains__(self, item):
        return item in self.get_subsection_types()
    
    def append(self, subsection):
        self.subsections.append(subsection)
    
    ###########################################################################
    #Parsers/Writers
    ###########################################################################
    def to_data(self):
        data         = {der.subsection_types[subsection.subsection_type][1]: subsection.to_data() for subsection in self}
        if self.section_type == 'model':
            data['model_key'] = self.name
        else:
            raise NotImplementedError('No implementation for {self.section_type} section.')
        return data
    
    def __str__(self):
        if self.section_type == 'model':
            marker = '`'
        else:
            raise NotImplementedError('No marker for {self.section_type} section.')
            
        return f'{marker}{self.name}\n' + '\n'.join([str(subsection) for subsection in self])
    
    def __repr__(self):
        code = str(self)
        return type(self).__name__ + '{\n' + code + '\n}'
    
class RepeatedSubsectionError(Exception):
    def __init__(self, model_key, subsection_type):
        super().__init__(f'Detected duplicates of one or more subsection_types in Section {model_key}.')

###############################################################################
#Subsection Class
###############################################################################
class Subsection:
    ###########################################################################
    #Instantiators
    ###########################################################################
    def __init__(self, subsection_type, elements=None):
        self.subsection_type = subsection_type
        self.elements        = [] if elements is None else list(elements)
        
        self._check_subsection_type(*self.elements)
    
    ###########################################################################
    #Iterators/Checkers
    ###########################################################################
    def __iter__(self):
        return iter(self.elements)
    
    def _check_subsection_type(self, *elements):
        for element in elements:
            if element.subsection_type != self.subsection_type:
                raise SubsectionMismatchError(self.subsection_type)
    
    ###########################################################################
    #Accessors/Modifiers
    ###########################################################################
    def __getitem__(self, idx):
        return self.elements[idx]
    
    def __setitem__(self, idx, value):
        self._check_subsection_type(value)
        self.elements[idx] = value
    
    def append(self, element):
        self._check_subsection_type(element)
        self.elements.append(element)
    
    def new_element(self, string=''):
        new = Element(string, self.subsection_type)
        self.append(new)
    
    ###########################################################################
    #Parsers/Writers
    ###########################################################################
    def to_data(self):
        result = {}
        for element in self:
            result = {**result, **element.to_data()}
        return result
    
    def to_file(self):
        code = str(self)
        with open(self.name, 'w') as file:
            file.write(code)
        
        return code
    
    def __str__(self):
        return f'``{self.subsection_type}\n' + '\n'.join([str(element) for element in self])
    
    def __repr__(self):
        code = str(self)
        return type(self).__name__ + '{\n' + code + '\n}'
    
class SubsectionMismatchError(Exception):
    def __init__(self, expect):
        super().__init__(f'Subsection has inconsistent typing. Expected {expect}')
        
###############################################################################
#Element Class
###############################################################################
class Element:
    ###########################################################################
    #Instantiators
    ###########################################################################
    @classmethod
    def read_data(cls, data):
        raise NotImplementedError()
    
    def __init__(self, string, subsection_type):
        self.string          = string
        self.subsection_type = subsection_type
    
    ###########################################################################
    #Accessors/Modifiers
    ###########################################################################
    def __add__(self, other):
        if type(other) == str:
            new = self.string + other
        
        else:
            new = self.string + ', ' + other.string
            if self.subsection_type != other.subsection_type:
                raise SubsectionAdditionError(self.subsection_type, other.subsection_type)
        
        new = type(self)(new, self.subsection_type)
        return new
    
    def __iadd__(self, other):
        return self + other
    
    ###########################################################################
    #Parsers/Writers
    ###########################################################################
    def to_data(self):
        return der.parse_element(self.string, self.subsection_type)
    
    def __str__(self):
        return self.format_element(self.string)
    
    def __repr__(self):
        code = str(self)
        return type(self).__name__ + '{\n' + code + '\n}'
    
    @staticmethod
    def format_element(element):
        element = element.strip()
        result  = ''
        i0      = 0
        for i, char in enumerate(element):
            if char in '~!':
                result += element[i0: i].strip() + '\n  ' + char
                i0      = i + 1
        
        if i0 == 0:
            return element
        
        if i0 < len(element) - 1:
            result += element[i0:]
        return result
    
class SubsectionAdditionError(Exception):
    def __init__(self, t1, t2):
        super().__init__(f'Attempted to add a {t1} element to a {t2}')
    
###############################################################################
#Exceptions
###############################################################################


###############################################################################
#Collation of Parsers
###############################################################################
raw_code_parsers = {'.dun' : DunRawCode,
                    }
