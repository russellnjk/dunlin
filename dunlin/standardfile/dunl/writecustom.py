from numbers import Number

import dunlin.standardfile.dunl.writedictlist as wdl

'''
These functions are not part of the default .dunl code generation algorithm. 
They are instead meant to be used as part of user-defined functions for 
custom formatting.
'''

###############################################################################
#Formatting DataFrames
###############################################################################
def format_num(x: Number) -> str:
    if int(x) == float(x):
        return str(int(x))
    
    if 0.1 <= x <= 1e3:
        return '{:.5}'.format(x)
    
    s = '{:.4e}'.format(x)
    s = s.replace('e+', 'e')
    s = s.replace('e0', 'e')
    s = s.replace('e0', '')
    
    return s

def write_numeric_df(df, n_format: callable = format_num) -> str:
    if all(df.index == list(range(len(df.index)))):
        return write_numeric_df_no_index(df, n_format)
        
    max_len_col = 0
    max_len_idx = 0
    columns     = []
    indices     = []
    
    #Extract column values
    for col in df.columns:
        if type(col) == tuple:
            col = wdl.write_list(col)
        else:
            col = wdl.write_primitive(col)
    
        max_len_col = max(len(col), max_len_col)
        columns.append(col)
        
    for idx in df.index:
        if type(idx) == tuple:
            idx = wdl.write_list(idx)
        else:
            idx = wdl.write_primitive(idx)
        
        max_len_idx = max(len(idx), max_len_idx)
        indices.append(idx)
    
    code = ''
    for name, col in zip(columns, df.columns):
        array = df[col]
        line  = name + ' '*(max_len_col-len(col) + 1) + ' : ['
        for subname, value in zip(indices, array):
            lhs   = subname + ' '*(max_len_idx-len(idx) + 1)
            rhs   = n_format(value) + ', ' 
            line += lhs + ' : ' + rhs
        line  = line[:-2] + ']'
        code += line + '\n'
    
    return code

def write_numeric_df_no_index(df, n_format: callable = format_num) -> str:
    max_len_col = 0
    columns     = []
    
    #Extract column values
    for col in df.columns:
        if type(col) == tuple:
            col = wdl.write_list(col)
        else:
            col = wdl.write_primitive(col)
    
        max_len_col = max(len(col), max_len_col)
        columns.append(col)
        
    code = ''
    for name, col in zip(columns, df.columns):
        array = df[col]
        line  = name + ' '*(max_len_col-len(col) + 1) + ' : ['
        for value in array:
            line += n_format(value) + ', '
        line  = line[:-2] + ']'
        code += line + '\n'
    
    return code

###############################################################################
#Custom Lists
###############################################################################
def write_multiline_list(lst, indent_type='\t', indent_level=1) -> str:
    
    code = [indent_type*indent_level + wdl.write_primitive(x) for x in lst]
    code = '\n'.join(code)
    code = '[\n' + code + f'\n{indent_type*indent_level}]' 
    
    return code

###############################################################################
#Increasing Directory
###############################################################################
def increase_directory(code: str, *keys: list[str]) -> str:
    result = []
    for line in code.split('\n'):
        if line:
            if line[0] == ';':
                line_ = ''.join([';' + k for k in keys]) + line 
                result.append(line_)
                continue
        
        result.append(line)
    
    result = '\n'.join(result)
    return result
    
