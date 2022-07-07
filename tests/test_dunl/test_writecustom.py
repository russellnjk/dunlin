import numpy  as np
import pandas as pd
from datetime import datetime

import addpath
import dunlin as dn
from dunlin.standardfile.dunl.writecustom import *
from dunlin.standardfile.dunl.readdunl import read_dunl_code

###############################################################################
#Check DataFrame Formatting
###############################################################################
vector = np.linspace(0, 99, 100)
arr    = np.reshape(vector, [10, 10])
df     = pd.DataFrame(arr, columns=list('abcdefghij'))

r = write_numeric_df_no_index(df)
s = read_dunl_code(';A\n' + r)['A']
s = pd.DataFrame(s)
assert all(s == df)

#Test write_numeric_df
index  = list('abcdefghij')
df     = pd.DataFrame(arr, columns=list('abcdefghij'), index=index)

r = write_numeric_df(df)
s = read_dunl_code(';A\n' + r)['A']
s = pd.DataFrame(s)
assert all(s == df)

#Test multiindex and write_numeric_df
index = pd.MultiIndex.from_product([['c0', 'c1'], [0, 1, 2, 3, 4]])
df     = pd.DataFrame(arr, columns=list('abcdefghij'), index=index)

r = write_numeric_df(df)
s = read_dunl_code(';A\n' + r)['A']
s = pd.DataFrame(s)
assert all(s == df)

#Test multiindex and write_numeric_df
index   = pd.MultiIndex.from_product([['c0', 'c1'], [0, 1, 2, 3, 4]])
columns = pd.MultiIndex.from_product([['x0', 'x1'], [0, 1, 2, 3, 4]])
df      = pd.DataFrame(arr, columns=columns, index=index)

r = write_numeric_df(df)
s = read_dunl_code(';A\n' + r)['A']
s = pd.DataFrame(s)
assert all(s == df)

# print(df)
# print(r)
# print(read_dunl_code(';A\n' + r))

###############################################################################
#Check Multiline List Formatting
###############################################################################
a = ['a', 0, True]
r = write_multiline_list(a, indent_level=1)
s = '''
[
	a
	0
	True
	]
'''

class C:
    @classmethod
    def to_dunl(cls):
        s = 'c : ' + write_multiline_list(['a = 0', 'b = 1'])
        
        return s
    
    
r = C.to_dunl()
s = '''
c : [
	a = 0
	b = 1
	]
'''
assert r.strip() == s.strip()
#Note that tabs(\t) are used instead spaces
print(r)
