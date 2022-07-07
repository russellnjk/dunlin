import addpath
import dunlin as dn
import dunlin.standardfile.dunl.readdunl as rdn

#Test read_chunk
'''
Updates dct, curr_lst and interpolators during each call.
Returns the new curr_dct

Updates curr_dct after every chunk outside the function.
'''
dct           = {}
curr_lst      = []
curr_dct      = dct
interpolators = {}

#Interpolator
chunk = '`x` : 0'

r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
assert interpolators == {'x': '0'}
assert dct           == {} 
assert curr_lst      == []
assert r is dct
curr_dct = r

#Directory
chunk = ';a;b'

r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
assert interpolators == {'x': '0'}
assert dct      == {'a': {'b': {}}}
assert curr_lst == ['a', 'b']
assert r is dct['a']['b']
curr_dct = r

#Element
chunk = '''
{{x}} : [`x` : [{x}]]
    $x   : {x}
    $x.x : 0, 1, 2
    $$x  : x0, x1
'''

r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
assert interpolators == {'x': '0'}
assert dct           == {'a': {'b': {'x0': {0: [0, 1, 2]}, 
                                     'x1': {0: [0, 1, 2]}
                                     }
                               }
                         } 
assert curr_lst == ['a', 'b']
assert r is dct['a']['b']
curr_dct = r

#Ambiguous/erroneous chunk
chunk = ';a;b : 5'
try:
    r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
except:
    assert True
else:
    assert False

chunk = '`y``'
try:
    r = rdn.read_chunk(dct, curr_lst, curr_dct, interpolators, chunk)
except:
    assert True
else:
    assert False

# print(dct)

#Test read_lines
code = '''
`x` : 0
;a;b
{{x}} : [`x` : [{x}]]
    $x   : {x}
    $x.x : 0, 1, 2
    $$x  : x0, x1

#haha
;c #hehe
y : 3
'''
lines = code.split('\n')

r = rdn.read_lines(lines, includes_newline=False)
assert r == {'a': {'b': {'x0': {0: [0, 1, 2]}, 
                         'x1': {0: [0, 1, 2]}
                         }
                   }, 
             'c': {'y': 3}
             }

# print(r)

#Test on file
r  = rdn.read_dunl_file('example_data.dunl')
M1 = r['M1']

assert M1['states'] == {'x': {0: 0.1, 1: 0.1},
                        's': {0: 1000.0, 1: 1000.0},
                        'P': {0: 0, 1: 0},
                        'g': {0: 0, 1: 0},
                        'R': {0: 0.2, 1: 0.2},
                        'M': {0: 0, 1: 0},
                        'U': {0: 0, 1: 0},
                        'H': {0: 0.03, 1: 0.03},
                        'Rel': {0: 0, 1: 0}
                        }
assert M1['parameters'] == {'v_uptake': [1116.5, 1116.5],
                            'k_uptake': [206070.0, 206070.0],
                            'Yield': [9.7929, 9.7929],
                            'v_synprot': [1580.0, 1580.0],
                            'k_synprot': [1.8833, 1.8833],
                            'jcon': [0.4568, 0.4568],
                            'fR_con': [1.0, 1.0],
                            'fR_var': [1.0, 1.0],
                            'fM_con': [0.1241, 0.1241],
                            'fM_var': [19.4025, 19.4025],
                            'fU_con': [4.5642, 4.5642],
                            'fU_var': [0.0191, 0.0191],
                            'fH_con': [0.01, 0.01],
                            'fH_var': [0.05],
                            'k_fR': [3.1791e-05, 3.1791e-05],
                            'n_fR': [2.0, 2.0],
                            'k_fM': [0.6051, 0.6051],
                            'k_fs': [38.7304, 38.7304],
                            'n_fs': [1.0, 1.0],
                            'n_fM': [1.0, 1.0],
                            'k_fH': [482.1578, 482.1578],
                            'n_fH': [2.0, 2.0],
                            'k_P': [0.046, 0.046],
                            'ind': [0, 1],
                            'syng': [122070.0, 122070.0],
                            'k_ut': [582.7975, 582.7975],
                            'k_ct': [1.7564, 1.7564],
                            'degg': [0.1653, 0.1653],
                            'degspot': [0.4226, 0.4226],
                            'degrel': [0.2812, 0.2812],
                            'fRel_con': [1.0751e-06, 1.0751e-06],
                            'fRel_var': [0.0062684, 0.0062684],
                            'k_fRel': [0.34, 0.34],
                            'n_fRel': [1.0, 1.0]
                            }
assert M1['variables'] == {'n_R': 'R/7459',
                           'n_M': 'M/300',
                           'n_H': 'H/230',
                           'allprot': 'R+M+U+H',
                           'R_frac': 'R/allprot',
                           'H_frac': 'H/allprot',
                           'x2aa_umol': '1/110*1e6',
                           'x_in_aa_umol': 'x*x2aa_umol',
                           'uptake': 'v_uptake*n_M*s/(k_uptake + s)',
                           'synp': 'uptake',
                           'rsat': 'P/(P + k_synprot)',
                           'A': 'rsat',
                           'A_': '1-rsat',
                           'synprot': 'v_synprot*n_R*rsat',
                           'mu': 'synprot',
                           'regR': 'k_fR**n_fR/(g**n_fR + k_fR**n_fR)',
                           'regM': 'P**n_fM/(P**n_fM + k_fM**n_fM)*s**n_fs/(k_fs**n_fs+s**n_fs)',
                           'regU': 1,
                           'regH': 'ind*k_fH**n_fH/(k_fH**n_fH+s**n_fH)*P/(k_P+P)',
                           'regRel': 'P**n_fRel/(P**n_fRel + k_fRel**n_fRel)',
                           'jvar': '1-jcon',
                           'sum_con': 'fR+fM+fU+fH+fRel',
                           'sum_var': 'fR*regR+fM*regM+fU*regU+fH*regH+fRel*regRel',
                           'jR': 'jcon*fR_con/sum_con + jvar*fR_var*regR/sum_var',
                           'jM': 'jcon*fM_con/sum_con + jvar*fM_var*regM/sum_var',
                           'jU': 'jcon*fU_con/sum_con + jvar*fU_var*regU/sum_var',
                           'jH': 'jcon*fH_con/sum_con + jvar*fH_var*regH/sum_var',
                           'jRel': 'jcon*fRel_con/sum_con + jvar*fRel_var*regRel/sum_var',
                           'syng_eff': 'syng*Rel*n_R*(A_*k_ut)/(1 +A_*k_ut +rsat*k_ct)',
                           'degg_eff': 'degspot*rsat + degg'
                           }
assert M1['rates'] == {'s': '-uptake*x_in_aa_umol',
                       'x': 'mu*x',
                       'P': '-mu*P -synprot +synp*Yield',
                       'R': '-mu*R +syn_R',
                       'M': '-mu*M +syn_M',
                       'U': '-mu*U +syn_U',
                       'H': '-mu*H +syn_H',
                       'g': '-mu*g +syng_eff -degg_eff*g',
                       'Rel': '-mu*Rel +syn_Rel -degrel*Rel'
                       }
    
# print(r)
