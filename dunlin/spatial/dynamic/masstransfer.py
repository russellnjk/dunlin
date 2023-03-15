import numpy as np

import dunlin.utils as ut


vrb_template = '''
    {name} = {expr}'''

rxn_template1 = '''
    #Reaction {name}
    {name} = {expr} 
'''

rxn_template_bulk = '''
    {diff} += {coeff}*{name}'''

rxn_template_edge = '''
    {diff}[{edge}] += {coeff}*{name}'''
    
mt_template = '''
    #Mass transfer {state}, {axis}
    _src  = _np.array({src})
    _dst  = _np.array({dst})
    _size = _np.array({size})
    _adv  = {adv_coeff} *{state}[_src] *_size**{dims}
    _dfn  = {dfn_coeff} *({state}[_src] - {state}[_dst]) *_size**{dims}
    
    _adv{state}_{axis} = _np.zeros({nzero})
    _dfn{state}_{axis} = _np.zeros({nzero})
    
    _adv{state}_{axis}[_dst] += _adv
    _adv{state}_{axis}[_src] -= _adv
    _dfn{state}_{axis}[_dst] += _dfn
    _dfn{state}_{axis}[_src] -= _dfn
    
    {diff} += _adv{state}_{axis} + _dfn{state}_{axis} 
'''

Neumann_template_plus = '''
    #Neumann boundary condition {state}, +{axis}
    _src  = _np.array({src})
    _size = _np.array({size})
    
    _bc{state}_{axis} = {expr} *_size**{dims}
    
    {diff}[_src] += _bc{state}_{axis}
'''
 #{state}[_bnd{state}_{axis}] 

