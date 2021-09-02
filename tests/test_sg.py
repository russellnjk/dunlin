###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin              as dn
import dunlin.strike_goldd as dsg

if __name__ == '__main__':
    ###############################################################################
    #Part 1: Test Symbolic Variable Generation
    ###############################################################################
    model_filename          = 'sg_test_files/M1.dun'
    dun_data, models        = dn.read_file(model_filename)
    model                   = models['M1'] 
    model.strike_goldd_args = {'observed': ['x1'], 
                                'unknown' : ['p0', 'p1'],
                                'init'    : {'x0': 1, 'x1': 0, 'x2': 0},
                                'inputs'  : {},
                                'decomp'  : []
                                }
    
    symbolic, r, template = dsg.convert2symbolic(model)
    
    assert symbolic['r0']   ==  symbolic['p0']*symbolic['x0']
    assert symbolic['r1']   ==  symbolic['p1']*symbolic['x1']
    assert symbolic['r2']   ==  symbolic['p2']*symbolic['x2']
    assert symbolic['d_x0'] == -symbolic['r0'] 
    assert symbolic['d_x1'] ==  symbolic['r0'] - symbolic['r1']
    assert symbolic['d_x2'] ==  symbolic['r1'] - symbolic['r2']
    
    ###############################################################################
    #Part 2: Test Low Level Call
    ###############################################################################
    model                   = models['M1'] 
    model.strike_goldd_args = {'observed': ['x1'], 
                                'unknown' : ['p0', 'p1'],
                                'init'    : {'x0': 1, 'x1': 0, 'x2': 0},
                                'inputs'  : {},
                                'decomp'  : []
                                }
    
    symbolic, r, template = dsg.convert2symbolic(model)
    result                = dsg.sga.strike_goldd(**r)
    
    result = {str(k): v for k, v in result.items()}
    answer = {'x0': True, 'x1': True, 'x2': False, 'p0': True, 'p1': True}
    
    for k, v in answer.items():
        assert result[k] == v
        
    model                   = models['M2'] 
    model.strike_goldd_args = {'observed' : ['x'], 
                                'unknown' : ['yield_S'],
                                'init'    : {'x': 0.05, 'S': 0.5},
                                'inputs'  : {},
                                'decomp'  : []
                                }
    
    symbolic, r, template = dsg.convert2symbolic(model)
    
    result = dsg.sga.strike_goldd(**r)
    result = {str(k): v for k, v in result.items()}
    answer = {'x': True, 'S': True, 'yield_S': True}
    
    for k, v in answer.items():
        assert result[k] == v
    
    ###############################################################################
    #Part 3: Test High Level Call
    ###############################################################################
    model  = models['M2'] 
    result = dsg.run_strike_goldd(model)
    
    answer = {'x': True, 'S': True, 'yield_S': True}
    
    for k, v in answer.items():
        assert result[k] == v
    
    ###############################################################################
    #Part 4: Test Markup
    ###############################################################################
    model_filename          = 'sg_test_files/M2.dun'
    dun_data, models        = dn.read_file(model_filename)
    model  = models['M2'] 
    result = dsg.run_strike_goldd(model)
    
    answer = {'x': True, 'S': True, 'yield_S': True}
    
    for k, v in answer.items():
        assert result[k] == v   