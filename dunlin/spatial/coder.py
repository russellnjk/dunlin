import numpy as np
from numba       import njit
from numba.core  import types
from numba.typed import Dict

import dunlin.utils as ut
from .shape_stack          import ShapeStack
from dunlin.datastructures import SpatialModelData

class Coder(ShapeStack):
    '''Takes the body code of the ShapeStack Parent class and generates the rhs 
    for integration and post processing.
    
    '''
    rhs_name      : str
    rhs_dict_name : str

    def __init__(self, spatial_data: SpatialModelData, numba: bool=True):
        super().__init__(spatial_data)
        
        float64    = types.float64
        voxel_type = types.UniTuple(float64, self.ndims)
        self.scope = {'_np'     : np, 
                      '_njit'   : njit,
                      '_Dict'   : Dict,
                      '_single' : voxel_type,
                      '_double' : types.UniTuple(voxel_type, 2),
                      '_float'  : float64
                      }
        
        #Make definitions/return values for rhs
        _njit      = '@_njit' if numba else ''
        ref        = spatial_data.ref
        rhs_name   = f'_model_{ref}'
        rhs_def    = f'{_njit}\ndef {rhs_name}(time, states, parameters):\n'
        rhs_return = '\treturn _d_states\n\n'
        
        #Combine chunks
        self.rhs_code = '\n'.join([rhs_def,
                                   self.body_code,
                                   rhs_return
                                   ])
        
        #Make definitions/return values for rhs dict
        ref            = spatial_data.ref
        rhs_dict_name  = f'_dict_{ref}'
        rhs_def        = f'def {rhs_dict_name}(time, states, parameters):\n'
        states         = list(spatial_data.states.keys()) 
        parameters     = list(spatial_data.parameters.keys())
        variables      = list(spatial_data.variables.keys())
        reactions      = list(spatial_data.reactions.keys())
        rhs_return     = states + parameters + variables + reactions
        
        for state in states:
            rhs_return.append(ut.diff(state))
            
            if state not in self.mass_transfer_templates:
                continue
            
            rhs_return.append(f'_adv_{state}')
            rhs_return.append(f'_dfn_{state}')
            
            bcs = self.mass_transfer_templates[state]['boundary_conditions']
            for shift, bc in bcs.items():
                if bc is None:
                    continue
                rhs_return.append(self._make_boundary(state, shift))
                

        rhs_return = [f'{i} = {i}' for i in rhs_return]
        rhs_return = 'dict(\n\t\t' +  ',\n\t\t'.join(rhs_return) + '\n\t\t)'
        
        rhs_return = f'\treturn {rhs_return}\n\n'
        
        #Combine chunks
        self.rhs_dict_code = '\n'.join([rhs_def,
                                        self.body_code,
                                        rhs_return
                                        ])
        
        #Save the function names
        self.rhs_name      = rhs_name
        self.rhs_dict_name = rhs_dict_name
    
        #Execute
        exec(self.rhs_code,      self.scope)
        exec(self.rhs_dict_code, self.scope)
        
        #Extract
        self.rhs      = self.scope[rhs_name]
        self.rhs_dict = self.scope[rhs_dict_name]
        
        self.rhs.code      = self.rhs_code
        self.rhs_dict.code = self.rhs_dict_code
        
        
    @staticmethod
    def _empty_dict_code(n=1):
        '''Overwrites to ensure
        '''
        # return '{}'
        if n == 1:
            return '_Dict.empty(key_type=_single, value_type=_float)'
        else:
            return '_Dict.empty(key_type=_double, value_type=_float)'
    