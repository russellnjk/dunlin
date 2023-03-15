
states = {'H' : [0, 0],
          'A' : [0, 0],
          'B' : [0, 0],
          'C' : [1, 1]
          }

parameters = {
    'k_synH'   : [1,   1],
    'k_synB'   : [10, 10],
    'k_synH_A' : [0.05, 0.05],
    
    'J_A'   : [0.1, 0.1],
    'J_B_x' : [0.1, 0.1],
    'J_B_y' : [0.1, 0.1],
    
    'F_A'   : [0, 0],
    'F_B_x' : [0.05, 0.05],
    'F_B_y' : [0.05, 0.05]
    }

reactions = {
    'synH' : ['      -> H', 'k_synH*C'],
    'synB' : ['H + A -> B', 'k_synB*H*A', 'vrb1*vrb2']
    }

functions = {
    'func0': ['x', 'k', '-x*k']
    }

variables = {
    'vrb0' : 'func0(k_synH, C)',
    'vrb1' : 'B*A',
    'vrb2' : 1,
    'vrb3' : 'vrb0*vrb1*vrb2'
    }

rates = {
    'C': 'vrb0'
    }

diffusion = {
    'H' : 0.1,
    'A' : 'J_A',
    'B' : ['J_B_x', 'J_B_y'],
    }

advection = {
    'H' : 0.1, 
    'A' : 'F_A',
    'B' : ['F_B_x', 'F_B_y'],
    }

compartments = {
    'cytosol' : {
        'domain_type': 'cytosolic',
        'contains'   : ['A', 'H', 'C'],
        'unit_size'  : 1
        },
    'medium' : {
        'domain_type': 'extracellular',
        'contains'   : ['B'],
        'unit_size'  : 1
        }
    }

boundary_conditions = {
    'bcBx' : ['B', 0, 'Neumann',   'x'],
    'bcBy' : ['B', 7, 'Dirichlet', 'y']
    }


coordinate_components = {
    'x': [0, 10], 
    'y': [0, 10]
    }

grid_config = {
    'gr_main': {'config'  : [2, [0, 10], [0, 10]], 
                'children': ['gr_0']
                },
    'gr_0'   : {'config': [1, [2, 8 ], [2, 8 ]], 
                'children': ['gr_1']
                },
    'gr_1'   : {'config': [0.5, [3, 7 ], [3, 7 ]]                 
                }
    }

domain_types = {
    'extracellular' : {
        'medium0': [1, 1]
        },
    'cytosolic': {
        'cytosol0': [5, 5]
        }
    }

adjacent_domains = {
    'cytosol0_medium0': ['cytosol0', 'medium0']
    }

geometry_definitions = {
    'cell': {
        'definition' : 'csg',
        'domain_type': 'cytosolic',
        'order'      : 1,
        'node'       : ['circle', ['translate', 5, 5]]
        },
    'field': {
        'definition' : 'csg',
        'domain_type': 'extracellular',
        'order'      : 0,
        'node'       : ['square', ['scale', 5, 4], ['translate', 5, 5]]
        }
    }

geometry = {
    'coordinate_components': coordinate_components,
    'grid_config'          : grid_config,
    'domain_types'         : domain_types,
    'adjacent_domains'     : adjacent_domains,
    'geometry_definitions' : geometry_definitions
    }

M0 = {'states'              : states,
      'parameters'          : parameters,
      'reactions'           : reactions,
      'functions'           : functions,
      'variables'           : variables,
      'rates'               : rates,
      'diffusion'           : diffusion,
      'advection'           : advection,
      'compartments'        : compartments,
      'boundary_conditions' : boundary_conditions,
      'geometry'            : geometry
      }

all_data = {'M0': M0}