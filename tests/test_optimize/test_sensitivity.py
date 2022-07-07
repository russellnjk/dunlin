import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
import seaborn           as     sns
 
###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                      as dn  
import dunlin.simulate             as sim
import dunlin.optimize.sensitivity as sen     
import dunlin.utils_plot           as upp

plt.ion()
plt.close('all')

m1 = {'states'     : {'x0' : [0, 100],
                      'x1' : [0, 100]
                      },
      'parameters' : {'p0' : [0.1]*2,
                      'p1' : [0.1]*2,
                      'u0' : [1  ]*2,
                      'u1' : [9  ]*2
                      },
      'reactions' : {'g0' : ['   -> x0', 'u0'   ],
                     'g1' : ['   -> x1', 'u1'   ],
                     'r0' : ['x0 ->   ', 'p0*x0'],
                     'r1' : ['x1 ->   ', 'p1*x1']
                     },
      'variables' : {'v0' : 'x0'
                     },
      'tspan'     : {0: np.linspace(0, 110, 111),
                     1: np.linspace(0, 110, 111),
                     },
      'extra'     : {'final_x1': ['index', 'x1', -1]
                     },
      'sim_args'  : {'line_args': {'linestyle': '-',
                                   'color'    : {0: 'cobalt',
                                                 1: 'crimson'
                                                 }
                                   },
                     },
      'optim_args': {'free_parameters' : {'u0': {'bounds': [0, 10], 
                                                 'prior': ['uniform', 0, 10]
                                                 },
                                          'u1': {'bounds': [0, 10], 
                                                 'prior': ['uniform', 0, 10]
                                                 },
                                          },
                     'settings'   : {'disp'   : False,
                                     'popsize': 5
                                     },
                     'sensitivity' : {'sample'  : {'N' : 256},
                                      'analyze' : {'disp': True}
                                      },
                     },
      'data_args' : {'dataset' : {'line_args':{'color': {0 : 'cobalt',
                                                         1 : 'crimson'
                                                         },
                                               'marker'    : 'x'
                                               }
                                  }
                    }
      }

all_data = {'M1': m1}

#Read model
model = dn.ODEModel('M1', **m1)

st      = sen.SensitivityTest(model)
problem = st.problem
answer = {'num_vars': 2, 'names': ['u0', 'u1'], 'bounds': [[0, 10], [0, 10]]}
for key, value in answer.items():
    assert problem[key] == value

# problem, free_parameter_samples = st.sample_saltelli()

# Y = st.evaluate('final_x1', problem, free_parameter_samples)

#Sobol
senresult = st.analyze_sobol('final_x1')

# #With groups
# groups = {'gr0': ['u0', 'u1']}
# st.sensitivity_args['groups'] = groups
# senresult = st.analyze_sobol('final_x1')

#Without combining scenarios
# problem, free_parameter_samples = st.sample_saltelli()
# Y = st.evaluate('final_x1', problem, free_parameter_samples, False)

senresult = st.analyze_sobol('final_x1', combine_scenarios=False)

fig, AX_ = upp.figure(2, 2)

free_parameters = list(st.free_parameters)

group = ['S1', 'S1_conf']

for ax, (c, Si) in zip(AX_[:2], senresult.Si.items()):
    df = pd.DataFrame({g: Si[g] for g in group}, index=free_parameters)
    sns.heatmap(df, ax=ax, annot=True)
    ax.set_title(f'Scenario {c}')
