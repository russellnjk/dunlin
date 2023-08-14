import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd

import addpath
import dunlin                    as dn  
import dunlin.data.timeresponse  as dts
import dunlin.utils_plot         as upp

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
      'reactions' : {'g0' : {'stoichiometry' : {'x0': 1},
                             'rate'          : 'u0'
                             },
                     'g1' : {'stoichiometry' : {'x1': 1},
                             'rate'          : 'u1'
                             },
                     'r0' : {'stoichiometry' : {'x0': -1},
                             'rate'          : 'p0*x0'
                             },
                     'r1' : {'stoichiometry' : {'x1': -1},
                             'rate'          : 'p1*x1'
                             },
                     },
      'variables' : {'v0' : 'x0'
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
                     },
      'data_args' : {'dataset' : {'line_args':{'color': {0 : 'purple',
                                                         1 : 'cobalt'
                                                         },
                                               'marker'    : 'x',
                                               'capsize'   : 5 
                                               }
                                  }
                    }
      }

all_data = {'M1': m1}

#Read model
model = dn.ODEModel('M1', **m1)

time        = np.linspace(0, 100, 51)
y_data0     = 50 - 50*np.exp(-0.1*time)
y_data1     = 50 + 50*np.exp(-0.1*time)
cols0 = pd.MultiIndex.from_product([['x0', 'x1'], [0]])
cols1 = pd.MultiIndex.from_product([['x0', 'x1'], [1]])
df0   = pd.DataFrame(np.array([y_data0, y_data0]).T, index=time, columns=cols0)
df1   = pd.DataFrame(np.array([y_data1, y_data1]).T, index=time, columns=cols1)
exdf  = pd.DataFrame([[10, 20]], columns=['ex0', 'ex1'], index=[0, 1])

##############################################################################
#Instantiate without Model
##############################################################################
trd = dts.TimeResponseData(df0, df1, exdf)

value = trd.get('x0')
# assert 0 in value['x0']
# assert 1 in value['x0']
# print(value)

value = trd['x0', 0]
# print(value.head())
###############################################################################
#Instantiate with Model
###############################################################################

trd = dts.TimeResponseData(df0, df1, model=model)

# print(trd['x0', 0].head())

fig, AX_ = upp.figure(1, 2)

def label(scenario, variable, ref):
    return f'Ref {ref}, Scenario {scenario}'

trd.plot_line(AX_[0], 'x0', label=label)
AX_[0].legend()

###############################################################################
#MultiIndex
###############################################################################
plus_y  = df0 + 5
minus_y = df0 - 5
df0_    = pd.concat(dict(enumerate([minus_y, df0, plus_y])))
df0_    = df0_.swaplevel(axis=0)
df0_    = df0_.loc[sorted(df0_.index)]

trd = dts.TimeResponseData(df0_, df1, model=model)

def label(scenario, variable, ref):
    return f'Ref {ref}, Scenario {scenario}'

trd.plot_line(AX_[1], 'x0', label=label)
AX_[1].legend()

###############################################################################
#Bar Plot
###############################################################################
trd = dts.TimeResponseData(df0, df1, exdf, model=model)

fig, AX_ = upp.figure(1, 2)

trd.plot_bar(AX_[0], ['ex0', 'ex1'])
AX_[0].legend()

#Create multiple trials/replicates with MultiIndex
plus_y  = exdf + 5
minus_y = exdf - 5
exdf_   = pd.concat(dict(enumerate([minus_y, exdf, plus_y])))
exdf_   = exdf_.swaplevel(axis=0)
exdf_   = exdf_.loc[sorted(exdf_.index)]

trd = dts.TimeResponseData(df0, df1, exdf_)

trd.plot_bar(AX_[1], ['ex0', 'ex1'])
AX_[1].legend()