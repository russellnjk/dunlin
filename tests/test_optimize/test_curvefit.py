import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                    as dn  
import dunlin.optimize.curvefit  as cf
import dunlin.utils_plot         as upp

from data import all_data

plt.ion()
plt.close('all')

#Read model
model = dn.ODEModel.from_data(all_data, 'M1')

#The y values are first order diff equations i.e. dx = g - p*x
time    = np.linspace(0, 100, 51)
y_data0 = 50 - 50*np.exp(-0.1*time)
y_data1 = 50 + 50*np.exp(-0.1*time)

#Formatted as scenario -> state -> series
series0 = pd.Series(y_data0, index=time)
series1 = pd.Series(y_data1, index=time)

data = {0 : {'x0' : series0,
             'x1' : series0
             },
        1 : {'x0' : series1,
             'x1' : series1
             }
        }

#Case 1: Function-based parameter estimation
print('Function-based parameter estimation')
cfts = cf.fit_model(model, data, algo='differential_evolution')
cft  = cfts[0]

trace = cft.trace
o     = trace.raw
assert all(np.isclose(o.x, [5, 5], rtol=1e-3) )
assert type(trace.samples) == pd.DataFrame

fig, AX = upp.figure(1, 2)

trace.plot_steps(AX[0], 'posterior', linestyle='-')
trace.plot_steps(AX[1], ['u0', 'u1'], linestyle='-')

AX[0].legend()

#Integrate and plot
fig, AX = upp.figure(1, 2)

cft.plot_line(AX[0], 'x0')
cft.plot_line(AX[1], 'x1')

cft.plot_data(AX[0], 'x0')
cft.plot_data(AX[1], 'x1')

AX[0].legend()

#Test master plotting function
fig, AX = upp.figure(1, 2)
cft.plot_result(AX[0], 'x0')
cft.plot_result(AX[1], 'x1')
AX[0].legend()
