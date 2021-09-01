import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin            as dn  
import dunlin.model      as dml
import dunlin.simulate as sim
        
if __name__ == '__main__':
    #Some overhead for testing
    plt.close('all')
    
    ###############################################################################
    #Part 1: Low-level ODE and EXV 
    ###############################################################################
    def plot(t, y, AX, label='_nolabel'):
        for i, ax in enumerate(AX):
            ax.plot(t, y[i], label=label)
            top = np.max(y[i])
            top = top*1.2 if top else 1
            top = np.maximum(top, ax.get_ylim()[1])
            bottom = -top*.05 
            ax.set_ylim(bottom=bottom, top=top)
            
            if label != '_nolabel':
                ax.legend()
    
    rtol      = 1e-2
    
    dun_data, models = dn.read_file('simulate_test_files/M1.dun')
    model           = models['M1']
    event_objs       = model._events
    
    y0    = np.array([1, 0, 0])
    p     = np.array([0.01, 0.01])
    
    fig  = plt.figure()
    AX   = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    #Integration
    print('Integration') 
    t, y = model.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Integration')
    
    #Test extraction of event timepoints and assignments
    print('Test extraction of event timepoints and assignments')
    record = sim.SimResult.get_event_record(model)
    
    assert len(record) == 2
    
    timepoint, new_p = record[0]
    assert timepoint == 200
    assert all(new_p == [0.03, 0.01])
    
    timepoint, new_p = record[1]
    assert timepoint == 600
    assert all(new_p == [0.001, 0.01])
    
    #Test tabulation of params
    print('Test tabulation of params')
    p_table = sim.SimResult.tabulate_params(model, t, y, p, 's0')
    
    assert all(p_table[:,1]     == 0.01)
    assert all(p_table[:20,0]   == 0.01)
    assert all(p_table[21:62,0] == 0.03)
    assert all(p_table[62:,0] == 0.001)
    
    ###############################################################################
    #Part 2: SimResult
    ############################################################################### 
    
    ###
    #df = pd.DataFrame(y_vals[:,np.newaxis], index=x_vals)
    #df.to_csv(f'simulate_test_files/M1_{x}.csv')
    ###
    
    #Test manual instantiation
    fig = plt.figure()
    gs  = fig.add_gridspec(2, 4) 
    AX  = [fig.add_subplot(gs[:,i]) for i in range(3)] + \
          [fig.add_subplot(gs[i,3]) for i in range(2)]

    sim_result = sim.SimResult(model, t, y, p, 's0')
    
    for i, x in enumerate(model._states + tuple(model._exvs)):
        x_vals, y_vals  = sim_result.get2d(x)
        AX[i].plot(x_vals, y_vals)
        AX[i].set_title(x)
    
    #Test by calling integration function
    fig = plt.figure()
    gs  = fig.add_gridspec(2, 4) 
    AX  = [fig.add_subplot(gs[:,i]) for i in range(3)] + \
          [fig.add_subplot(gs[i,3]) for i in range(2)]

    sim_results = sim.integrate_model(model, multiply=True)
    sim_result  = sim_results['s0'][0]
    
    for i, x in enumerate(model._states + tuple(model._exvs)):
        x_vals, y_vals  = sim_result.get2d(x)
        
        AX[i].plot(x_vals, y_vals)
        AX[i].set_title(x)
        
    assert len(t) == 105
    
    #Test multiplicative simulate
    dun_data, models = dn.read_file('simulate_test_files/M2.dun')
    model            = models['M2']
    
    fig = plt.figure()
    gs  = fig.add_gridspec(2, 4) 
    AX  = [fig.add_subplot(gs[:,i]) for i in range(3)] + \
          [fig.add_subplot(gs[i,3]) for i in range(2)]
          
    sim_results = sim.integrate_model(model, multiply=True)
    assert len(sim_results) == 2
    
    for key, value in sim_results.items():
        assert len(value) == 2
    
    
    all_vars = list(enumerate(model._states + tuple(model._exvs)))
    
    for scenario in sim_results:
        for estimate, sim_result in sim_results[scenario].items():
            for i, x in all_vars:
                x_vals, y_vals  = sim_result.get2d(x)
                
                AX[i].plot(x_vals, y_vals, label=(scenario, estimate))
                AX[i].set_title(x)
    AX[0].legend()
    
    #Test non multiplicative simulate
    fig = plt.figure()
    gs  = fig.add_gridspec(2, 4) 
    AX  = [fig.add_subplot(gs[:,i]) for i in range(3)] + \
          [fig.add_subplot(gs[i,3]) for i in range(2)]
          
          
    sim_results = sim.integrate_model(model, multiply=False)
    assert len(sim_results) == 2
    
    for key, value in sim_results.items():
        assert len(value) == 1
    
    for scenario in sim_results:
        for estimate, sim_result in sim_results[scenario].items():
            for i, x in all_vars:
                x_vals, y_vals  = sim_result.get2d(x)
                
                AX[i].plot(x_vals, y_vals, label=(scenario, estimate))
                AX[i].set_title(x)
    AX[0].legend()
    
    ###############################################################################
    #Part 2: Plotting
    ############################################################################### 
    #Modify line arguments
    line_args = {'marker': 'o', 'linestyle': 'None'}
    
    #Test Axes generation
    print('Test Axes generations')
    fig, AX = sim.gridspec(2, 4, 
                           [0, 2, 0, 1], 
                           [0, 2, 1, 2], 
                           [0, 2, 2, 3], 
                           [0, 1, 3, 4], 
                           [1, 2, 3, 4]
                           )
         
    sim_results = sim.integrate_model(model, multiply=True)
    assert len(sim_results) == 2
    
    for key, value in sim_results.items():
        assert len(value) == 2
    
    #Test plotting function
    print('Test plotting function')
    AX_ = dict(zip([*model.states.columns, *model.exvs], AX))
    AX_ = sim.plot_sim_results(sim_results, AX_, marker='o', linestyle={'s0': '-', 's1': {1: ':'}})
    
    AX[0].legend()
    
    ###############################################################################
    #Part 3: Multi-Model
    ############################################################################### 
    dun_data, models = dn.read_file('simulate_test_files/M3.dun')
    
    fig0, AX0 = sim.gridspec(2, 4, 
                             [0, 2, 0, 1], 
                             [0, 2, 1, 2], 
                             [0, 2, 2, 3], 
                             [0, 1, 3, 4], 
                             [1, 2, 3, 4]
                             )
    fig1, AX1 = sim.gridspec(2, 4, 
                             [0, 2, 0, 1], 
                             [0, 2, 1, 2], 
                             [0, 2, 2, 3], 
                             [0, 1, 3, 4], 
                             [1, 2, 3, 4]
                             )
    
    AX  = AX0 + AX1
    
    AX_ = {'M3'  : dict(zip([*models['M3' ].states.columns, *models['M3' ].exvs], AX0)),
           'M3a' : dict(zip([*models['M3a'].states.columns,], AX1))
            }
    
    print('Test multi model integration and plotting')
    all_sim_results = sim.integrate_models(models) 
    AX_             = sim.plot_all_sim_results(all_sim_results, AX_, marker={'M3a': '+'}) 
    
    ###############################################################################
    #Part 4: .dun Files
    ############################################################################### 
    dun_data, models = dn.read_file('simulate_test_files/M4.dun')
    
    fig, AX = sim.gridspec(2, 4, 
                           [0, 2, 0, 1], 
                           [0, 2, 1, 2], 
                           [0, 2, 2, 3], 
                           [0, 1, 3, 4], 
                           [1, 2, 3, 4]
                           )
    
    AX_ = {'M4'  : dict(zip([*models['M4' ].states.columns, *models['M4' ].exvs], AX)),
           }
    
    print('Test sim_args in dun file')
    all_sim_results = sim.integrate_models(models) 
    AX_             = sim.plot_all_sim_results(all_sim_results, AX_) 
    
    
    