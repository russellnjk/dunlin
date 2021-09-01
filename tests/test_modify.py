import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                              as dn 
import dunlin._utils_model.dun_file_reader as dfr
import dunlin._utils_model.ivp             as ivp
import dunlin._utils_model.ode_coder       as odc

if __name__ == '__main__':
    #Some overhead for testing
    plt.close('all')
    
    ###############################################################################
    #Part 1: Manual Instantiation
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
    
    def modify(y, p, scenario):
        x0, x1, x2 = y
        p0, p1     = p
        
        if scenario == 's0':
            x1 = 0.5
        
        new_y = np.array([x0, x1, x2])
        new_p = np.array([p0, p1])
        
        return new_y, new_p    
    
    def func(t, y, p):
        x0 = y[0]
        x1 = y[1]
        x2 = y[2]
        
        p0 = p[0]
        p1 = p[1]
        
        r0 = p0*x0
        r1 = p1*x1
        
        d_x0 = -r0 
        d_x1 = +r0 -r1
        d_x2 = r1 
        
        return np.array([d_x0, d_x1, d_x2])
    
    y0    = np.array([1, 0, 0])
    p     = np.array([0.01, 0.01])
    tspan = np.linspace(0, 1000, 101)
    
    fig  = plt.figure()
    AX   = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    rtol = 1e-3
    
    ###For saving###
    #df = pd.DataFrame(np.concatenate(([t], y), axis=0))
    #df.to_csv('event_test_files/simulate_event_1.csv', index=False)
    ################
    
    #Case 31: Modify
    print('Case 31: Modify')
    t, y = ivp.integrate(func, tspan, y0, p, modify=modify, overlap=True, include_events=True, scenario='s0')
    
    plot(t, y, AX, 'Case 31: Modify')
    df = pd.read_csv('event_test_files/simulate_event_31.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    ###############################################################################
    #Part 2: Dynamic Instantiation
    ###############################################################################
    #Set up
    dun_data   = dfr.read_file('event_test_files/M2.dun')
    model_data = dun_data['M2']
    func_data  = odc.make_ode_data(model_data)
    
    func_ = func_data['rhs'][1]
    y0    = np.array([1, 0, 0])
    p     = np.array([0.01, 0.01])
    tspan = np.linspace(0, 1000, 101)
    
    modify = func_data['modify'][1]
    
    fig  = plt.figure()
    AX   = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    #Case 41: Modify
    print('Case 41: Modify')
    t, y = ivp.integrate(func_, tspan, y0, p, modify=modify, overlap=True, include_events=True, scenario='s0')
    
    plot(t, y, AX, 'Case 41: Modify')
    df = pd.read_csv('event_test_files/simulate_event_31.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    