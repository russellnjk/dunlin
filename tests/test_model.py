import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                              as dn  
import dunlin.model                        as dml  
import dunlin._utils_model.dun_file_reader as dfr
import dunlin._utils_model.ivp             as ivp

if __name__ == '__main__':
    #Some overhead for testing
    plt.close('all')
    
    ###############################################################################
    #Part 1A: Instantiation from .dun
    ###############################################################################
    dun_data = dfr.read_file('model_test_files/M3.dun')
    models   = dml.make_models(dun_data)
    
    try:
        dun_data = dfr.read_file('model_test_files/M3_missing.dun')
        models   = dml.make_models(dun_data)
    except dml.DunlinModelError as e:
        assert e.num == 10
    else:
        assert False
        
    try:
        dun_data = dfr.read_file('model_test_files/M3_mismatched.dun')
        models   = dml.make_models(dun_data)
    except dml.DunlinModelError as e:
        assert e.num == 11
    else:
        assert False
    
    try:
        dun_data = dfr.read_file('model_test_files/M3_recursive.dun')
        models   = dml.make_models(dun_data)
    except dml.DunlinModelError as e:
        assert e.num == 12
    else:
        assert False
    
    ###############################################################################
    #Part 1B: One-Step Instantiation from .dun
    ###############################################################################
    dun_data, models = dml.read_file('model_test_files/M3.dun')
    
    ###############################################################################
    #Part 2: RHS Tests
    ###############################################################################
    #Test rhs
    m  = models['M3']
    y0 = np.ones(len(m.states.columns))
    p  = np.ones(len(m.params.columns))
    t  = 0
    
    dy = m._rhs(t, y0, p)
    assert all( dy == np.array([-1.,   0.5,  0.5,  1. ]))
    
    ###############################################################################
    #Part 3A: Low-Level ODE Integration with 
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
    
    dun_data, models = dml.read_file('event_test_files/M1.dun')
    model0           = models['M1']
    event_objs       = model0._events
    
    func_ = model0._rhs
    y0    = np.array([1, 0, 0])
    p     = np.array([0.01, 0.01])
    tspan = np.linspace(0, 1000, 101)
    
    fig  = plt.figure()
    AX   = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    #Case 21: Dynamically generated Timer
    print('Case 21: Dynamically generated Timer')
    #Expect: Spike at 200
    events_ = event_objs[:1]
    t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 21: Dynamically generated Timer')
    df = pd.read_csv('event_test_files/simulate_event_1.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 22: Dynamically generated Event
    # print('Case 22: Dynamically generated Event')
    # #Expect: Spike at 160
    # events_ = event_objs[1:2]
    # t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 22: Dynamically generated Event')
    # df = pd.read_csv('event_test_files/simulate_event_2.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 23: Dynamically generated Event with delay
    # print('Case 23: Dynamically generated Event with delay')
    # #Expect: Spike at 425 and 830
    # events_ = event_objs[2:3]
    # t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 23: Dynamically generated Event with delay')
    # df = pd.read_csv('event_test_files/simulate_event_3.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 24: Dynamically generated Event with delay and not persistent
    # print('Case 24: Dynamically generated Event with delay and not persistent')
    # #Expect: No spike
    # events_ = event_objs[3:4]
    # t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 24: Dynamically generated Event with delay and not persistent')
    # df = pd.read_csv('event_test_files/simulate_event_4.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 25: Dynamically generated Event with delay and not persistent
    # print('Case 25: Dynamically generated Event with trigger at start')
    # #Expect: Spike at 10
    # events_ = event_objs[4:5]
    # t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 25: Dynamically generated Event with trigger at start')
    # df = pd.read_csv('event_test_files/simulate_event_5.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 26: Dynamically generated Event with multiple events
    # print('Case 26: Dynamically generated Event with multiple events')
    # #Expect: Spike at 25 and 300
    # events_ = [event_objs[1], event_objs[5]]
    # t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 26: Dynamically generated Event with multiple events')
    # df = pd.read_csv('event_test_files/simulate_event_6.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 27: Dynamically generated Event with prioritized events
    # print('Case 27: Dynamically generated Event with prioritized events')
    # #Expect: Spike at 800 with new value of 3.5
    # events_ = [event_objs[6], event_objs[7]]
    # t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 27: Dynamically generated Event with prioritized events')
    # df = pd.read_csv('event_test_files/simulate_event_7.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    ###############################################################################
    #Part 3B: High-Level ODE Integration with 
    ###############################################################################
    dun_data, models = dn.read_file('event_test_files/M1.dun')
    model0           = models['M1']
    event_objs       = model0._events
    
    y0    = np.array([1, 0, 0])
    p     = np.array([0.01, 0.01])
    
    fig  = plt.figure()
    AX   = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    #Case 21: Dynamically generated Timer
    print('Case 21: Dynamically generated Timer')
    #Expect: Spike at 200
    model0._events = event_objs[:1]
    t, y = model0.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 21: Dynamically generated Timer')
    df = pd.read_csv('event_test_files/simulate_event_1.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 22: Dynamically generated Event
    print('Case 22: Dynamically generated Event')
    #Expect: Spike at 160
    model0._events = event_objs[1:2]
    t, y = model0.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 22: Dynamically generated Event')
    df = pd.read_csv('event_test_files/simulate_event_2.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 23: Dynamically generated Event with delay
    print('Case 23: Dynamically generated Event with delay')
    #Expect: Spike at 425 and 830
    model0._events = event_objs[2:3]
    t, y = model0.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 23: Dynamically generated Event with delay')
    df = pd.read_csv('event_test_files/simulate_event_3.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 24: Dynamically generated Event with delay and not persistent
    print('Case 24: Dynamically generated Event with delay and not persistent')
    #Expect: No spike
    model0._events = event_objs[3:4]
    t, y = model0.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 24: Dynamically generated Event with delay and not persistent')
    df = pd.read_csv('event_test_files/simulate_event_4.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 25: Dynamically generated Event with delay and not persistent
    print('Case 25: Dynamically generated Event with trigger at start')
    #Expect: Spike at 10
    model0._events = event_objs[4:5]
    t, y = model0.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 25: Dynamically generated Event with trigger at start')
    df = pd.read_csv('event_test_files/simulate_event_5.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 26: Dynamically generated Event with multiple events
    print('Case 26: Dynamically generated Event with multiple events')
    #Expect: Spike at 25 and 300
    model0._events = [event_objs[1], event_objs[5]]
    t, y = model0.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 26: Dynamically generated Event with multiple events')
    df = pd.read_csv('event_test_files/simulate_event_6.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 27: Dynamically generated Event with prioritized events
    print('Case 27: Dynamically generated Event with prioritized events')
    #Expect: Spike at 800 with new value of 3.5
    model0._events = [event_objs[6], event_objs[7]]
    t, y = model0.integrate('s0', y0, p, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 27: Dynamically generated Event with prioritized events')
    df = pd.read_csv('event_test_files/simulate_event_7.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    ###############################################################################
    #Part 4: Attribute Tests
    ###############################################################################
    m_ = m.copy()
    assert m_ is not m
    for k in m.keys():
        r = m_[k] == m[k]
        try:
            assert r
        except:
            try:
                assert all(r)
            except:
                assert False
    
    