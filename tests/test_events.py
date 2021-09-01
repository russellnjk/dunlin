import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

###############################################################################
#Non-Standard Imports
###############################################################################
import addpath
import dunlin                              as dn 
import dunlin._utils_model.events          as uev
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
                
    def execute1(t, y, p):
        new_y  = y.copy()
        new_p  = p.copy()
        
        new_y[0] = 3

        return new_y, new_p
    
    def execute2(t, y, p):
        new_y  = y.copy()
        new_p  = p.copy()

        new_y[0] = 0.5 + new_y[0] 

        return new_y, new_p
    
    def execute3(t, y, p):
        new_y  = y.copy()
        new_p  = p.copy()
        
        new_y[0] = 1

        return new_y, new_p
    
    def execute4(t, y, p):
        new_y  = y.copy()
        new_p  = p.copy()
        
        new_p[0] = 0.03

        return new_y, new_p
    
    def trigger_func1(t, y, p):
        return y[1] - 0.2
    
    def trigger_func2(t, y, p):
        return y[2] - 2.5
    
    def trigger_func3(t, y, p):
        return 0.2 - y[0]
    
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
    #df.to_csv('event_test_files/simulate_event_.csv', index=False)
    ################
    
    #Case 1: Timer
    print('Case 1: Timer')
    #Expect: Spike at 209
    event0  = uev.Event(name='E0', execute=execute1, trigger_func=lambda t, *args: t-209)
    # event0.remove = True
    events  = [event0]
    t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 1: Timer')
    df = pd.read_csv('event_test_files/simulate_event_1.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 2: Event
    print('Case 2: Event')
    #Expect: Spike at 25
    event0  = uev.Event(name='E0', trigger_func=trigger_func1, execute=execute1)
    events  = [event0]
    t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 2: Event')
    df = pd.read_csv('event_test_files/simulate_event_2.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 3: Event with delay
    print('Case 3: Event with delay')
    #Expect: Spike at 425 and 830
    event0  = uev.Event(delay=400, persistent=True, name='E0', trigger_func=trigger_func1, execute=execute1)
    events  = [event0]
    t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 3: Event with delay')
    df = pd.read_csv('event_test_files/simulate_event_3.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 4: Event with delay and not persistent
    print('Case 4: Event with delay and not persistent')
    #Expect: No spike
    event0  = uev.Event(delay=400, persistent=False, name='E0', trigger_func=trigger_func1, execute=execute1)
    events  = [event0]
    t, y = ivp.integrate(func, tspan, y0, p, events=events)
    
    plot(t, y, AX, 'Case 4: Event with delay and not persistent')
    df = pd.read_csv('event_test_files/simulate_event_4.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 5: Trigger at start
    print('Case 5: Trigger at start')
    #Expect: Spike at 10
    flipped = lambda *args: -trigger_func1(*args)
    event0  = uev.Event(delay=10, persistent=True, name='E0', trigger_func=flipped, execute=execute1)
    events  = [event0]
    t, y = ivp.integrate(func, tspan, y0, p, events=events)
    
    plot(t, y, AX, 'Case 5: Trigger at start')
    df = pd.read_csv('event_test_files/simulate_event_5.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 6: Multiple events
    print('Case 6: Multiple events')
    #Expect: Spike at 25 and 300
    event0  = uev.Event(delay=0, name='E0', trigger_func=trigger_func1, execute=execute1)
    event1  = uev.Event(delay=0, name='E1', trigger_func=trigger_func2, execute=execute1)
    events  = [event0, event1]
    t, y = ivp.integrate(func, tspan, y0, p, events=events)
    
    plot(t, y, AX, 'Case 6: Multiple events')
    df = pd.read_csv('event_test_files/simulate_event_6.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 7: Prioritized events
    print('Case 7: Prioritized events')
    #Expect: Spike at 800 with new value of 3.5
    event0  = uev.Event(name='E0', execute=execute2, priority=0, trigger_func=lambda t, *args: t-800)
    event1  = uev.Event(name='E1', execute=execute1, priority=1, trigger_func=lambda t, *args: t-800)
    events  = [event0, event1]
    t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 7: Prioritized events')
    df = pd.read_csv('event_test_files/simulate_event_7.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # ###############################################################################
    # #Part 1A: Other Events
    # ###############################################################################
    # y0    = np.array([1, 0, 0])
    # p     = np.array([0.01, 0.01])
    # tspan = np.linspace(0, 1000, 101)
    
    # fig  = plt.figure()
    # AX   = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    # #Case 1a: Change in parameter
    # print('Case 1a: Change in parameter')
    # #Expect: Change in rate at 200
    # event0  = uev.Event(name='E0', trigger_func=lambda t, *args: t-200, execute=execute4)
    # events  = [event0]
    # t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 1a: Change in parameter')
    # df = pd.read_csv('event_test_files/simulate_event_1a.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 2a: Assignment affects r 
    # print('Case 2a: Assignment affects r')
    # #Expect: x0 is reset cyclically
    # event0  = uev.Event(name='E0', trigger_func=trigger_func3, execute=execute3)
    # events  = [event0]
    # t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 2a: Assignment affects r ')
    # df = pd.read_csv('event_test_files/simulate_event_2a.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 3: Test reset
    # print('Case 3: Test reset')
    # #Expect: x0 is reset cyclically
    # t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True)
    
    # plot(t, y, AX, 'Case 3: Test reset')
    # df = pd.read_csv('event_test_files/simulate_event_2a.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # ###############################################################################
    # #Part 2: Integration and Formatting Options
    # ###############################################################################
    # y0    = np.array([1, 0, 0])
    # p     = np.array([0.01, 0.01])
    # tspan = np.linspace(0, 1000, 101)
    
    # fig  = plt.figure()
    # AX   = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    # #Case 8: Unprioritized events
    # print('Case 8: Unprioritized events')
    # #Expect: Spike at 900 with new value of 3
    # event0  = uev.Event(name='E0', execute=execute2, priority=0, trigger_func=lambda t, *args: t-900)
    # event1  = uev.Event(name='E1', execute=execute1, priority=1, trigger_func=lambda t, *args: t-900)
    # events  = [event0, event1]
    # t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=True, _sort=False)
    
    # plot(t, y, AX, 'Case 8: Unprioritized events')
    # df = pd.read_csv('event_test_files/simulate_event_8.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 9: Exclude events
    # print('Case 9: Exclude events')
    # #Expect: Spike at 450 with new value of 3.5. 455 is not in t.
    # event0  = uev.Event(name='E0', execute=execute2, priority=0, trigger_func=lambda t, *args: t-455)
    # event1  = uev.Event(name='E1', execute=execute1, priority=1, trigger_func=lambda t, *args: t-455)
    # events  = [event0, event1]
    # # events  = [event0]
    # t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=True, include_events=False)

    # plot(t, y, AX, 'Case 9: Exclude events')
    # t_, y_ = t, y
    # df = pd.read_csv('event_test_files/simulate_event_9.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 10: Exclude events and overlap
    # print('Case 10: Exclude events and overlap')
    # #Expect: Spike at 450 but slants instead of going to new value of 3.5. 455 is not in t.
    # event0  = uev.Event(name='E0', execute=execute2, priority=0, trigger_func=lambda t, *args: t-455)
    # event1  = uev.Event(name='E1', execute=execute1, priority=1, trigger_func=lambda t, *args: t-455)
    # events  = [event0, event1]
    # # events  = [event0]
    # t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=False, include_events=False)

    # plot(t, y, AX, 'Case 10: Exclude events and overlap')
    # df = pd.read_csv('event_test_files/simulate_event_10.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    # #Case 11: Exclude overlap
    # print('Case 11: Exclude overlap')
    # #Expect: Spike at 455. 455 appears once in t.
    # event0  = uev.Event(name='E0', execute=execute2, priority=0, trigger_func=lambda t, *args: t-455)
    # event1  = uev.Event(name='E1', execute=execute1, priority=1, trigger_func=lambda t, *args: t-455)
    # events  = [event0, event1]
    # t, y = ivp.integrate(func, tspan, y0, p, events=events, overlap=False, include_events=True)
    
    # plot(t, y, AX, 'Case 11: Exclude overlap')
    # df = pd.read_csv('event_test_files/simulate_event_11.csv')
    # answer = df.values
    # values = np.concatenate(([t], y), axis=0)
    # assert np.all( np.isclose(answer, values, rtol=rtol))
    
    ###############################################################################
    #Part 3: Dynamic Instantiation
    ###############################################################################
    #Set up
    dun_data   = dfr.read_file('event_test_files/M1.dun')
    model_data = dun_data['M1']
    func_data  = odc.make_ode_data(model_data)
    event_objs = uev.make_events(func_data, model_data)
    
    func_ = func_data['rhs'][1]
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
    
    #Case 22: Dynamically generated Event
    print('Case 22: Dynamically generated Event')
    #Expect: Spike at 160
    events_ = event_objs[1:2]
    t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 22: Dynamically generated Event')
    df = pd.read_csv('event_test_files/simulate_event_2.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 23: Dynamically generated Event with delay
    print('Case 23: Dynamically generated Event with delay')
    #Expect: Spike at 425 and 830
    events_ = event_objs[2:3]
    t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 23: Dynamically generated Event with delay')
    df = pd.read_csv('event_test_files/simulate_event_3.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 24: Dynamically generated Event with delay and not persistent
    print('Case 24: Dynamically generated Event with delay and not persistent')
    #Expect: No spike
    events_ = event_objs[3:4]
    t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 24: Dynamically generated Event with delay and not persistent')
    df = pd.read_csv('event_test_files/simulate_event_4.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 25: Dynamically generated Event with delay and not persistent
    print('Case 25: Dynamically generated Event with trigger at start')
    #Expect: Spike at 10
    events_ = event_objs[4:5]
    t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 25: Dynamically generated Event with trigger at start')
    df = pd.read_csv('event_test_files/simulate_event_5.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 26: Dynamically generated Event with multiple events
    print('Case 26: Dynamically generated Event with multiple events')
    #Expect: Spike at 25 and 300
    events_ = [event_objs[1], event_objs[5]]
    t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 26: Dynamically generated Event with multiple events')
    df = pd.read_csv('event_test_files/simulate_event_6.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    #Case 27: Dynamically generated Event with prioritized events
    print('Case 27: Dynamically generated Event with prioritized events')
    #Expect: Spike at 800 with new value of 3.5
    events_ = [event_objs[6], event_objs[7]]
    t, y = ivp.integrate(func_, tspan, y0, p, events=events_, overlap=True, include_events=True)
    
    plot(t, y, AX, 'Case 27: Dynamically generated Event with prioritized events')
    df = pd.read_csv('event_test_files/simulate_event_7.csv')
    answer = df.values
    values = np.concatenate(([t], y), axis=0)
    assert np.all( np.isclose(answer, values, rtol=rtol))
    
    