from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt


# constants
price = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

pv_out = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# hours
hours_in_day = 24
hours_in_window = 48

# number of sliding windows in price dataset:
num_windows = int((len(price) / hours_in_day) - 1)

# Create an array of windows
price_windows = []
pv_windows = []
for i in range(num_windows):
    # define window
    price_window = price[i*24:(i*24)+48]
    pv_window = pv_out[i*24:(i*24)+48]
    
    # append windows to lists
    price_windows.append(price_window)
    pv_windows.append(pv_window)
    
# Add last window
price_windows.append(price[-24:])
pv_windows.append(pv_out[-24:])
    

# Initializations
SOC_init = 0
R_init = 0

# Empty Lists for Storing Each Day
SOC_list = []
R_list = []

for price_window, pv_window in zip(price_windows, pv_windows):
    print('Initial Parameters:')
    print('State of Charge (MWh): {}'.format(SOC_init))
    print('Revenue ($):  {}'.format(R_init))
    
    # create GEKKO model
    m = GEKKO()
    
    price = m.Param(price_window)
    pv_out = m.Param(pv_window)
    
    # time points
    t = len(price)
    m.time = list(range(0, t))
    
    # utilization:  charge/discharge rate
    u = m.MV(fixed_initial=False)
    u.STATUS = 1
    u.DCOST = 0.001
    u.LOWER = -50
    u.UPPER = 50
    
    # Initialize Battery 
    SOC = m.Var(value=SOC_init)
    SOC.LOWER = 0
    SOC.UPPER = 50
    
    Charging_Energy = m.Var(fixed_initial=False)
    
    # System Governing Equations
    m.Equation(SOC.dt() == u)
    
    # objective (Revenue)
    R = m.Var(value=0)
    # final objective
    Rf = m.FV()
    Rf.STATUS = 1
    m.Connection(Rf,R,pos2='end')
    m.Equation(R.dt() == ((pv_out) * price) +
                         (-u * price))
    # maximize profit
    m.Obj(-Rf)
    
    # options
    m.options.IMODE = 6  # optimal control
    m.options.NODES = 100  # collocation nodes
    m.options.SOLVER = 3 # solver (IPOPT)
    
    # solve optimization problem
    m.solve(disp=False)
    
    # get additional solution information
    import json
    with open(m.path+'//results.json') as f:
        results = json.load(f)
    
    # plot results
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(m.time, R.value,'r--', label='Revenue')
    plt.plot(m.time[-1],Rf.value[0],'ro',markersize=10,\
             label='final revenue = '+str(Rf.value[0]))
    plt.axvline(x=23)
    plt.ylabel('Revenue')
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(m.time, SOC.value,'b-',label='State of Charge')
    plt.axvline(x=23)
    plt.ylabel('Value')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(m.time,u.value,'k.-',label='charge/discharge')
    plt.axvline(x=23)
    plt.ylabel('Rate')
    plt.xlabel('Time (hour)')
    plt.legend()
    plt.show()
    
    # Set values for next run
    SOC_init = SOC[23]
    SOC_window = SOC[0:24]
    
    R_init = R[23]
    R_window = R[0:24]
    
    # Append to Lists
    SOC_list.append(SOC_window)
    R_list.append(R_init)
    
    