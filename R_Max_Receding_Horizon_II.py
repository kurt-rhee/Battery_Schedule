from gekko import GEKKO

import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd

# constants
price = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

pv_out = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

df = pd.read_csv('Test.csv')
print(df.head())

price = df['Price']
pv_out = df['Gen']

price = list(price)
pv_out = list(pv_out)



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

iterator = 1
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
    
    # Initialize Battery 
    SOC = m.Var(value=SOC_init)
    SOC.LOWER = 0
    SOC.UPPER = 160
    
    charging_energy = m.MV(fixed_initial=False)
    charging_energy.STATUS = 1
    charging_energy.DCOST = 0.001
    charging_energy.LOWER = 0
    charging_energy.UPPER = 41
    
    discharging_energy = m.MV(fixed_initial=False)
    discharging_energy.STATUS = 1
    discharging_energy.DCOST = 0.001
    discharging_energy.LOWER = 0
    discharging_energy.UPPER = 41
    
    # System Governing Equations
    m.Equation(charging_energy <= pv_out)
    m.Equation(discharging_energy <= SOC)
    m.Equation(SOC.dt() == charging_energy - discharging_energy)
    
    # objective (Revenue)
    R = m.Var(value=0)
    # final objective
    Rf = m.FV()
    Rf.STATUS = 1
    m.Connection(Rf,R,pos2='end')
    m.Equation(R.dt() == ((pv_out - charging_energy) * price) +
                         (discharging_energy * price))
    # maximize profit
    m.Obj(-Rf)
    
    # options
    m.options.IMODE = 6  # optimal control
    m.options.NODES = 100  # collocation nodes
    m.options.SOLVER = 3 # solver (IPOPT)
    
    # solve optimization problem
    m.solve(disp=False)
    
    # Set values for next run
    SOC_init = SOC[23]
    SOC_window = SOC[0:24]
    
    R_init = R[23]
    R_window = R[0:24]
    
    # Append to Lists
    SOC_list.append(SOC_window)
    R_list.append(R_init)
    
    
    # plot results
    plt.figure(iterator)
    plt.subplot(2,1,1)
    plt.plot(m.time, R.value,'r--', label='Revenue')
    plt.axvline(x=23)
    plt.ylabel('Revenue')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(m.time, SOC.value,'b-',label='State of Charge')
    plt.axvline(x=23)
    plt.ylabel('Value')
    plt.legend()
    display(plt.figure(iterator))
    plt.close()
    
    iterator += 1
    