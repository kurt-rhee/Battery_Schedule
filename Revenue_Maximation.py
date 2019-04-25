from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# create GEKKO model
m = GEKKO()

# constants
Price = m.Param([0, 10, 1, 1, 1, 10, 10, 0])

# time points
t = len(Price)
m.time = list(range(0, t))

# utilization:  charge/discharge rate
u = m.MV(lb=-50, ub=50, fixed_initial=False)
u.STATUS = 1
u.DCOST = 0

# Initialize Battery 
SOC = m.Var(value=0)
SOC.LOWER = 0

# State of Charge
m.Equation(SOC.dt() == u)

# objective (Revenue)
R = m.Var(value=0)
# final objective
Rf = m.FV()
Rf.STATUS = 1
m.Connection(Rf,R,pos2='end')
m.Equation(R.dt() == -u * Price)
# maximize profit
m.Obj(-Rf)

# options
m.options.IMODE = 6  # optimal control
m.options.NODES = 10  # collocation nodes
m.options.SOLVER = 3 # solver (IPOPT)

# solve optimization problem
m.solve(debug=True)

# print profit
print('Optimal Profit: ' + str(Rf.value[0]))

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
plt.ylabel('Revenue')
plt.legend()
plt.subplot(3,1,2)
plt.plot(m.time, SOC.value,'b-',label='State of Charge')
plt.ylabel('Value')
plt.legend()
plt.subplot(3,1,3)
plt.plot(m.time,u.value,'k.-',label='charge/discharge')
plt.ylabel('Rate')
plt.xlabel('Time (yr)')
plt.legend()
plt.show()
