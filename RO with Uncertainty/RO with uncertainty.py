import pandas as pd
import numpy as np
from numpy.linalg import norm
from rsome import ro, grb_solver as grb

# Create a minimization problem
prob = ro.Model()
data = pd.read_csv('LP.csv')
cost = data['Cost ($/m3)'][:6].values.T
capacity = data['Flow rate (m3/year)'][:6]
source_salinity = data['Salinity (dS/m)'][:6].values.T
crop_salinity = data['Optimum Salinity Tolerance (dS/m)'][:7]
crop_water_requirement = data['Irrigation Water Requirements (m3/hectare/year)'][:7].values
sources = data['Source'][:6]
crops = data['Crop'][:7]
domestic = data['Domestic Water Demand'][0]
prices = data['Revenue ($/hectare)'][:7].values
expenses = data['Expenses ($/hectare)'][:7].values
max_yield = data['Maximum Yield(ton/hectare)'][:7].values
min_quantity = data['Minimum Quantity for Local Market (ton)'][:7].values

standard_dev = 142482
demand_mean = 498250
# demand_mean = 497762
# standard_dev = 136308

# Define decision variables
q = prob.dvar((len(sources), len(crops)))  # matrix indicating the amount of water from a source allocated to a crop
d = prob.dvar(len(sources))
land = prob.dvar(len(crops))
d_uncertain = prob.rvar()
s_uncertain = prob.rvar(len(sources))

# Defining the box uncertainty set
d_set = (d_uncertain >= -1 * standard_dev, d_uncertain <= 1 * standard_dev)
# Making it an ellipsoidal uncertainty set
# d_set = (-d_uncertain, d_uncertain, norm(d_uncertain) <= 1)

s_set1 = (s_uncertain[0] >= -2 * (0.3*capacity[0]), s_uncertain[0] <= 0.5 * (0.3*capacity[0]))
# s_set = []
# for i in range(len(capacity)-1):
#     set1 = (s_uncertain[i] >= -1 * (0.1*capacity[i]), s_uncertain[i] <= 0.5 * (0.1*capacity[i]))
#     s_set.append(set1)
# s_set.append((s_uncertain[5] >= -1 * standard_dev, s_uncertain[5] <= 1 * standard_dev))
s_set = (s_uncertain >= -2 * 1000000, s_uncertain >= 0.3 * 1000000)
# Define the objective function
prob.max((prices - expenses) @ land - ((cost @ q).sum() + (cost @ d).sum()))

# Define the constraints
prob.st((d.sum() - demand_mean >= d_uncertain).forall(d_set))  # demand constraint
prob.st(q.sum(axis=0) >= crop_water_requirement * land)  # The irrigation water requirements
# for i in range(len(sources)):
#     prob.st((q.sum(axis=1)[i] + d[i] <= np.array(capacity)[i] + s_uncertain[i]).forall(s_set[i]))
# prob.st((q.sum(axis=1) + d <= np.array(capacity) + s_uncertain).forall(s_set))  # the water sources constraint
prob.st((q.sum(axis=1) + d <= np.array(capacity)))  # the water sources constraint
#
prob.st(q >= 0)  # non-negativity constraint
prob.st(d >= 0)
prob.st(land >= 5)
prob.st(land <= 100)
# prob.st(0.1 * min_quantity <= max_yield * land <= 0.6 * min_quantity)
# prob.st(land.sum() <= 500)

prob.st(d[5] == 0)
# Quality constraint
for i in range(len(crops)):
    prob.st(source_salinity @ q[:, i] <= crop_salinity[i] * q[:, i].sum())
prob.st(source_salinity @ d <= 0.5 * d.sum())

# Solve the problem
prob.solve(grb)

# # Print the optimal solution
print(f"Optimal Solution: {prob.get()}")
print(q.get())
print(d.get())
print(land.get())

# # Create a dataframe from the list
# d1 = pd.DataFrame(data=q.get(), index=sources, columns=crops)
# d2 = pd.DataFrame(data=d.get(), index=sources, columns=['Domestic Use'])
# d3 = pd.DataFrame(data=land.get(), index=crops, columns=['Land Allocated'])
# df = pd.concat([d1, d2], axis=1)
# df.to_excel('RO Uncertainty1.xlsx', sheet_name='Water Allocated')
# d3.to_excel('Land Uncertainty1.xlsx')
