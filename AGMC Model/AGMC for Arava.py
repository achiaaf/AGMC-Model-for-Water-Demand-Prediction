import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import inv
import math
from Functions import grey_degree, projection, AGMC_12, AGMC_13, GMC_12, GMC_13, MAPE, mse
import matplotlib.pyplot as plt

# Importing and defining the data
data = pd.read_csv('Arava Data.csv')
Time = data['Year'].values
water_consumption = data['Water Consumption'].values
water_consumption_train = np.array(water_consumption)[:8]
gdp = data['GDP'].values
population = data['Population'].values
cost = np.array(data['Cost per m3'])

# Calculating the standard deviation and mean for the water consumption data
standard_deviation = np.std(water_consumption)
print(np.mean(water_consumption))
print(standard_deviation)
# Calculating the grey relational degree
[x, y] = grey_degree(water_consumption, gdp, population)
print(f"The grey degree between water consumption and GDP is : {round(x, 3)}")
print(f"The grey degree between water consumption and Population is : {round(y, 3)}")

# # Calculating or forecasting the water consumption
# '''Inputting year for the prediction'''
# t = 2030
#
# gdp_new = np.hstack((gdp, projection(gdp, Time, t)))
# population_new = np.hstack((population, projection(population, Time, t)))
# Time_new = np.hstack((Time, np.arange(Time[-1] + 1, t + 1, 1)))
#
# # Predicting the water consumption
# Predicted_water_consumption = AGMC_13(water_consumption_train, gdp_new, population_new, 0.009)
Predicted_water_consumption = AGMC_13(water_consumption_train, gdp, population, 0.001)
print(f"MAPE error value for the AGMC(1,3) is: {round(MAPE(water_consumption, Predicted_water_consumption), 2)}%")
# print(len(gdp))
# X = np.vstack([np.ones(len(gdp)), gdp, population])
# p = inv(X.T @ X) @ X.T @ water_consumption
# Predicted_water_consumption1 = X @ p


# plt.figure()
# plt.plot(Time, Predicted_water_consumption, color='c', marker='s', label='Modelling Values Using AGMC(1,N) Model')
# # plt.plot(Time, Predicted_water_consumption1, marker='v', label='Modelling Values Using Linear Regression Model')
# plt.plot(Time, water_consumption, marker='*', label='Actual Values')
# plt.xlabel('Time (Year)')
# plt.ylabel('Water Consumption (cubic metres)')
# plt.legend()
# plt.title('Forecasting with Adjacent Grey Model with Convolution Integral')
# plt.show()
print(np.mean(Predicted_water_consumption))
print(np.std(Predicted_water_consumption))
params = stats.norm.fit(water_consumption)
fitted_pdf = stats.norm.pdf(water_consumption, *params)
# Plot the histogram of the data
plt.hist(water_consumption, bins='auto', density=True, alpha=0.7, label='Data')

# Plot the fitted PDF
plt.plot(water_consumption, fitted_pdf, 'r-', label='Fitted PDF')

# Display the plot
plt.legend()
plt.show()
