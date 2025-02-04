"""
#_______________________________________________________________________

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plot as pt
from sklearn.linear_model import LinearRegression



# Define the model function

def constant_model(x, a):
    return a

def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x ** 2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x * d



def deg5_model(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

def deg6_model(x, a, b, c, d, e, f, g):
    return a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2 + f * x + g

def deg7_model(x, a, b, c, d, e, f, g, h):
    return a * x ** 7 + b * x ** 6 + c * x ** 5 + d * x ** 4 + e * x ** 3 + f * x ** 2 + g * x + h

def deg8_model(x, a, b, c, d, e, f, g, h, i):
    return a * x ** 8 + b * x ** 7 + c * x ** 6 + d * x ** 5 + e * x ** 4 + f * x ** 3 + g * x ** 2 + h * x + i


polynomials = [constant_model, linear_model, quadratic_model, 
               cubic_model, deg4_model, deg5_model, 
               deg6_model, deg7_model, deg8_model]



# Generate some data
print(pt.soc_ocv("C","06"))
x_data = pt.soc_ocv("C","06")["OCV"]

y_data = pt.soc_ocv("C","06")["SoC"]

# Fit the model to the data
params, covariance = curve_fit(cubic_model, x_data, y_data)

y_data_check = params[3] + params[2] * x_data + params[1] * x_data**2 + params[0] * x_data**3

# Output the parameters
print("Fitted parameters:", params)
plt.plot(x_data, y_data, "ro")
plt.plot(x_data, y_data_check, "b--")
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import plot as pt
import data as dt
import pandas as pd
import os
import Tianhe_csvPlot as ti
import math


# Plot entire voltage
# plt.plot(dt.extract("D", "01")["Total Time"],dt.extract("D", "01")["Voltage"], "x")
# plt.show()

def extract_pulse(cell,test):
    '''
    Parameters: cell (string), test(string) test number "01","09",10", etc..

    Returns the dataframe of a spike in the middle of the pulse region of selected data
    '''
    df_pre = dt.extract(cell,test)
    df = df_pre[abs(df_pre["Current"])== max(df_pre["Current"])] # Selects all pulse indexes
    pulse_time = df["Total Time"].iloc[math.floor(len(df)/2)] # Selects an index in the middle of the data
    # Selection of frame of impulse
    if cell == "C":  
        pulse_df = df_pre[(df_pre["Total Time"] >= pulse_time -20) & (df_pre["Total Time"] <= pulse_time + 15)]
    elif cell == "D":
        pulse_df = df_pre[(df_pre["Total Time"] >= pulse_time -200) & (df_pre["Total Time"] <= pulse_time + 150)]
    else:
        print("Invalid choice of cell")
        return None
    return pulse_df

# Extract one pulse
cell = input("which cell would you like to analyse: ")
test = input("which test would you like to analyse: ")
df = extract_pulse(cell,test)

#df = df[(df["Total Time"] >= 56500) & (df["Total Time"] <= 56620)]

plt.plot(df["Total Time"], df["Voltage"], "x")
plt.show()


def spike_index(pulse):
    '''
    Parameters: pulse (dataframe) data of your selected pulse

    Returns the index of the pulse spike
    '''
    voltage_diff = np.diff(pulse["Voltage"].values)

    # threshold for detection
    threshold = 0.05  

    # index of spike
    spike_index = np.argmax(np.abs(voltage_diff) > threshold)

    return spike_index


# Oth order
spike = spike_index(df)
U_ocv = df["Voltage"].iloc[spike-1]

R0 = abs(U_ocv - df["Voltage"].iloc[spike+1])/abs(df["Current"].iloc[spike+1])
print("R0", R0)

model_voltage_0 = [(U_ocv - R0*abs(df["Current"].iloc[i])) for i in range(len(df))]
print(model_voltage_0)

#plt.plot(df["Total Time"],model_voltage_0,'b')
plt.plot(df["Total Time"], df["Voltage"], 'r')

# 1st order non fitted
R1 = abs(df["Voltage"].iloc[spike+1] - min(df["Voltage"])) / \
    abs(df["Current"][df["Voltage"] == min(df["Voltage"])]).iloc[0]
print("R1", R1)
target_voltage = df["Voltage"].iloc[spike+1] - 0.63 * \
    abs(df["Voltage"].iloc[spike+1]-min(df["Voltage"]))

idx = (df["Voltage"] - target_voltage).abs().idxmin()

tau = (df["Total Time"].loc[idx] - df["Total Time"].iloc[0])

print(tau)
print("voltages", df["Voltage"])
print("correct idx voltage,", df["Voltage"].loc[idx])

model_voltage_1 = [(model_voltage_0[i] - R1 * abs(df["Current"].iloc[i])*(1-np.exp(-(
    df["Total Time"].iloc[i]-abs(df["Total Time"].iloc[0]))/tau))) for i in range(len(df))]

print(df)
print(model_voltage_1)
plt.plot(df["Total Time"], model_voltage_1, "b")
plt.show()





# 1st order fitted

from scipy.optimize import curve_fit

def battery_model(t, R0, R1, tau, U_ocv):
    current = np.abs(df["Current"].values)  # Ensure the current is positive
    return U_ocv + R0 * current + R1 * current * (1 - np.exp(-t / tau))

print([R0, R1, tau, U_ocv])

popt, _ = curve_fit(battery_model, df["Total Time"]-df["Total Time"].iloc[0], df["Voltage"], p0=[R0, R1, tau, U_ocv])

R0_fit, R1_fit, tau_fit, U_ocv_fit = popt

model_voltage_fit = battery_model(df["Total Time"]-df["Total Time"].iloc[0], *popt)
print(popt)
print(model_voltage_fit)
#plt.plot(df["Total Time"], df["Voltage"], 'r', label='Data')
#plt.plot(df["Total Time"], model_voltage_fit, 'b', label='Fitted Model')
#plt.plot(df['Total Time'],model_voltage_0,'g')
#plt.legend()
#plt.show()



# Error estimation

def calculate_distance(voltages):
    error= 0
    errors = []
    for i in range(len(voltages)):
        error+= np.sqrt(voltages[i]**2 - df["Voltage"].iloc[i]**2)
        errors.append(error)
    return errors

plt.plot(df["Total Time"],calculate_distance(model_voltage_0),'r')
plt.plot(df["Total Time"], calculate_distance(model_voltage_1),'g')
plt.show()