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

# Example Data (Replace with your actual OCV and SoC arrays)
ocv = pt.soc_ocv("C", "06")["OCV"]

# Replace with your OCV
soc = pt.soc_ocv("C", "06")["SoC"]  # Replace with your SoC values

# Fit a polynomial of degree 2
coefficients = np.polyfit(ocv, soc, 4)
polynomial = np.poly1d(coefficients)

# Generate fitted values for plotting
ocv_range = np.linspace(min(ocv), max(ocv), 100)
fitted_soc = polynomial(ocv_range)


def soc_ocv_fitted(cell,test):
    soc = pt.soc_ocv(cell, test)["OCV"]
    ocv = pt.soc_ocv(cell, test)["SoC"] 

    # Fit a polynomial of degree 2
    coefficients = np.polyfit(ocv, soc, 4)
    polynomial = np.poly1d(coefficients)

    # Generate fitted values for plotting
    ocv_range = np.linspace(min(ocv), max(ocv), 100)
    fitted_soc = polynomial(ocv_range)
    return coefficients

def deg4_model(x, a, b, c, d, e):
    return e * x ** 4 + d * x ** 3 + c * x ** 2 + b * x + a

def calculate_ocv(soc,cell,test):
    coefficients = soc_ocv_fitted(cell,test)
    return [deg4_model(soc,coefficients[0],coefficients[1],coefficients[2],coefficients[3],coefficients[4]) for soc in soc]

# Plot the original data and the polynomial fit
print(calculate_ocv(dt.soc("C","06"),"C","06"))
print(soc_ocv_fitted("C","06"))
plt.scatter(ocv, soc, label="Data points")
plt.plot(ocv_range, fitted_soc, color="red", label=f"Polynomial fit: degree 4")
plt.title("SoC vs OCV with Polynomial Fit")
plt.xlabel("OCV (V)")
plt.ylabel("SoC (%)")
plt.legend()
plt.show()








