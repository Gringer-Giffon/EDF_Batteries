import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



# Define the model function

def constant_model(x, a):
    return a

def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x ** 2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x * d

def deg4_model(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def deg5_model(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

def deg6_model(x, a, b, c, d, e, f, g):
    return a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2 + f * x + g

def deg7_model(x, a, b, c, d, e, f, g, h):
    return a * x ** 7 + b * x ** 6 + c * x ** 5 + d * x ** 4 + e * x ** 3 + f * x ** 2 + g * x + h

def deg8_model(x, a, b, c, d, e, f, g, h, i):
    return a * x ** 8 + b * x ** 7 + c * x ** 6 + d * x ** 5 + e * x ** 4 + f * x ** 3 + g * x ** 2 + h * x + i


def find_deg_pol_min_cov(covariances):
    """
    Parameters: covariances (list): list of covariances of each polynomial's curve fitting 

    Returns: min_error_index (int): the index of the covariances list in which its covariance is the minimum
    """
    
    min_error = min(covariances)
    min_error_index = 0
    
    for i in range(len(covariances)):
        if min_error == covariances[i]:
            min_error_index = i
            break
    
    return min_error_index

def curve_fit_all_pols(pols, x_data, y_data):
    
    """
    Description: generalize the curve_fit() function to apply to all the polynomials
    Parameters: 
        pols (list): list of polynomials we want to use to apply curve_fit
        x_data and y_data being the coordinate of samples
        
    Returns:
        [params_pols covariance_pols], which is a list of lists, params_pols an covariance_pols 
        giving the parameters and covariance, respectively, of all polynomials
    """
    
    params_pols = []
    covariance_pols = []
    
    for pol in pols:
        params, covariance = curve_fit(pol, x_data, y_data)
        params_pols.append(params)
        covariance_pols.append(covariance)
    
    return params_pols, covariance_pols
    

polynomials = [constant_model, linear_model, quadratic_model, 
               cubic_model, deg4_model, deg5_model, 
               deg6_model, deg7_model, deg8_model]

# Generate some data
x_data = np.linspace(0, 10, 50)

y_data = 2.5 * x_data + 1.0 + np.random.normal(0, 1, 50)

# Fit the model to the data
params, covariance = curve_fit(linear_model, x_data, y_data)

y_data_check = params[0] * x_data + params[1] 

# Output the parameters
print("Fitted parameters:", params)
plt.plot(x_data, y_data, "ro")
plt.plot(x_data, y_data_check, "b--")
plt.show()
    