''' LINEAR REGRESSION FOR RANDOM GIVEN CURVE 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define a synthetic function f(time) to generate SoC-like behavior
def f(time):
    return 0.8 * np.sin(0.5 * time) + 0.2 * time + 0.1 * np.exp(-0.2 * time) + 0.1 * np.exp(0.02 * time)

# Generate synthetic time values
time_values = np.linspace(0, 10, 100)  # Time from 0 to 10 (acts like OCV)
soc_values = f(time_values)  # Compute SoC values based on function f(time)

# Dynamic polynomial model
def dynamic_poly_model(time, *coeffs):
    return sum(c * time**i for i, c in enumerate(coeffs))

# Fitting parameters
target_error = 0.001
max_order = 8
max_attempts = 25
order = 1  
error = float('inf')
fitted_params = None

for attempt in range(max_attempts):
    if order > max_order:  
        print(f"Reached max polynomial order {max_order}, stopping.")
        break

    # Use np.polyfit for better initial guesses
    initial_guess = np.polyfit(time_values, soc_values, order)

    # Curve fitting
    try:
        fitted_params, _ = curve_fit(dynamic_poly_model, time_values, soc_values, p0=initial_guess)
        soc_pred = dynamic_poly_model(time_values, *fitted_params)
        error = np.mean((soc_values - soc_pred) ** 2)

        print(f"Attempt: {attempt + 1}, Order: {order}")
        print(f"Coefficients: {fitted_params}")
        print(f"Error: {error:.6f}\n")

        if error <= target_error:
            print(f"Target error reached at order {order}!\n")
            break

    except Exception as e:
        print(f"Curve fitting failed at order {order}: {e}")
    
    order += 1  # Increase order for next attempt

# Generate high-resolution curve
time_smooth = np.linspace(min(time_values), max(time_values), 500)
soc_pred_smooth = dynamic_poly_model(time_smooth, *fitted_params)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(time_values, soc_values, color='gray', label='Generated Data', alpha=0.6)
plt.plot(time_smooth, soc_pred_smooth, color='red', linewidth=2, label=f'Fitted Curve (Order {order})')
plt.xlabel('Time')
plt.ylabel('SOC')
plt.title('Synthetic SoC vs. Time with Polynomial Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plot as pt 

cell = "D"
test = "00"

# Load data
ocv_values = np.array(pt.soc_ocv(cell, test)["OCV"])  
soc_values = np.array(pt.soc_ocv(cell, test)["SoC"])  

# Dynamic polynomial model
def dynamic_poly_model(time, *coeffs):
    return sum(c * time**i for i, c in enumerate(coeffs))

# Error threshold and iteration setup
target_error = 0.001  
max_order = 7  # Maximum polynomial order
max_attempts = 10  # Maximum attempts
order = 1  # Start with a linear model
error = float('inf')

for attempt in range(max_attempts):  # Ensures loop doesn't run infinitely
    if order > max_order:  
        print(f"Reached max polynomial order {max_order}, stopping.")
        break

    initial_guess = [1] * (order + 1)

    # Curve fitting using current polynomial order
    try:
        fitted_params, _ = curve_fit(dynamic_poly_model, ocv_values, soc_values, p0=initial_guess)
        soc_pred = dynamic_poly_model(ocv_values, *fitted_params)
        error = np.mean((soc_values - soc_pred) ** 2)

        # Print progress
        print(f"Attempt: {attempt + 1}, Order: {order}")
        print(f"Coefficients: {fitted_params}")
        print(f"Error: {error:.6f}\n")

        # Stop if the error is low enough
        if error <= target_error:
            print(f"Target error threshold reached with order {order}!\n")
            break

    except Exception as e:
        print(f"Curve fitting failed at order {order}: {e}")
    
    # Increase order for the next attempt
    order += 1  

# Sorting for smooth curve plotting
sorted_indices = np.argsort(ocv_values)
ocv_sorted = ocv_values[sorted_indices]
soc_pred_sorted = soc_pred[sorted_indices]

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(ocv_values, soc_values, color='gray', label='Measured SoC Data')
plt.plot(ocv_sorted, soc_pred_sorted, color='red', linestyle='--', label='Fitted Curve')
plt.xlabel('OCV')
plt.ylabel('SOC')
plt.title('SoC vs. OCV with Polynomial Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()



"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plot as pt 

cell = "D"
test = "00"
# Simulated "measured" voltage vs. time data
ocv_values = pt.soc_ocv(cell, test)["OCV"]
soc_values = pt.soc_ocv(cell, test)["SoC"]
#soc_values += np.random.normal(0, 0.5, len(ocv_values))  # Add unknown noise (simulated real measurement)

# Dynamic polynomial model
def dynamic_poly_model(time, *coeffs):
    return sum(c * time**i for i, c in enumerate(coeffs))

# Error threshold and iteration setup
target_error = 0.05  # Target MSE (adjustable)
max_order = 7
attempt = 0
order = 1
error = float('inf')

while attempt <= 10:
    attempt = attempt + 1
    while error > target_error:
        initial_guess = [1] * (order + 1)
    
        # Curve fitting using current polynomial order
        fitted_params, _ = curve_fit(dynamic_poly_model, ocv_values, soc_values, p0=initial_guess)
        
        # Generate predictions
        voltage_pred = dynamic_poly_model(ocv_values, *fitted_params)
        
        # Compute error (Mean Squared Error)
        error = np.mean((soc_values - voltage_pred) ** 2)
        
        # Print the current polynomial order, coefficients, and error
        print(f"Order: {order}")
        print(f"Coefficients: {fitted_params}")
        print(f"Error: {error:.6f}\n")
        
        if error <= target_error:
            print(f"Target error threshold reached with order {order}!\n")
            break

# Plot the results
plt.figure(figsize=(10, 6))
#plt.scatter(ocv_values, soc_values, color='gray', label='Measured Voltage Data')
plt.plot(soc_values,ocv_values)
plt.xlabel('OCV')
plt.ylabel('SOC')
plt.title('Soc vs. Ocv with Polynomial Curve Fitting')
plt.legend()
plt.show()
"""