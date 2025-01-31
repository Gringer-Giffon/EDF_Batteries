"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plot as pt
import data as dt

#now I want to be able to plot the SOC-OCV curve very well

ocv_values = pt.soc_ocv("C", "06")["OCV"]# Replace with your OCV
soc_values = pt.soc_ocv("C", "06")["SoC"]  # Replace with your SoC values

# Fitting parameters
target_error = 0.001
max_order = 8
max_attempts = 25
order = 1  
error = float('inf')
fitted_params = None

# Fitting code
for attempt in range(max_attempts):
    if order > max_order:  
        print(f"Reached max polynomial order {max_order}, stopping.")
        break

    # Use np.polyfit for better initial guesses
    initial_guess = np.polyfit(ocv_values, soc_values, order)

    # Curve fitting
    try:
        fitted_params, _ = curve_fit(dynamic_poly_model, ocv_values, soc_values, p0=initial_guess)
        soc_pred = dynamic_poly_model(ocv_values, *fitted_params)
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

# Dynamic polynomial model
def dynamic_poly_model(time, *coeffs):
    return sum(c * time**i for i, c in enumerate(coeffs))

# Generate high-resolution curve
time_smooth = np.linspace(min(ocv_values), max(ocv_values), 500)
soc_pred_smooth = dynamic_poly_model(time_smooth, *fitted_params)

# Plot results
plt.figure(figsize=(10, 6))
#plt.scatter(ocv_values, soc_values, color='gray', label='Generated Data', alpha=0.6)

#plot the data 
plt.scatter(ocv_values, soc_values, label="Data points", marker= "+" ) 

#plot the fitted curve 
plt.plot(time_smooth, soc_pred_smooth, color='red', linewidth=2, label=f'Fitted Curve (Order {order})') 

plt.xlabel('Time')
plt.ylabel('SOC')
plt.title('Synthetic SoC vs. Time with Polynomial Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()

"""


# LINEAR REGRESSION FOR RANDOM GIVEN CURVE 
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

