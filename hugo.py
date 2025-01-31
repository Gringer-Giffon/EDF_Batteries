import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plot as pt

# Load data
ocv_values = np.array(pt.soc_ocv("C", "06")["OCV"])  # OCV values
soc_values = np.array(pt.soc_ocv("C", "06")["SoC"])  # SoC values

# Ensure data is clean (remove NaNs, sort for better fitting)
ocv_values = np.nan_to_num(ocv_values)
soc_values = np.nan_to_num(soc_values)

# Define polynomial model
def dynamic_poly_model(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# Fitting parameters
target_error = 0.0001
max_order = 5
max_attempts = 10
order = 1  
error = float('inf')
fitted_params = None

# Fitting loop
for attempt in range(max_attempts):
    if order > max_order:  
        print(f"Reached max polynomial order {max_order}, stopping.")
        break

    # Generate initial guess (fallback to simple ones if polyfit fails)
    try:
        initial_guess = np.polyfit(ocv_values, soc_values, order)
    except Exception:
        initial_guess = np.ones(order + 1)  # Default to ones

    # Curve fitting
    try:
        fitted_params, _ = curve_fit(dynamic_poly_model, ocv_values, soc_values, p0=initial_guess)

        if fitted_params is not None:  
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

    order += 1  # Increase polynomial order for next attempt

# Ensure `fitted_params` is valid before plotting
if fitted_params is not None:
    # Generate smooth curve for plotting
    ocv_smooth = np.linspace(min(ocv_values), max(ocv_values), 500)
    soc_pred_smooth = dynamic_poly_model(ocv_smooth, *fitted_params)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(ocv_values, soc_values, label="Data Points", marker="+", color="gray", alpha=0.6)
    plt.plot(ocv_smooth, soc_pred_smooth, color='red', linewidth=2, label=f'Fitted Curve (Order {order})')
    plt.xlabel('OCV')
    plt.ylabel('SOC')
    plt.title('SoC vs. OCV with Polynomial Curve Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(" Curve fitting completely failed. No plot generated.")
