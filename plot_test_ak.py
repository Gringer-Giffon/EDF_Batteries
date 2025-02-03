import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plot as pt


# Generate some data
#x_data = np.linspace(0, 10, 50)

#y_data = 2.5 * x_data + 1.0 + np.random.normal(0, 1, 50)

pt.plot_soh("D")



"""
# Fit the model to the data
params, covariance = curve_fit(linear_model, x_data, y_data)

y_data_check = params[0] * x_data + params[1] 

# Output the parameters
print("Fitted parameters:", params)
plt.plot(x_data, y_data, "ro")
plt.plot(x_data, y_data_check, "b--")
plt.show()
    """
    

