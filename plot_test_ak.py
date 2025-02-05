import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plot as pt
import data as dt
import OCV_fit



cell = "C"
test = "08"

def model_data_soc_ocv(cell, test):
    
    pt.plot_soc_ocv(cell, test)

    polynomial = dt.soc_ocv_fitted(cell, test)

    x = np.linspace(0, 1, 100)

    plt.plot(x, polynomial(x), "r--")

"""
model_data_soc_ocv(cell, test)
plt.show()
"""














def plot_soc_var(cell, test, var):
    if var == "R0":
        dt.add_R0(cell, test)
    elif var == "OCV":
        dt.plot_soc_ocv(cell, test)
        
plot_soc_var(cell, test, "R0")
plt.show()


def model_data_soc_var(cell, test, var):
    
    plot_soc_var(cell, test, var)
    
        
            
    

