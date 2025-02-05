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

    plt.plot(x, polynomial(x), "b-")


model_data_soc_ocv(cell, test)
plt.show()

        
            
    

