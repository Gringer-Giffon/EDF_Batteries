import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plot as pt
import data as dt
import OCV_fit



cell = "D"
test = "0"

def model_data_soc_ocv(cell, test):
    
    pt.plot_soc_ocv(cell, test)

    polynomial = dt.soc_ocv_fitted(cell, test)

    x = np.linspace(0, 1, 100)

    plt.plot(x, polynomial(x))

def model_data_soc_ocv_soh(cell):
    
    model_data_soc_ocv(cell, 0)
    model_data_soc_ocv(cell, 5)
    model_data_soc_ocv(cell, 11)



model_data_soc_ocv_soh("C")
                       
plt.show()

