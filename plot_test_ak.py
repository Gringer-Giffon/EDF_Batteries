import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plot as pt
import data as dt
import OCV_fit



cell = "C"
test = "0"


pt.plot_r1_soc(cell, test)

plt.show()

#df = dt.calculate_model_voltage_0(cell, test)  # Compute 0th-order model
#filtered_df = df[(df['Total Time'] >= 1200) & (df['Total Time'] <= 1400)]

#print(df)
                       

