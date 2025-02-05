import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plot as pt
import data as dt
import OCV_fit

cell = "D"
test = "08"

pt.plot_soc_ocv(cell, test)
SoH = dt.soh(cell, test)

df = dt.soc_ocv(cell, test)
OCV_estimated = [OCV_fit.f(soc,SoH) for soc in df["SoC"]]

df = df.sort_values(by="SoC", ascending = False)

plt.plot(df["SoC"], OCV_estimated)
plt.show()
