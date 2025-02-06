import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plot as pt
import data as dt
import OCV_fit



cell = "C"
test = "0"


print(dt.soc_R1_fitted(cell, test))
