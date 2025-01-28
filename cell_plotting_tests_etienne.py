import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframes = [pd.read_csv(file) for file in cells_data]

time = data["Total Time"]
current = data["Current"]
voltage = data["Voltage"]
step = data["Step"]

fig, axs = plt.subplots(3, 1)
axs[0].plot(time, current)
axs[1].plot(time, voltage)
axs[2].plot(time, step)

plt.show()
