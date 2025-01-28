import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\User\modelling_weeks_2\battery_project\cells_data\CELL_C_TEST_00.csv")

def extract():
    
time = data["Total Time"]
current = data["Current"]
voltage = data["Voltage"]
step = data["Step"]

fig, axs = plt.subplots(3,1)
axs[0].plot(time, current)
axs[1].plot(time,voltage)
axs[2].plot(time,step)

plt.show()