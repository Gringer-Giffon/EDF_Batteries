import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import main


folderPath = f'./cells_data'

# dataframes = [pd.read_csv(file) for file in os.listdir(folderPath)]

"""
time = data["Total Time"]
current = data["Current"]
voltage = data["Voltage"]
step = data["Step"]
"""

print(main.extract("C", "01")["Step"])


# Q remaining is the integral of I on a full discharge
# Q available at a time tis the difference between the Q remaining and the integral of Q up to t

def soc_d(test):
    data = main.extract_step(21, 23, "D", test)
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]
    Q_remaining = I*t/3600
    print("Remaining charge: "+str(Q_remaining))

    Q_available = [Q_remaining - I*(data["Total Time"].iloc[i] -
                                    data["Total Time"].iloc[0])/3600 for i in range(len(data["Total Time"]))]
    print("Available charge" + str(Q_available))

    SOC = [Q_available[i]/Q_remaining for i in range(len(data["Total Time"]))]
    soc_voltage_dict = {
        data["Voltage"].iloc[i]: Q_available[i] / Q_remaining
        for i in range(len(data["Total Time"]))}

    return SOC, soc_voltage_dict


# if we then associate a State of Charge to voltage, it should be okay right, because there is a direct relationship between the OCV and SoC
data = main.extract_step(21, 23, "D", "01")["Voltage"]
SOC = [soc_d("01")[1][voltage] for voltage in data]

plt.plot(main.extract_step(21, 23, "D", "01")["Total Time"], SOC)
plt.show()
