import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import main
import Tianhe_csvPlot as tianhe

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
    '''
    Parameters: test (string) in the form 00, 01, etc..
    
    Calculates the state of charge of D cell at a given time for the discharge at the end of the test
    Returns list of SOC values
    '''

    data = main.extract_step(21, 24, "D", test)

    #Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    #Calculate Q remaining and Q available
    Q_remaining = I*t/3600
    

    Q_available = [Q_remaining - I*(data["Total Time"].iloc[i] -
                                    data["Total Time"].iloc[0])/3600 for i in range(len(data["Total Time"]))]

    
    SOC = [Q_available[i]/Q_remaining for i in range(len(data["Total Time"]))]
    
    """
    soc_voltage_dict = {
        data["Voltage"].iloc[i]: Q_available[i] / Q_remaining
        for i in range(len(data["Total Time"]))}
    """

    return SOC


# if we then associate a State of Charge to voltage, it should be okay right, because there is a direct relationship between the OCV and SoC
data = main.extract("D", "01")["Voltage"]
SOC = []
"""
for element in data:
    if element in soc_d("01")[1].keys():

        SOC.append(soc_d("01")[1][element])
    else:
        pass
"""


tianhe.c_locate_ABCD_n()





plt.plot(main.extract_step(21,24,"D", "01")["Total Time"], soc_d("01"))  
plt.show()

