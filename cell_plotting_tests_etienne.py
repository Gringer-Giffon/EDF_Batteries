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
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of charge of D cell at a given time for the discharge at the end of the test
    Returns list of SOC values
    '''

    data = main.extract_step(21, 24, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
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
plt.plot(main.extract_step(21, 24, "D", "01")["Total Time"], soc_d("01"))
plt.show()


"""
def extract_OCV_D():
    '''
    USELESS FUNCTION

    Takes average of OCV for each test on cell D

    Returns dictionary of test number and average OCV
    '''
    extract_data = {test: 0 for test in range(0, 13+1)}
    for element in extract_data:
        print(element)
        average = 0
        if element < 10:
            step_data = extract_all_steps(
                5, 7, "D", "0"+str(element))[extract_all_steps(5, 7, "D", "0"+str(element))["Step"] == 7]
            print(step_data)
        else:
            step_data = extract_all_steps(5, 7, "D", str(element))[
                extract_all_steps(5, 7, "D", str(element))["Step"] == 7]
        for value in step_data["Voltage"]:
            average += value
        extract_data[element] = average/len(step_data["Voltage"])
    return extract_data
    '''
    extract_data = {test:0 for test in range(0,13+1)}
    for element in extract_data:
        step_data = extract_step(5, 7, "D", element)
        extract_data[element] = step_data["Voltage"].iloc[-1]
    return extract_data
    '''


def q_initial_d_precise():
    '''
    Returns initial charge for D cell
    '''
    voltage_at_time = extract("D", "00")[(extract("D", "00")[
        "Total Time"] >= 123658.7) & (extract("D", "00")["Total Time"] <= 142474.6)]
    I = abs(voltage_at_time["Current"].mean())
    t = voltage_at_time["Total Time"].iloc[-1] - \
        voltage_at_time["Total Time"].iloc[0]

    return I*t/3600


def q_initial_d():
    data = extract_step(21, 23, "D", "00")

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    return Q_remaining
"""


def soc_d_t(test, time):
    data = extract_step(21, 24, "D", test)

    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    Q_remaining = I*t/3600

    Q_available = Q_remaining - I*(data["Total Time"].iloc[time] -
                                   data["Total Time"].iloc[0])/3600

    SOC = Q_available/Q_remaining

    return SOC


def soc_d_time(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of charge of D cell at a given time for the discharge at the end of the test
    Returns list of SOC values
    '''

    data = extract_step(21, 24, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    Q_available = [(Q_remaining - I*(data["Total Time"].iloc[i] -
                                     data["Total Time"].iloc[0])/3600, data["Total Time"].iloc[i]) for i in range(len(data["Total Time"]))]

    SOC = [(Q_available[i][0]/Q_remaining, data["Total Time"].iloc[i])
           for i in range(len(data["Total Time"]))]

    """
    soc_voltage_dict = {
        data["Voltage"].iloc[i]: Q_available[i] / Q_remaining
        for i in range(len(data["Total Time"]))}
    """
    return SOC
