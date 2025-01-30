from scipy.integrate import cumulative_trapezoid
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


directory = f"./cells_data"

# extracts all csv file names in directory
csv_files = [f for f in os.listdir(directory)]

# extract function to extract all data


def plot_test(cell, test):
    '''
    Parameters: cell (string) C or D, test (string) in the form 00, 01, etc..

    Plots voltage current and step for given test
    Does not show plot

    Returns none
    '''

    plot_data = extract(cell, test)
    # Data extraction from dataframe
    time = plot_data["Total Time"]
    current = plot_data["Current"]
    voltage = plot_data["Voltage"]
    step = plot_data["Step"]

    # Plotting
    fig, axs = plt.subplots(3, 1)

    fig.suptitle("Cell: "+cell.upper()+",test: "+test)  # main title

    axs[0].plot(time, current, "g")
    axs[0].set_title("Current")
    axs[1].plot(time, voltage, "g")
    axs[1].set_title("Voltage")
    axs[2].plot(time, step, "g")
    axs[2].set_title("Step")

    plt.subplots_adjust(hspace=1)  # adjust space between plots

    return None


def extract(cell, test):
    '''
    Parameters : cell (string) C or D, test (string) in the form 00, 01, etc..

    Extracts raw data from csv files corresponding to given cell and test
    Returns dataframe of extracted data
    '''

    file = [pd.read_csv(os.path.join(directory, f))
            for f in csv_files if '_'+(str(cell).upper())+"_" in f and str(test) in f]
    data = pd.concat(file)  # dataframe of corresponding csv data

    if file == []:
        print("No test found for given cell and test. Cell entry must be C/c or D/d")
        return None

    return data


def extract_all_steps(first, second, cell, test):
    '''
    Parameters: first (int) first step, second (int) second step, cell (string) C or D, test (string) in the form 00, 01, etc..

    Returns dataframe for given step interval, does not remove duplicate step sequences
    '''

    data = extract(cell, test)
    step_data = data[data["Step"].isin(list(range(first, second+1)))]
    return step_data


def extract_step(first, second, cell, test):
    '''
    Parameters: first (int) first step, second (int) second step, cell (string) C or D, test (string) in the form 00, 01, etc..

    Returns dataframe for given step interval
    '''

    data = extract(cell, test)
    step_data = data[data["Step"].isin(list(range(first, second+1)))]

    # Remove duplicate step sequences
    step_indices = step_data.index
    for i in range(1, len(step_indices)):
        # Check for breaks in the sequence
        if step_indices[i] != step_indices[i - 1] + 1:
            # Keep only the first block
            step_data = step_data.loc[step_indices[:i]]
            break
    return step_data


def plot_step(first, second, cell, test):
    '''
    Parameters: first (int) first step, second (int) second step, cell (string) C or D, test (string) in the form 00, 01, etc..

    Plots voltage current and step for given step interval
    Returns None
    '''

    # Extracting data
    step_data = extract_step(first, second, cell, test)
    time = step_data["Total Time"]
    current = step_data["Current"]
    voltage = step_data["Voltage"]
    step = step_data["Step"]

    # Plotting
    fig, axs = plt.subplots(3, 1)

    fig.suptitle("Cell: "+cell.upper()+",test: "+test)  # main title

    axs[0].plot(time, current, "g")
    axs[0].set_title("Current")
    axs[1].plot(time, voltage, "g")
    axs[1].set_title("Voltage")
    axs[2].plot(time, step, "g")
    axs[2].set_title("Step")

    plt.subplots_adjust(hspace=1)  # adjust space between plots

    return None


def q_initial_d():
    data = extract_step(21, 23, "D", "00")

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    return Q_remaining


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


def soc_full_c(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Returns the list of SOC values for cell C
    '''
    data_full = extract("C", test)
    data = extract_step(26, 27, "C", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600
    Q_available = [Q_remaining + (cumulative_trapezoid(data_full["Current"], data_full["Total Time"], initial=0)[
                                  i])/3600 for i in range(len(cumulative_trapezoid(data_full["Current"], data_full["Total Time"], initial=0)))]

    SOC = [(Q_available[i]/Q_remaining) - min(Q_available) /
           Q_remaining for i in range(len(Q_available))]

    return SOC


def soc_full_d(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Returns list of SOC for cell D
    '''

    data_full = extract("D", test)
    data = extract_step(21, 23, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    Q_available = [Q_remaining + (cumulative_trapezoid(data_full["Current"], data_full["Total Time"], initial=0)[
                                  i])/3600 for i in range(len(cumulative_trapezoid(data_full["Current"], data_full["Total Time"], initial=0)))]

    SOC = [(Q_available[i]/Q_remaining) - min(Q_available) /
           Q_remaining for i in range(len(Q_available))]

    return SOC


def soc_d(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of charge of D cell at a given time for the discharge at the end of the test
    Returns list of SOC values in the full discharge phase for cell D
    '''

    data = extract_step(21, 24, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    Q_available = [Q_remaining - I*(data["Total Time"].iloc[i] -
                                    data["Total Time"].iloc[0])/3600 for i in range(len(data["Total Time"]))]

    SOC = [Q_available[i]/Q_remaining for i in range(len(data["Total Time"]))]

    return SOC


def soh(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of health of a cell at a given time 
    Returns SOH value of test
    '''

    data = extract_step(21, 23, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining
    Q_remaining = I*t/3600
    q_initial = q_initial_d()

    SOH = Q_remaining / q_initial
    print(Q_remaining, q_initial)
    return SOH


def plot_soh_d():
    '''
    Plots the SOH for cell D
    Returns nothing
    '''

    soh_s = []
    for i in range(0, 13+1):
        if i < 10:
            soh_s.append(soh("0"+str(i)))
        else:
            soh_s.append(soh(str(i)))
    plt.plot(list(range(0, 13+1)), soh_s)


def ocv_voltage():
    soc_d_t_list = soc_d_time("01")
    print(soc_d_t_list)
    soc_ocv_v = {0.05*i for i in range(0, 21)}
    for element in soc_ocv_v:
        for pair in soc_d_t_list:
            if element == pair[0]:
                soc_ocv_v[element] = extract("D", "01")["Voltage"][extract("D", "01")[
                    "Total Time"] == pair[1]]
    print(soc_ocv_v)
    return None


if __name__ == '__main__':
    data = extract("D", "02")
    soc = soc_full_d("02")
    plt.plot(data["Total Time"], soc)
    plot_test("D", "02")
    # ocv_voltage()
    plt.show()

    
    # look at C







