import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

directory = f"./cells_data"

# extracts all csv file names in directory
csv_files = [f for f in os.listdir(directory)]

# extract function to extract all data


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_test(cell, test):
    '''
    Parameters:
        cell (string) C or D,
        test (string) in the form 00, 01, etc..

    Plots voltage, current, and step for given test.
    Adds a slider to adjust a vertical line dynamically.

    Returns none.
    '''

    plot_data = extract(cell, test)
    time = plot_data["Total Time"]
    current = plot_data["Current"]
    voltage = plot_data["Voltage"]
    step = plot_data["Step"]

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    fig.suptitle(f"Cell: {cell.upper()}, Test: {test}")  

    axs[0].plot(time, current, "g")
    axs[0].set_title("Current")
    axs[1].plot(time, voltage, "g")
    axs[1].set_title("Voltage")
    axs[2].plot(time, step, "g")
    axs[2].set_title("Step")

    # Initial vertical line at the middle of the time range
    vline_x = time.iloc[len(time) // 2]  
    vlines = [ax.axvline(x=vline_x, color='r', linestyle='--') for ax in axs]

    # Adjust layout to make space for the slider
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, hspace=1)

    # Slider axis and control
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, "Time", time.min(), time.max(), valinit=vline_x)

    def update(val):
        """ Updates the vertical line when the slider moves """
        vline_x = slider.val
        for vline in vlines:
            vline.set_xdata([vline_x])
        fig.canvas.draw_idle()  # This ensures the update is shown immediately

    # Connect slider update function
    slider.on_changed(update)

    plt.show()


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

def soc_d(test):
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

    Q_available = [Q_remaining - I*(data["Total Time"].iloc[i] -
                                    data["Total Time"].iloc[0])/3600 for i in range(len(data["Total Time"]))]

    SOC = [Q_available[i]/Q_remaining for i in range(len(data["Total Time"]))]

    """
    soc_voltage_dict = {
        data["Voltage"].iloc[i]: Q_available[i] / Q_remaining
        for i in range(len(data["Total Time"]))}
    """

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

def find_OCV(cell, test):
    """
    Parameters: cell (str) C or D, 
                test (str) in the form of 01, 02, etc...
                
    Returns a list of different times that the circuit has reached OCV
    """
    
    data = extract(cell, test)[extract(cell,test)["Step"] == 5]
    
    
    
    print(data)
    data_voltage = data["Voltage"]
    
    total_time = len(data_voltage)

              



if __name__ == '__main__':    
    
    find_OCV("D", "03")
    
    plot_test("D", "03")
    plot_test("D", "04")
    plt.show()
    
    
    """
    # plt.plot(extract_step(21, 24, "D", "01")["Total Time"], soc_d("01"))
    data = extract("D", "00")
    # print(data.loc[(data["Total Time"] > 142460) & (data["Total Time"] < 142480)])
    # plot_test("D", "01")
    # plot_test("D","13")
    soh_s = []
    for i in range(0, 13+1):
        if i < 10:
            soh_s.append(soh("0"+str(i)))
        else:
            soh_s.append(soh(str(i)))
    plt.plot(list(range(0, 13+1)), soh_s)
    plt.show()
    """
