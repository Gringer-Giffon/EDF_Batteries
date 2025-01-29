import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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


if __name__ == '__main__':
    '''
    test1 = extract("D", "04")
    test2 = extract("D","13")
    test3 = extract("D",12)

    plt.plot(test1["Total Time"], test1["Voltage"])
    plt.plot(test2["Total Time"], test2["Voltage"])
    plt.plot(test3["Total Time"], test3["Voltage"])
    plt.legend(["Test 1", "Test 2","Test 3"])
    '''
    # plot_step(7, 8, "D", "01")
