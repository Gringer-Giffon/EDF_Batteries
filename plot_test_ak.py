import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

directory = f"./cells_data"

# extracts all csv file names in directory
csv_files = [f for f in os.listdir(directory)]

# extract function to extract all data


def plot_test(cell, test, time_start, time_end):
    '''
    Parameters: 
        cell (string) C or D, 
        test (string) in the form 00, 01, etc...
        time_start, time_end: (float) Time range to zoom into

    Plots voltage, current, and step for given test.
    '''

    plot_data = extract(cell, test)
    if plot_data is None:
        return

    # Convert time column to numeric type (if necessary)
    plot_data["Total Time"] = pd.to_numeric(plot_data["Total Time"], errors="coerce")

    # Filter the data for the specified time range
    zoom_data = plot_data[(plot_data["Total Time"] >= time_start) & 
                          (plot_data["Total Time"] <= time_end)]

    if zoom_data.empty:
        print("No data found in the specified time range.")
        return

    # Data extraction from dataframe
    time = zoom_data["Total Time"]
    current = zoom_data["Current"]
    voltage = zoom_data["Voltage"]
    step = zoom_data["Step"]

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))

    fig.suptitle(f"Cell: {cell.upper()}, Test: {test} (Time {time_start}-{time_end}s)") 

    axs[0].plot(time, current, "g")
    axs[0].set_title("Current")
    axs[1].plot(time, voltage, "b")
    axs[1].set_title("Voltage")
    axs[2].plot(time, step, "r")
    axs[2].set_title("Step")

    plt.xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.5)  # Adjust space between plots

    return



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

if __name__ == '__main__':
    plot_test("D", "01", 120000, 140000)
    plt.show()

