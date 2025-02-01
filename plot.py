import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import data as dt

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

    plot_data = dt.extract(cell, test)
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


def plot_step(first, second, cell, test):
    '''
    Parameters: first (int) first step, second (int) second step, cell (string) C or D, test (string) in the form 00, 01, etc..

    Plots voltage current and step for given step interval
    Returns None
    '''

    # Extracting data
    step_data = dt.extract_step(first, second, cell, test)
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


def plot_soh(cell):
    '''
    Plots the SOH for cell D
    Returns nothing
    '''

    soh_s = []
    for i in range(0, 13+1):
        if i < 10:
            soh_s.append(dt.soh(cell, "0"+str(i)))
        else:
            soh_s.append(dt.soh(cell, str(i)))

    # Plotting
    plt.plot(list(range(0, 13+1)), soh_s)
    plt.xlabel("Test Number")
    plt.ylabel("SOH (%)")
    plt.title("State of Health vs Test Number")


def plot_soc(cell, test):
    '''
    Parameters : cell (string) C or D, test (string) in the form 00, 01, etc..

    Plots the state of charge of given cell for the given test
    Returns nothing
    '''

    soc = dt.soc(cell, test)  # list of soc values

    # Conversion to percentage
    soc = [value*100 for value in soc]

    df = dt.extract(cell, test)
    df["SoC"] = soc

    # Select pulse and full charge/discharge regions
    pulse_region = df[df["Step"].isin([5, 12])]
    if cell == "D":
        full_charge_region = df[df["Step"].isin([20, 22])]
    elif cell == "C":
        full_charge_region = df[df["Step"].isin([20, 28])]
    else:
        print("Invalid cell")
        return None

    # Define arrow coordinates
    pulse_start = pulse_region["Total Time"].iloc[0]
    pulse_soc = pulse_region["SoC"].iloc[0]

    full_charge_start = full_charge_region["Total Time"].iloc[0]
    full_charge_soc = full_charge_region["SoC"].iloc[0]
    full_discharge_end = full_charge_region["Total Time"].iloc[-1]
    full_discharge_soc = full_charge_region["SoC"].iloc[-1]

    # Plot SoC over Time
    plt.plot(df["Total Time"], df["SoC"], label="SoC Curve")

    # Annotate key points
    plt.annotate("Pulse Start", xy=(pulse_start, pulse_soc), xytext=(pulse_start + 50, pulse_soc + 2.5),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=9)

    plt.annotate("Pulse End/Full Charge Start", xy=(full_charge_start, full_charge_soc),
                 xytext=(full_charge_start + 50, full_charge_soc - 4),
                 arrowprops=dict(facecolor='green', arrowstyle='->'), fontsize=9)
    plt.annotate("Full Discharge End", xy=(full_discharge_end-40, full_discharge_soc),
                 xytext=(full_discharge_end+50, full_discharge_soc - 3),
                 arrowprops=dict(facecolor='green', arrowstyle='->'), fontsize=9)

    # Add labels, legend, and grid
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title(f"State of Charge vs Time: cell {cell} test {test}")
    plt.grid(True)
    plt.show()
    return None


def plot_soh(cell):
    '''
    Parameters: cell (string), cell "D" or "C"

    Plots SoH as a function of test number for given cell
    Returns None
    '''
    soh = []

    if cell == "D":
        # Calculating SoH
        for test in range(0,13+1):
            soh_i = dt.soh(cell,test)*100
            soh.append(soh_i)
        # Plotting
        plt.plot(list(range(0,13+1)),soh)
    elif cell == "C":
        # Calculating SoH
        for test in range(0,23+1):
            soh_i = dt.soh(cell,test)*100
            soh.append(soh_i)
        # Plotting
        plt.plot(list(range(0,23+1)),soh)
    else:
        print("Invalid cell entered")
        return None
    
    # Plotting
    plt.xlabel("Test number")
    plt.ylabel("SOH (%)")
    plt.title("SOH vs Test: cell "+cell)
    plt.show()
    return None


def plot_soc_ocv(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test (string) "01","02","10",etc...

    Plots OCV as a function of SoC for certain measure points
    Returns a dataframe containing initial data with SoC and OCV
    '''

    # Dataframe of initial data with SoC
    df_pre = pd.DataFrame(data={"Total Time": dt.extract(cell, test)[
                          "Total Time"], "SoC": dt.soc(str(cell), str(test))})

    # Extracting data for measurable OCVs
    col1 = dt.find_OCV(str(cell), str(test))["Total Time"]
    col2 = dt.find_OCV(str(cell), str(test))["Current"]
    col3 = dt.find_OCV(str(cell), str(test))["Voltage"]
    
    # Selecting respective SoCs for measured OCV points
    if cell == "C":
        col4 = [df_pre["SoC"].loc[df_pre["Total Time"] == i].values[0]
                if i in df_pre["Total Time"].values else np.nan for i in col1]
    elif cell == "D":
        col4 = [df_pre["SoC"].loc[df_pre["Total Time"] == i].values[0]
                if i in df_pre["Total Time"].values else np.nan for i in col1]
    else:
        print("Invalid cell")
        return None

    # New dataframe with OCV and SoC
    d = {"Total Time": col1, "Current": col2, "OCV": col3, "SoC": col4}
    df = pd.DataFrame(data=d)

    # Plotting
    plt.plot(df["SoC"],df["OCV"], "+")
    plt.ylabel("OCV (V)")
    plt.xlabel("SoC (% ratio)")
    plt.title("OCV vs SoC: cell "+cell+" test "+test)
    plt.show()
    return df


if __name__ == '__main__':
    plot_soc_ocv("C","01")
    plot_soc_ocv("D","01")
    '''
    data = extract("D", "02")
    soc = soc_full_d("02")
    plt.plot(data["Total Time"], soc)
    plot_test("D", "02")
    '''
    """
    for i in range(0, 1):
        if i < 10:
            soc_ocv("D", "0"+str(i))
        else:
            soc_ocv("D", str(i))
    """
    # ocv_voltage()

    # plt.text(x=0,y=0,s=str(soh("D","00")))
    plt.show()

    '''
    plot_test("D","01")

    data = extract("D", "01")[extract("D","01")["Current"] == 0.0]
    data_no_dupes = data.loc[~(data["Total Time"].diff().abs() < 3600)]
    print(data,data_no_dupes)
    
    plt.show()
    '''
