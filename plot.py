import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import data as dt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.color'] = 'darkred'  # Global text color
plt.rcParams['axes.labelcolor'] = 'darkred'  # Axis label color
# plt.rcParams['lines.color'] = 'darkred'
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['darkred'])

directory = f"./cells_data"
centrale_red = "#AF4458"
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

    axs[0].plot(time, current, "o", ms = 1, color=centrale_red)
    axs[0].set_title("Current vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Current (A)")

    axs[1].plot(time, voltage,  "o", ms = 1, color=centrale_red)
    axs[1].set_title("Voltage vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Voltage (V)")

    axs[2].plot(time, step, "o", ms = 1, color= centrale_red)
    axs[2].set_title("Step vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Step")

    plt.subplots_adjust(hspace=1)  # adjust space between plots
    plt.show()

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
        for test in range(0, 13+1):
            soh_i = dt.soh(cell, test)*100
            soh.append(soh_i)
        # Plotting
        plt.plot(list(range(0, 13+1)), soh)
    elif cell == "C":
        # Calculating SoH
        for test in range(0, 23+1):
            soh_i = dt.soh(cell, test)*100
            soh.append(soh_i)
        # Plotting
        plt.plot(list(range(0, 23+1)), soh)
    else:
        print("Invalid cell entered")
        return None

    # Plotting
    plt.xlabel("Test number")
    plt.ylabel("SOH (%)")
    plt.title("SOH vs Test: cell "+cell)
    plt.show()
    return None


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


def plot_soc_ocv(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test (string) "01","02","10",etc...

    Plots OCV as a function of SoC for certain measure points
    Returns None
    '''

    # Dataframe of initial data with SoC and OCV
    df = dt.soc_ocv(cell, test)

    # Plotting
    plt.plot(df["SoC"], df["OCV"], "+")
    plt.ylabel("OCV (V)")
    plt.xlabel("SoC (% ratio)")
    plt.title("OCV vs SoC: cell "+cell+" test "+test)
    return None


def plot_model_data_soc_ocv(cell, test):
    
    plot_soc_ocv(cell, test)

    polynomial = dt.soc_ocv_fitted(cell, test)

    x = np.linspace(0, 1, 100)

    plt.plot(x, polynomial(x), "b-")


def plot_model_voltage_0(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string) test number

    Plots Oth order model voltage
    '''
    df1 = dt.extract(cell, test)
    df = dt.calculate_model_voltage_0(cell, test)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(df1["Total Time"], df1["Voltage"])
    axs[1].plot(df["Total Time"], df["Model Voltage 0"])
    axs[0].set_title("Measured voltage over time")
    axs[1].set_title("Model voltage over time")
    plt.show()


def plot_model_voltage_1(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string) test number

    Plots 1st order model voltage
    '''
    df = dt.calculate_model_voltage_1(cell, test)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(df["Total Time"], df["Voltage"])
    axs[1].plot(df["Total Time"], df["Model Voltage 0"])
    axs[0].set_title("Measured voltage over time")
    axs[1].set_title("Model voltage 0 over time")
    axs[2].plot(df["Total Time"], df["Model Voltage 1"])
    axs[2].set_title("Model voltage 1 over time")
    plt.subplots_adjust(hspace=1)
    plt.show()

def plot_simultaneous_0(cell,test):
    df = dt.calculate_model_voltage_0(cell, test)
    plt.plot(df["Total Time"], df["Voltage"], "bx")
    plt.plot(df["Total Time"], df["Model Voltage 0"], "gx")
    plt.legend(["Data","Model Voltage 0"])
    plt.show()

def plot_simultaneous(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string) test number

    Simultaneously plot measured, order 0 and order 1 voltage on subplots
    '''
    df = dt.calculate_model_voltage_1(cell, test)
    plt.plot(df["Total Time"], df["Voltage"], "bx")
    plt.plot(df["Total Time"], df["Model Voltage 0"], "gx")
    plt.plot(df["Total Time"], df["Model Voltage 1"], "rx")

    plt.show()

def plot_simultaneous_1(cell,test):
    df = dt.calculate_model_voltage_1(cell, test)
    plt.plot(df["Total Time"], df["Voltage"], "bx")
    plt.plot(df["Total Time"], df["Model Voltage 1"], "rx")
    plt.legend(["Data","Model Voltage 1"])


def plot_r0_soc(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string) test number

    Plots R as a function of SoC
    Returns nothing
    '''

    df = dt.calculate_model_voltage_0(cell, test)
    plt.plot(df["SoC"], df["R0"], 'o')  # should be upside down U
    plt.title("R0 vs SoC")
    plt.show()

def plot_r1_soc(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string) test number

    Plots R as a function of SoC
    Returns nothing
    '''

    df = dt.calculate_model_voltage_1(cell, test)
    plt.plot(df["SoC"], df["R1"])  # should be upside down U
    plt.title("R1 vs SoC")
    plt.show()

def plot_tau_soc(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string) test number

    Plots R as a function of SoC
    Returns nothing
    '''

    df = dt.calculate_model_voltage_1(cell, test)
    plt.plot(df["SoC"], df["tau"])  # should be upside down U
    plt.title("tau vs SoC")
    plt.show()

def plot_c1_soc(cell,test):
    '''
    Parameters: cell (string) "C" or "D", test(string) test number

    Plots R as a function of SoC
    Returns nothing
    '''

    df = dt.calculate_model_voltage_1(cell, test)
    plt.plot(df["SoC"], df["C1"],"o")  # should be upside down U
    plt.title("C1 vs SoC")
    plt.show()


if __name__ == '__main__':
    # plot_simultaneous("C","01")
    #plot_test("C", "01")  # 7-9
    #plot_test("D", "01")  # bigger than 6 to 7

    #plot_r0_soc("C","01")
    #plot_r1_soc("D","01")
    #plot_c1_soc("C","01")
    #plot_tau_soc("C","01")


    #plot_simultaneous("D", "01")
    #plot_simultaneous("C", "01")
    #plot_soc_ocv("C","01")
    plot_test("D","01")
    plt.show
    plot_simultaneous_1("D","01")
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
