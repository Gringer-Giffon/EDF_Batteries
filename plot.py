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
    soc = dt.soc(cell,test)
    
    # Plotting
    plt.plot(dt.extract(cell, test)["Total Time"], soc)
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title("State of Charge vs Time")





def soc_ocv(cell, test):

    df_pre = pd.DataFrame(data={"Total Time": dt.extract(cell, test)[
                          "Total Time"], "SoC": dt.soc(str(cell),str(test))})
    print(df_pre)
    # plt.plot(df_pre["Total Time"], df_pre["SoC"])
    # plt.show()

    col1 = dt.find_OCV(str(cell), str(test))["Total Time"]
    col2 = dt.find_OCV(str(cell), str(test))["Current"]
    col3 = dt.find_OCV(str(cell), str(test))["Voltage"]
    print("here:")
    print(col1)
    if cell == "C":
        col4 = [df_pre["SoC"].loc[df_pre["Total Time"] == i].values[0]
                if i in df_pre["Total Time"].values else np.nan for i in col1]
    elif cell == "D":
        col4 = [df_pre["SoC"].loc[df_pre["Total Time"] == i].values[0]
                if i in df_pre["Total Time"].values else np.nan for i in col1]
    else:
        print("Invalid cell")
        return None

    d = {"Total Time": col1, "Current": col2, "Voltage": col3, "SoC": col4}
    df = pd.DataFrame(data=d)

    print(df)
    plt.plot(df["Voltage"], df["SoC"], "+")


if __name__ == '__main__':
    '''
    data = extract("D", "02")
    soc = soc_full_d("02")
    plt.plot(data["Total Time"], soc)
    plot_test("D", "02")
    '''

    for i in range(0, 1):
        if i < 10:
            soc_ocv("D", "0"+str(i))
        else:
            soc_ocv("D", str(i))

    # ocv_voltage()
    plt.xlabel("OCV (V)")
    plt.ylabel("SoC (%)")
    plt.title("SoC vs OCV")
    # plt.text(x=0,y=0,s=str(soh("D","00")))
    plt.show()

    '''
    plot_test("D","01")

    data = extract("D", "01")[extract("D","01")["Current"] == 0.0]
    data_no_dupes = data.loc[~(data["Total Time"].diff().abs() < 3600)]
    print(data,data_no_dupes)
    
    plt.show()
    '''



    
