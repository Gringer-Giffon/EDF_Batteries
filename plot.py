import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid


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


def q_initial(cell):
    if cell == "C":
        data = extract_step(26,27,"C", "00")
    elif cell == "D":
        data = extract_step(21,23,"D", "00")
    else:
        print("Invalid cell entry. Cell entry must be C or D")
        return None

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

    data = extract_step(21, 23, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    Q_available = [Q_remaining - I*(data["Total Time"].iloc[i] -
                                    data["Total Time"].iloc[0])/3600 for i in range(len(data["Total Time"]))]

    SOC = [Q_available[i]/Q_remaining for i in range(len(data["Total Time"]))]

    return SOC


def soh(cell, test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of health of a cell at a given time 
    Returns SOH value of test
    '''
    if cell == "C":
        data = extract_step(26, 27, "C", test)
    elif cell == "D":
        data = extract_step(21, 23, "D", test)
    else:
        print("Invalid cell entry. Cell entry must be C or D")
        return None

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining
    Q_remaining = I*t/3600
    
    q_init = q_initial(cell)

    SOH = Q_remaining / q_init
    print(Q_remaining, q_init)
    return SOH


def plot_soh(cell):
    '''
    Plots the SOH for cell D
    Returns nothing
    '''

    soh_s = []
    for i in range(0, 13+1):
        if i < 10:
            soh_s.append(soh(cell,"0"+str(i)))
        else:
            soh_s.append(soh(cell,str(i)))

    # Plotting
    plt.plot(list(range(0, 13+1)), soh_s)
    plt.xlabel("Test Number")
    plt.ylabel("SOH (%)")
    plt.title("State of Health vs Test Number")


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

def plot_soc(cell,test):
    '''
    Parameters : cell (string) C or D, test (string) in the form 00, 01, etc..

    Plots the state of charge of given cell for the given test
    Returns nothing
    '''
    if cell == "C":
        soc = soc_full_c(test)
    elif cell == "D":
        soc = soc_full_d(test)
    else:
        print("Invalid cell entry. Cell entry must be C or D")
        return None
    
    # Plotting
    plt.plot(extract(cell,test)["Total Time"],soc)
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.title("State of Charge vs Time")

def find_OCV(cell, test):
    """
    Parameters: cell (str) C or D, 
                test (str) in the form of 01, 02, etc...
                
    Returns a list of different times that the circuit has reached OCV
    """
    
    data = extract(cell, test)[extract(cell,test)["Step"] == 5]
    data_no_dupes = data.loc[~(data["Total Time"].diff().abs() < 3600)]
    print(data_no_dupes)
    return data_no_dupes

def soc_ocv(cell, test):

    df_pre = pd.DataFrame(data = {"Total Time":extract(cell,test)["Total Time"], "SoC":soc_full_d(str(test))})
    print(df_pre)
    #plt.plot(df_pre["Total Time"], df_pre["SoC"])
    #plt.show()

    col1 = find_OCV(str(cell),str(test))["Total Time"]
    col2 = find_OCV(str(cell),str(test))["Current"]
    col3 = find_OCV(str(cell),str(test))["Voltage"]
    print("here:")
    print(col1)
    if cell == "C": 
        col4 = [df_pre["SoC"].loc[df_pre["Total Time"] == i].values[0] if i in df_pre["Total Time"].values else np.nan for i in col1]
    elif cell =="D":
        col4 = [df_pre["SoC"].loc[df_pre["Total Time"] == i].values[0] if i in df_pre["Total Time"].values else np.nan for i in col1]
    else:
        print("Invalid cell")
        return None

    d = {"Total Time": col1, "Current": col2, "Voltage": col3, "SoC": col4}
    df = pd.DataFrame(data=d)

    print(df)
    plt.plot(df["Voltage"], df["SoC"],"+")
    
    
    

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
    #plt.text(x=0,y=0,s=str(soh("D","00")))
    plt.show()

    #look at C
    #reorganise files
    #make stuff available for cell C
    #find R0
    
    
