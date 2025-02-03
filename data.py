import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import plot as pt
import R0_identification_tianhe as rz
import Tianhe_csvPlot as ti
import math


folderPath = f'./cells_data'

csv_files = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csv_files if '_C_' in f]
csvFiles_D = [f for f in csv_files if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file))
       for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]

def extract_all_steps(first, second, cell, test):
    '''
    Parameters: first (int) first step, second (int) second step, cell (string) C or D, test (string) in the form 00, 01, etc..

    Returns dataframe for given step interval, does not remove duplicate step sequences
    '''

    data = extract(cell, test)
    step_data = data[data["Step"].isin(list(range(first, second+1)))]
    return step_data


def extract(cell, test):
    '''
    Parameters : cell (string) C or D, test (string) in the form 00, 01, etc..

    Extracts raw data from csv files corresponding to given cell and test
    Returns dataframe of extracted data
    '''

    file = [pd.read_csv(os.path.join(folderPath, f))
            for f in csv_files if '_'+(str(cell).upper())+"_" in f and str(test) in f]
    data = pd.concat(file)  # dataframe of corresponding csv data

    if file == []:
        print("No test found for given cell and test. Cell entry must be C/c or D/d")
        return None

    return data


def extract_step_2(first, second, cell, test):
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

def extract_step(first, second, cell, test):
    if type(test) == str:
        test = int(test)
    if cell == 'C':
        df = dfc[test]
    elif cell == 'D':
        df = dfd[test]
    else:
        print('oops')
        return [0,0]
    t_s, x = ti.locate(df, first, 0)
    x, t_e = ti.locate(df, second, 0)
    return ti.extract(df, t_s, t_e)

def q_remaining(cell,test):
    '''
    Parameters: cell (string) "C" or "D", test (string) test number "01","02","10",etc...
    
    Returns full discharge capacity for given cell and test
    '''

    # Extract full discharge data
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

    #plt.plot(data["Total Time"], data["Current"])
    #plt.show()

    
    print('Cycle', test, ': ', I, t, '\n')

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    return Q_remaining


def soc(cell, test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Returns list of SOC for cell D
    '''

    data_full = extract(cell, test)
    
    Q_remaining = q_remaining(cell,test)

    Q_available = [Q_remaining + (cumulative_trapezoid(data_full["Current"], data_full["Total Time"], initial=0)[
                                  i])/3600 for i in range(len(cumulative_trapezoid(data_full["Current"], data_full["Total Time"], initial=0)))]

    SOC = [(Q_available[i]/Q_remaining) - min(Q_available) /
           Q_remaining for i in range(len(Q_available))]

    return SOC


def soh(cell, test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of health of a cell at a given time 
    Returns SOH value of test
    '''
    Q_remaining = q_remaining(cell,test)

    q_init = q_remaining(cell,"00")

    SOH = Q_remaining / q_init
    #print(Q_remaining, q_init)
    return SOH


def find_OCV(cell, test):
    """
    Parameters: cell (str) C or D, 
                test (str) in the form of 01, 02, etc...

    Returns a list of different times that the circuit has reached OCV
    """

    '''
    data = extract(cell,test)[extract(cell,test)["Step"]==5]
    data_no_dupes = data.loc[~(data["Total Time"].diff().abs() < 3600)]
    '''
    if cell == "D":
        time_between_dupes = 600  # allows reduction of measurement points on graph
    elif cell == "C":
        time_between_dupes = 350
    else:
        print("Invalid cell entry. Cell entry must be C or D")
        return None
    data = extract(cell, test)[extract(cell, test)["Current"] == 0]
    data_no_dupes = data.loc[~(
        data["Total Time"].diff().abs() < time_between_dupes)]
    return data_no_dupes


def add_R0(cell,test):
    '''
    Parameters: test (int) test number

    Returns dataframe with original data and SoC and R0

    '''
    time_between_dupes = 300 #added this
    df = extract(cell, test)
    df["SoC"] = soc(cell, test)
    df["OCV"] = calculate_ocv(soc(cell, test), cell, test)
    print(df)
    R0 = [(abs(df["OCV"].iloc[i] - df["Voltage"].iloc[i]) / abs(df["Current"].iloc[i]) if abs(df["Current"].iloc[i]) > 1 else 0)
          for i in range(len(df["Current"]))]

    #rz.R0_fill(dfc)
    # print(df, '\n')
    # R0 = dfc[int(test)]["R0"]  # complete R0 column for given test

    # df["SoC"] = soc("C", test)
    df["R0"] = R0
    R0_no_dupes = df.loc[~(
        df["Total Time"].diff().abs() < time_between_dupes)] #added this
    #df["R0"] = R0_no_dupes
    print('original data \n\n', df, '\n\n')
    df["R0"].to_csv("resistance")

    # rz.R0_replace(df)
    # print(df, '\n')
    return df #changed this

def soc_ocv(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test (string) "01","02","10",etc...

    Plots OCV as a function of SoC for certain measure points
    Returns a dataframe containing initial data with SoC and OCV
    '''

    # Dataframe of initial data with SoC
    df_pre = pd.DataFrame(data={"Total Time":extract(cell, test)[
                          "Total Time"], "SoC":soc(str(cell), str(test))})

    # Extracting data for measurable OCVs
    col1 = find_OCV(str(cell), str(test))["Total Time"]
    col2 = find_OCV(str(cell), str(test))["Current"]
    col3 = find_OCV(str(cell), str(test))["Voltage"]

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
    return df

def soc_ocv_fitted(cell, test):
    '''
    Parameters: cell (string), test (string)

    Returns coefficients of fitted polynomial between SoC and OCV
    '''
    soc = soc_ocv(cell, test)["OCV"]
    ocv = soc_ocv(cell, test)["SoC"]

    # Fit a polynomial of degree 4
    coefficients = np.polyfit(ocv, soc, 5) #4 is original, 6 is best, 12 is good CAREFUL WITH OVERFITTING
    polynomial = np.poly1d(coefficients)

    # Generate fitted values for plotting
    # ocv_range = np.linspace(min(ocv), max(ocv), 100)
    # fitted_soc = polynomial(ocv)

    return polynomial


def calculate_ocv(soc, cell, test):
    '''
    Parameters: soc (list) soc values, cell (string), test (string)

    Returns list of calculated OCV values using the polynomial relation between OCV and SoC
    '''

    coefficients = soc_ocv_fitted(cell, test)
    # print([deg4_model(soc,coefficients[0],coefficients[1],coefficients[2],coefficients[3],coefficients[4]) for soc in soc])

    """
    print(pt.soc_ocv(cell, test)["SoC"])
    plt.plot()
    """
    poly = soc_ocv_fitted(cell, test)
    return [poly(soc) for soc in soc]
    # return [4+deg4_model(soc, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4])/15 for soc in soc]


def add_ocv(cell, test):
    '''
    Parameters: test(string), cell test

    Returns dataframe containing OCV for cell c
    '''

    df = add_R0(cell,test)
    df["OCV"] = calculate_ocv(df["SoC"], cell, test)
    return df


def calculate_model_voltage_0(cell,test):
    '''
    Parameters: test(string), cell test

    Returns dataframe containing model voltage for 0th order Thevenin for cell C
    '''

    df = add_ocv(cell,test)
    df["Model Voltage 0"] = [df["OCV"].iloc[i]+df["R0"].iloc[i]
                           * df["Current"].iloc[i] for i in range(len(df))]
    return df


def plot_r_soc(cell,test):
    '''
    Parameters: test(string) test number

    Plots R as a function of SoC
    Returns nothing
    '''

    df = calculate_model_voltage_0(cell,test)

    plt.plot(df["SoC"], df["R0"],'o')  # should be upside down U
    plt.show()


def calculate_r1(cell,test):
    '''
    Parameters: test (int) test number

    Returns dataframe with original data and SoC and R0

    '''
    time_between_dupes = 300 #added this
    df = extract(cell, test)
    df["SoC"] = soc(cell, test)
    df["OCV"] = calculate_ocv(soc(cell, test), cell, test)
    print(df)
    R1 = [(abs(df["OCV"].iloc[i] - df["Voltage"].iloc[i]) / abs(df["Current"].iloc[i]) if abs(df["Current"].iloc[i]) == 30 else 0)
          for i in range(len(df["Current"]))]

    #rz.R0_fill(dfc)
    # print(df, '\n')
    # R0 = dfc[int(test)]["R0"]  # complete R0 column for given test

    # df["SoC"] = soc("C", test)
    df["R1"] = R1
    R1_no_dupes = df.loc[~(
        df["Total Time"].diff().abs() < time_between_dupes)] #added this
    print('original data \n\n', df, '\n\n')
    df.to_csv("resistance2")

    # rz.R0_replace(df)
    # print(df, '\n')
    return R1_no_dupes #changed this

def calc_tau(cell,test):
    df = extract(cell,test)
    j = 1
    pulse_coords =[]
    tau = []
    R1 = []
    pulse_start = 0
    while j < len(df)-1:
        if abs(df["Current"].iloc[j]) == 30 and not(abs(df["Current"].iloc[j-1]) in [29,30]):
            pulse_start = j
        elif abs(df["Voltage"].iloc[j] - df["Voltage"].iloc[j+1]) > 0.15:
            pulse_coords.append((pulse_start,j))
        j += 1
    for element in pulse_coords:
        tau.append(df["Total Time"].iloc[element[1]]-df["Total Time"].iloc[element[0]])
        R1.append(abs(df["Voltage"].iloc[element[0]]-df["Voltage"].iloc[element[1]])/30)
    return tau, R1

def calc_r1(cell,test):
    df = extract(cell,test)
    df = df[df['Step']==7]
    pulse_coords = []
    pulse_start = 0
    tau = []
    R1 = []
    for i in range(len(df)-1):
        if i ==0:
            continue
        elif abs(df.index[i]-df.index[i-1])>1:
            pulse_start = i
        elif abs(df.index[i] - df.index[i+1])>1:
            pulse_coords.append((pulse_start,i))
    print(df)

    for element in pulse_coords:
        tau.append(df["Total Time"].iloc[element[1]]-df["Total Time"].iloc[element[0]])
        R1.append(abs(df["Voltage"].iloc[element[0]]-df["Voltage"].iloc[element[1]])/30)
    return pulse_coords, tau, R1

def calc_r1_2(cell, test):
    df1 = extract(cell, test)
    if cell == "C":
        df = df1[df1['Step'] == 7].copy()  # Ensure copy to modify safely
    elif cell == "D":
        df = df1[df1['Step'] ==8].copy()
    else:
        print("Invalid cell input")
        return None
    pulse_coords = []
    pulse_start = 0
    tau_values = []
    r1_values = []

    # Initialize columns for tau and R1
    df1["tau"] = None
    df1["R1"] = None

    for i in range(1, len(df) - 1):  # Skip the first row
        if abs(df.index[i] - df.index[i - 1]) > 1:
            pulse_start = i
        elif abs(df.index[i] - df.index[i + 1]) > 1:
            pulse_coords.append((pulse_start, i))

    for element in pulse_coords:
        start_index = element[0]
        end_index = element[1]

        # Calculate R1
        voltage_start = df["Voltage"].iloc[start_index]
        voltage_end = df["Voltage"].iloc[end_index]
        R1_value = abs(voltage_start - voltage_end) / 32

        tau_value = df["Total Time"].iloc[end_index] - df["Total Time"].iloc[start_index]
        """
        # Calculate tau (time to reach 63% of the voltage change) voltage
        voltage_target = voltage_start - 0.63 * abs(voltage_end - voltage_start)

        # Find the closest index where the voltage meets or exceeds the target
        time_to_adapt = None
        for idx in range(start_index, end_index + 1):
            if df["Voltage"].iloc[idx] <= voltage_target:
                time_to_adapt = df["Total Time"].iloc[idx] - df["Total Time"].iloc[start_index]
                break

        tau_value = time_to_adapt if time_to_adapt is not None else 0
        """


        # Assign tau and R1 only at the start index
        #df1.loc[df1.index[start_index], "tau"] = tau_value
        #df1.loc[df1.index[start_index], "R1"] = R1_value

        df1.loc[df1.index.isin(range(start_index,end_index)),"tau"] = tau_value
        df1.loc[df1.index.isin(range(start_index,end_index)),"R1"] = R1_value

        # Store values for debugging or further processing
        tau_values.append(tau_value)
        r1_values.append(R1_value)
    df1.to_csv("r1 data")
    return df1


def plot_soc_tau_r1(cell,test):
    df = pd.DataFrame(data = {"tau": calc_r1_2(cell,test)[1], "R1": calc_r1_2(cell,test)[2]})
    print(df)

    df["tau"] = calc_r1_2(cell,test)[1]
    df["R1"] = calc_r1_2(cell,test)[2]

    print(df)

def calculate_model_voltage_1(cell,test):
    df = calculate_model_voltage_0(cell,test)
    df1 = calc_r1_2(cell,test)
    df["Model Voltage 1"] = [df["OCV"].iloc[i]+df["R0"].iloc[i]*df1["Current"].iloc[i]
                             +df1["R1"].iloc[i]*df1["Current"].iloc[i]*math.exp(-df1["Total Time"].iloc[i]/df1["tau"].iloc[i]) if df1["R1"].iloc[i]!=None 
                             else df["OCV"].iloc[i]+df["R0"].iloc[i]*df1["Current"].iloc[i] for i in range(len(df))]
    #voltage = OCV + R0 * I + R1 * I * exp(-t/tau)
    return df

if __name__ == "__main__":
    # print(soc("C","01"))
    """
    plot_r_soc("C","01")
    pt.plot_test("C","01")
    pt.plot_soc("D","03")
    plt.show()
    """
    #print(add_R0("C","01"))
    """
    df = calculate_model_voltage_1("C","01")
    plt.plot(df["Total Time"],df["Model Voltage 1"])
    plt.show()
    """

    
    df1 = calc_r1_2("C","01")
    
    plt.plot(df1["Total Time"],df1["R1"])
    plt.show()
    print(calculate_model_voltage_1("D","01"))
    plot_soc_tau_r1('C','01')
    print(calc_r1_2("C","01"))
    """
    df= calc_tau("D","01")
    print(df)
        
    df = calculate_model_voltage_0("D","03")

    polynomial = soc_ocv_fitted("D","03")
    y = [polynomial(soc) for soc in df["SoC"]]
    pt.plot_soc_ocv("D","03")
    plt.plot(df["SoC"],y)
    plt.show()


    fig, axs = plt.subplots(3, 1)
    axs[0].plot(df["Total Time"], df["Voltage"])
    df.to_csv("r0")
    df = df[~np.isnan(df["R0"])]
    print(df)
    axs[2].plot(df["Total Time"], df["OCV"])
    axs[1].plot(df["Total Time"], df["R0"])  # should be upside down U
    plt.show()

    plt.plot(df["SoC"], df["R0"], 'o')
    plt.show()
    # print(calc_r0("04"))

    df.to_csv("0th_model_data")
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(df["Total Time"], df["Model Voltage"])
    axs[0].set_title("Model Voltage vs Time")
    axs[1].plot(df["Total Time"], df["Voltage"])
    axs[1].set_title("Voltage vs Time")

    plt.show()
    """
    '''
    pt.soc_ocv("C", "05")
    pt.soc_ocv("D", "01")
    plt.show()
    '''
