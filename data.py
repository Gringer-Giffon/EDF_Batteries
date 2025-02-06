# --------------------------------------------INITIALISATION--------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import plot as pt
import R0_identification_tianhe as rz
import Tianhe_csvPlot as ti
import math
import R0_fit
import OCV_fit
from scipy.optimize import curve_fit
import cell_plotting_tests_etienne as et

pd.options.mode.chained_assignment = None

folderPath = f'./cells_data'

csv_files = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csv_files if '_C_' in f]
csvFiles_D = [f for f in csv_files if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file))
       for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]


# -----------------------------------------------CSV FILE EXTRACTION--------------------------------------------------------
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
        print('Invalid cell input')
        return [0, 0]
    t_s, x = ti.locate(df, first, 0)
    x, t_e = ti.locate(df, second, 0)
    return ti.extract(df, t_s, t_e)


# ----------------------------------------------------SOC AND SOH---------------------------------------------------

def q_remaining(cell, test):
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

    # plt.plot(data["Total Time"], data["Current"])
    # plt.show()

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    return Q_remaining


def soc(cell, test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Returns list of SOC for cell D
    '''

    data_full = extract(cell, test)

    Q_remaining = q_remaining(cell, test)

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
    Q_remaining = q_remaining(cell, test)
    q_init = q_remaining(cell, "00")
    SOH = Q_remaining / q_init
    # print(Q_remaining, q_init)
    return SOH


# -------------------------------------------------------OCV--------------------------------------------------------------
def find_OCV(cell, test):
    """
    Parameters: cell (str) C or D, 
                test (str) in the form of 01, 02, etc...

    Returns a dataframe of different times that the circuit has reached OCV
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


def soc_ocv(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test (string) "01","02","10",etc...

    Plots OCV as a function of SoC for certain measure points
    Returns a dataframe containing initial data with SoC and OCV
    '''

    # Dataframe of initial data with SoC
    df_pre = pd.DataFrame(data={"Total Time": extract(cell, test)[
                          "Total Time"], "SoC": soc(str(cell), str(test))})

    # Extracting data for measurable OCVs
    # col1 = find_OCV(str(cell), str(test))["Total Time"]
    # col2 = find_OCV(str(cell), str(test))["Current"]
    # col3 = find_OCV(str(cell), str(test))["Voltage"]

    col1 = [pulse["Total Time"].iloc[1]
            for pulse in et.extract_pulses(cell, test)]
    col2 = [pulse["Current"].iloc[1]
            for pulse in et.extract_pulses(cell, test)]
    col3 = et.measure_OCV(et.extract_pulses(cell, test))

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

    Returns fitted polynomial between SoC and OCV
    '''
    soc = soc_ocv(cell, test)["SoC"]
    ocv = soc_ocv(cell, test)["OCV"]

    # Fit a polynomial of degree 4
    # 4 is original, 6 is best, 12 is good CAREFUL WITH OVERFITTING
    coefficients = np.polyfit(soc, ocv, 7)
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

    '''
    print(pt.soc_ocv(cell, test)["SoC"])
    plt.plot()
    '''
    poly = soc_ocv_fitted(cell, test)
    return [poly(soc) for soc in soc]
    # return [4+deg4_model(soc, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4])/15 for soc in soc]


"""
def calculate_ocv(soc, cell, test):
    global soh_value
    return [OCV_fit.f(soc_val, soh_value) for soc_val in soc]
"""


def add_ocv(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string), cell test

    Returns dataframe containing OCV
    '''

    df = add_R0(cell, test)  # Data frame with R0
    df["OCV"] = calculate_ocv(df["SoC"], cell, test)
    return df


# -------------------------------------------------------R0--------------------------------------------------------------------
def add_R0(cell, test):
    '''
    Parameters: test (int) test number

    Returns dataframe with original data and SoC and R0

    '''
    #global soh_value
    soh_value = soh(cell, test)

    time_between_dupes = 300  # added this
    df = extract(cell, test)
    df["SoC"] = soc(cell, test)

    # df["OCV"] = calculate_ocv(soc(cell, test), cell, test)

    # R0 = [(abs(df["OCV"].iloc[i] - df["Voltage"].iloc[i]) / abs(df["Current"].iloc[i]) if abs(df["Current"].iloc[i]) > 1 else 0)
    # for i in range(len(df["Current"]))]

    # rz.R0_fill(dfc)
    # print(df, '\n')
    # R0 = dfc[int(test)]["R0"]  # complete R0 column for given test

    # df["SoC"] = soc("C", test)
    # df["R0"] = calculate_ocv(df["SoC"],cell,test)
    # R0_no_dupes = df.loc[~(
    # df["Total Time"].diff().abs() < time_between_dupes)] #added this
    # df["R0"] = R0_no_dupes

    df["R0"] = [R0_fit.f(soc_value, soh_value) for soc_value in df["SoC"]]

    # rz.R0_replace(df)
    # print(df, '\n')
    return df  # changed this


# ----------------------------------------------------MODEL 0---------------------------------------------------------------
def calculate_model_voltage_0(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string), cell test

    Returns dataframe containing model voltage for 0th order Thevenin model
    '''
    global soh_value

    soh_value = soh(cell, test)

    df = add_ocv(cell, test)  # Dataframe with R0 and OCV
    df["Model Voltage 0"] = [df["OCV"].iloc[i]+df["R0"].iloc[i]
                             * df["Current"].iloc[i] for i in range(len(df))]
    return df


# ---------------------------------------------------------R1------------------------------------------------------------------

def find_R1(cell, test):
    '''
    Parameters: cell (str) C or D, 
                test (str) in the form of 01, 02, etc...

    Returns a dataframe of different times where the capacitor acts, where R1 can be measured
    '''

    '''
    data = extract(cell,test)[extract(cell,test)["Step"]==5]
    data_no_dupes = data.loc[~(data["Total Time"].diff().abs() < 3600)]
    '''
    if cell == "D":
        time_between_dupes = 0  # allows reduction of measurement points on graph
        step = 6
    elif cell == "C":
        time_between_dupes = 0
        step = 7
    else:
        print("Invalid cell entry. Cell entry must be C or D")
        return None
    data = add_ocv(cell, test)[add_ocv(cell, test)["Step"] == step]
    # data_no_dupes = data.loc[~(
    # data["Total Time"].diff().abs() < time_between_dupes)]
    return data


def measure_r1(cell, test):
    '''
    Parameters: cell (str) C or D, 
                test (str) in the form of 01, 02, etc...

    Returns dataframe with measured values of R1 where it can be measured
    '''
    df = find_R1(cell, test)

    discontinuities = df.index.to_series().diff().gt(
        1)  # Find differences greater than 1

    # Initialize the list to hold DataFrame splits
    splits = []
    start_idx = 0

    # Split DataFrame at the discontinuities
    for i, discontinuity in enumerate(discontinuities):
        if discontinuity:  # Discontinuity found
            # Add the segment up to the discontinuity
            splits.append(df.iloc[start_idx:i])
            start_idx = i

    for split in splits:
        R1 = []
        for i in range(0, len(split)):
            R1.append(abs(split["Voltage"].iloc[0] -
                          split["Voltage"].iloc[-1]) / abs(split["Current"].iloc[-1]))
        split["R1"] = R1

    """
    df = splits[0]
    for i in range (1, len(splits)):
        df = df.merge(splits[i])
    """
    return pd.concat(splits)


def soc_R1_fitted(cell, test):
    '''
    Parameters: cell (string), test (string)

    Returns fitted polynomial between SoC and R1
    '''
    df = measure_r1(cell, test)
    soc = df["SoC"]
    R1 = df["R1"]

    # Fit a polynomial of degree 4
    # 5 is original, 6 is best, 12 is good CAREFUL WITH OVERFITTING
    coefficients = np.polyfit(soc, R1, 7)
    polynomial = np.poly1d(coefficients)

    # plt.plot(soc,R1,"b")

    # Generate fitted values for plotting
    # ocv_range = np.linspace(min(ocv), max(ocv), 100)
    # fitted_soc = polynomial(ocv)

    return polynomial


def calculate_r1(soc, cell, test):
    '''
    Parameters: soc (list) soc values, cell (string), test (string)

    Returns list of calculated R1 values using the fitted polynomial relation between OCV and R1
    '''

    # coefficients = soc_R1_fitted(cell, test)
    # print([deg4_model(soc,coefficients[0],coefficients[1],coefficients[2],coefficients[3],coefficients[4]) for soc in soc])

    """
    print(pt.soc_ocv(cell, test)["SoC"])
    plt.plot()
    """
    poly = soc_R1_fitted(cell, test)
    return [poly(soc) for soc in soc]


def add_r1(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string), cell test

    Returns dataframe containing R1
    '''

    df = add_ocv(cell, test)
    df["R1"] = calculate_r1(df["SoC"], cell, test)
    return df


# -------------------------------------------------------TAU----------------------------------------------------------------

def find_tau(cell, test):
    if cell == "D":
        df = add_r1(cell, test)[add_r1(cell, test)["Step"] == 6]
        print("r1", df)
    elif cell == "C":
        df = add_r1(cell, test)[add_r1(cell, test)["Step"] == 7]
        print("r1", df)
    else:
        print("Invalid cell")
        return None

    return df


def measure_tau(cell, test):
    df = find_tau(cell, test)

    discontinuities = df.index.to_series().diff().gt(1)

    # Initialize the list to hold DataFrame splits
    splits = []
    start_idx = 0

    # Split DataFrame at the discontinuities
    for i, discontinuity in enumerate(discontinuities):
        if discontinuity:  # Discontinuity found
            # Add the segment up to the discontinuity
            splits.append(df.iloc[start_idx:i])
            start_idx = i

    for split in splits:
        # split["tau"] = None
        # split["tau"] = abs(split["Total Time"][abs(split["Voltage"] == 0.63*split["Voltage"].iloc[-1]) < 0.2].iloc[0] - split["Total Time"].iloc[0])

        """
        time_min = split["Total Time"][split["Voltage"] == min(split["Voltage"])].iloc[0]
        target_voltage = 0.63*max(split["Voltage"])
        time_63 = split["Total Time"][abs(split["Voltage"] - target_voltage) < 0.2].iloc[0]
        tau = [abs(time_63-time_min)]*len(split["Total Time"])
        print("min",time_min)
        print("63",time_63)
        print("tau",tau)
        split["tau"] = tau
        """
        """
        target_voltage = df["Voltage"].iloc[spike+1] - 0.63 * \
        abs(df["Voltage"].iloc[spike+1]-min(df["Voltage"]))

        idx = (df["Voltage"] - target_voltage).abs().idxmin()

        tau = (df["Total Time"].loc[idx] - df["Total Time"].iloc[0])

        print(tau)
        """

        # target_voltage = df["Voltage"].iloc[0] - 0.63*abs(df["Voltage"].iloc[0]-min(df["Voltage"]))

        final_voltage = min(split["Voltage"])
        target_voltage = split["Voltage"].iloc[0] - 0.63 * \
            abs(split["Voltage"].iloc[0]-min(split["Voltage"]))

        # Find the index where voltage is closest to target
        idx = (split["Voltage"] - target_voltage).abs().idxmin()

        # split["tau"] = (split["Total Time"].loc[idx] - split["Total Time"].iloc[0])

        if idx is not None and idx > 0:
            split["tau"] = split["Total Time"].loc[idx] - \
                split["Total Time"].iloc[0]
        else:
            split["tau"] = np.nan  # or some default value
    return pd.concat(splits)


"""
def find_tau(cell,test):
    '''
    Parameters: cell (string) "C" or "D", test (string) test number

    Returns dataframe of a range of rows where tau can be measured
    '''
    if cell == "D":
        time_between_dupes = 0  # allows reduction of measurement points on graph
        data = add_ocv(cell,test)[add_ocv(cell,test)["Current"].isin(range(29,32))]
    elif cell == "C":
        time_between_dupes = 0
        data = add_ocv(cell,test)[add_ocv(cell,test)["Current"].isin(range(31,34))]
    else:
        print("Invalid cell entry. Cell entry must be C or D")
        return None
    data_no_dupes = data.loc[~(
        data["Total Time"].diff().abs() < time_between_dupes)]
    return data_no_dupes


def measure_tau(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test (string) test number

    Returns dataframe including calculated values for tau over measurable range
    '''
    df = find_tau(cell,test)

    discontinuities = df.index.to_series().diff().gt(1)  # Find differences greater than 1

    # Initialize the list to hold DataFrame splits
    splits = []
    start_idx = 0

    # Split DataFrame at the discontinuities
    for i, discontinuity in enumerate(discontinuities):
        if discontinuity:  # Discontinuity found
            splits.append(df.iloc[start_idx:i])  # Add the segment up to the discontinuity
            start_idx = i

    tau_values = []

    for split in splits:
        # Ignore empty splits
        if split.empty:
            continue

        target_voltage = 0.95 * split["Voltage"].iloc[-1]

        # Find the closest time when voltage reaches the target value
        closest_index = (split["Voltage"] - target_voltage).abs().idxmin()
        
        if closest_index != split.index[0]:  # Ensure a valid time difference
            tau_value = (split["Total Time"].loc[closest_index] - split["Total Time"].iloc[0])/5
        else:
            tau_value = np.nan

        # Append tau to the DataFrame and store in list
        split["tau"] = tau_value
        tau_values.append(split)
    
    '''
    df = splits[0]
    for i in range (1, len(splits)):
        df = df.merge(splits[i])
    '''
    '''
    x = []
    y = []

    for split in splits:
        x.append(split["SoC"].iloc[0])
        y.append(split["tau"].iloc[0])


    plt.plot(x,y,"x")
    '''

    return pd.concat(splits)

"""


def soc_tau_fitted(cell, test):
    '''
    Parameters: cell (string), test (string)

    Returns fitted polynomial between SoC and R1
    '''
    df = measure_tau(cell, test)
    soc = df["SoC"]
    tau = df["tau"]

    # Fit a polynomial of degree 4

    # 5 is original, 6 is best, 12 is good CAREFUL WITH OVERFITTING
    coefficients = np.polyfit(soc, np.log(tau), 5)
    fit = np.poly1d(coefficients)

    # plt.plot(soc,np.log(tau),"xb")

    # Generate fitted values for plotting
    # ocv_range = np.linspace(min(ocv), max(ocv), 100)
    # fitted_soc = polynomial(ocv)

    return fit


def calculate_tau(soc, cell, test):
    '''
    Parameters: soc (list) soc values, cell (string), test (string)

    Returns list of calculated R1 values using the fitted polynomial relation between OCV and R1
    '''

    # coefficients = soc_R1_fitted(cell, test)
    # print([deg4_model(soc,coefficients[0],coefficients[1],coefficients[2],coefficients[3],coefficients[4]) for soc in soc])

    """
    print(pt.soc_ocv(cell, test)["SoC"])
    plt.plot()
    """
    poly = soc_tau_fitted(cell, test)
    return [poly(soc) for soc in soc]


def add_tau(cell, test):
    df = add_r1(cell, test)
    df["tau"] = calculate_tau(df["SoC"], cell, test)
    return df


def add_c1(cell, test):
    df = add_tau(cell, test)
    df["C1"] = df["tau"]/df["R1"]
    return df


def time_pulses_calc(df):
    threashold = 10
    time = df['Total Time'][0]
    time_list = []
    time_list.append(time)
    for i in range(len(df)-1):
        # if abs(df['Current'][i] - df['Current'][i+1])/(df['Total Time'][i+1] - df['Total Time'][i]) >= threashold:
        if abs(abs(df['Current'][i]) - abs(df['Current'][i+1])) >= threashold:
            time = 0
        time += df['Total Time'][i+1] - df['Total Time'][i]
        time_list.append(time)
    df['Time'] = time_list
    return df


# -----------------------------------------------------MODEL 1------------------------------------------------------------

def calculate_model_voltage_1(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test (string) test number

    Returns dataframe of initial value and including 0th order and first order voltage and parameters
    '''
    global soh_value
    soh_value = soh(cell, test)

    df = calculate_model_voltage_0(cell, test)
    df1 = add_c1(cell, test)

    # df["Model Voltage 1"] = df["Model Voltage 0"].copy()

    # Merge df1 into df to align by "Total Time" directly
    df = df.merge(df1[["Total Time", "R1", "tau", "C1"]],
                  on="Total Time", how="left")

    # Calculate model voltage using vectorized operations
    df = time_pulses_calc(df)
    df["Model Voltage 1"] = [df["Model Voltage 0"].iloc[i] + df["R1"].iloc[i] *
                             df1["Current"].iloc[i] * (1-np.exp(-df["Time"].iloc[i] / df["tau"].iloc[i])) for i in range(len(df))]


    # Handle NaN results if tau was NaN or zero
    # df["Model Voltage 1"].fillna(df["Model Voltage 0"], inplace=True)
    return df


"""
def table_soc(cell,test):
    df = soc_ocv(cell,test)
    df1 = add_R0(cell,test)
    df = df.merge(df1[["Total Time", "R0"]],
                  on="Total Time", how="left")
    return df
"""

if __name__ == "__main__":

    # print(table_soc("C","01"))
    # print(soc("C","01"))
    """
    plot_r_soc("C","01")
    pt.plot_test("C","01")
    pt.plot_soc("D","03")
    plt.show()
    """
    # print(add_R0("C","01"))
    """
    df = calculate_model_voltage_1("C","01")
    plt.plot(df["Total Time"],df["Model Voltage 1"])
    plt.show()
    """
    # print(find_R1("D","01"))

    # ocv = add_ocv("D","01")

    # print(measure_tau("D","08"))
    # print(calculate_model_voltage_1("C", "01"))

    """
    df = measure_r1("C","01")
    plt.show()
    x = np.linspace(0,1)
    y = calculate_r1(x,"C","01")

    plt.plot()
    plt.plot(x,y)
    """

    # df = calc_r1_2("C","01")
    # plt.plot(df["SoC"][df["R1"] != 0],df["R1"][df["R1"]!=0])
    plt.show()

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
