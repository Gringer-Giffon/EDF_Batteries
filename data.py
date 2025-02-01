import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import plot as pt
import R0_identification_tianhe as rz


folderPath = f'./cells_data'

csv_files = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csv_files if '_C_' in f]
csvFiles_D = [f for f in csv_files if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file))
       for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]


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
    print(Q_remaining, q_init)
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
    print('original data \n\n', df, '\n\n')
    df["R0"].to_csv("resistance")

    # rz.R0_replace(df)
    # print(df, '\n')
    return R0_no_dupes #changed this

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
    df["Model Voltage"] = [df["OCV"].iloc[i]+df["R0"].iloc[i]
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
    R1 = [(abs(df["OCV"].iloc[i] - df["Voltage"].iloc[i]) / abs(df["Current"].iloc[i]) if abs(df["Current"].iloc[i])==30 else 0)
          for i in range(len(df["Current"]))]

    #rz.R0_fill(dfc)
    # print(df, '\n')
    # R0 = dfc[int(test)]["R0"]  # complete R0 column for given test

    # df["SoC"] = soc("C", test)
    df["R1"] = R1
    R1_no_dupes = df.loc[~(
        df["Total Time"].diff().abs() < time_between_dupes)] #added this
    print('original data \n\n', df, '\n\n')
    df["R1"].to_csv("resistance1")

    # rz.R0_replace(df)
    # print(df, '\n')
    return R1_no_dupes #changed this




if __name__ == "__main__":
    # print(soc("C","01"))
    """
    plot_r_soc("C","01")
    pt.plot_test("C","01")
    pt.plot_soc("D","03")
    plt.show()
    """
    
    df = calculate_model_voltage_0("C","03")

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
    
    '''
    pt.soc_ocv("C", "05")
    pt.soc_ocv("D", "01")
    plt.show()
    '''
