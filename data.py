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


def q_initial(cell):
    if cell == "C":
        data = extract_step(26, 27, "C", "00")
    elif cell == "D":
        data = extract_step(21, 23, "D", "00")
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


def add_R0_c(test):
    '''
    Parameters: test (int) test number

    Returns dataframe with original data and SoC and R0

    '''
    rz.R0_fill(dfc)
    R0 = dfc[int(test)]["R0"]  # complete R0 column for given test
    df = extract("C", test)
    df["SoC"] = soc("C", test)
    df["R0"] = R0
    return df


def soc_ocv_fitted(cell, test):
    '''
    Parameters: cell (string), test (string)

    Returns coefficients of fitted polynomial between SoC and OCV
    '''
    soc = pt.soc_ocv(cell, test)["OCV"]
    ocv = pt.soc_ocv(cell, test)["SoC"]

    # Fit a polynomial of degree 4
    coefficients = np.polyfit(ocv, soc, 4)
    polynomial = np.poly1d(coefficients)

    # Generate fitted values for plotting
    # ocv_range = np.linspace(min(ocv), max(ocv), 100)
    #fitted_soc = polynomial(ocv)
    
    return coefficients


def deg4_model(x, a, b, c, d, e):
    return e * x ** 4 + d * x ** 3 + c * x ** 2 + b * x + a


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
    return [deg4_model(soc, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4]) for soc in soc]


def add_ocv_c(test):
    '''
    Parameters: test(string), cell test

    Returns dataframe containing OCV for cell c
    '''

    df = add_R0_c(test)
    df["OCV"] = calculate_ocv(df["SoC"], "C", test)
    return df

def calculate_model_voltage_c(test):
    '''
    Parameters: test(string), cell test

    Returns dataframe containing model voltage for 0th order Thevenin for cell C
    '''

    df = add_ocv_c(test)
    df["Model Voltage"] = [df["OCV"].iloc[i]+df["R0"].iloc[i]
                           * df["Current"].iloc[i] for i in range(len(df))]
    return df

def plot_r_soc(test):
    '''
    Parameters: test(string) test number
    
    Plots R as a function of SoC
    Returns nothing
    '''

    df = calculate_model_voltage_c(test)
    
    print(df)
    plt.plot(df["SoC"], df["R0"],'o')  # should be upside down U
    plt.show()

def calc_r0(test):
    
    r0 = [abs(add_ocv_c(test)["OCV"].iloc[i]-add_ocv_c(test)["Voltage"].iloc[i]) for i in range(len(add_ocv_c(test)))]
    return r0


if __name__ == "__main__":
    # print(soc("C","01"))
    
    
    df = calculate_model_voltage_c("09")
    """
    df.to_csv("r0")
    df = df[~np.isnan(df["R0"])]
    print(df)
    plt.plot(df["SoC"], df["R0"],"x")  # should be upside down U
    plt.show()
    """
    #print(calc_r0("04"))
    
    df.to_csv("0th_model_data")
    fig, axs = plt.subplots(2,1)
    axs[0].plot(df["Total Time"],df["Model Voltage"])
    axs[0].set_title("Model Voltage vs Time")
    axs[1].plot(df["Total Time"],df["Voltage"])
    axs[1].set_title("Voltage vs Time")

    plt.show()
    

    print(df)
    '''
    pt.soc_ocv("C", "05")
    pt.soc_ocv("D", "01")
    plt.show()
    '''
    
