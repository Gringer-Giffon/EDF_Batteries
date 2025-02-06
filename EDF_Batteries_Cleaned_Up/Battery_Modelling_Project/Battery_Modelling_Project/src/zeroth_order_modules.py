import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
import os

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/cells_data') # Reaching the datas in the data folder
folderPath = data_file_path

csv_files = [f for f in os.listdir(data_file_path) if f.endswith('.csv')]

csvFiles_C = [f for f in csv_files if '_C_' in f]
csvFiles_D = [f for f in csv_files if '_D_' in f]

dfc = [pd.read_csv(os.path.join(data_file_path, file)) for file in csvFiles_C]  # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(data_file_path, file)) for file in csvFiles_D]  # Dataframes for Cell D

cell_c_R0_coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/R0_models/coefficients_R0_3.npy'))  # Coefficients for 0th Order model of Cell C
cell_d_R0_coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/R0_models/coefficients_R0_cell_d_2.npy'))   # Coefficients for 0th Order model of Cell D

data_pack_c = (cell_c_R0_coefficients, 7)
data_pack_d = (cell_d_R0_coefficients, 7)

cat =  """      ／＞　 フ
      | 　_　_|  
    ／` ミ＿xノ  
   /　　　　 | 
  /　 ヽ　　 ﾉ
 │　　|　|　| 
／￣|　　 |　| 
(￣ヽ＿_ヽ_)__) 
＼二)"""      # Error Context


# -------------------------------------------- Ready to Use R0 and Cost Function -----------------------------------------------

def f(x, y, cell='Cell C', first_order=False):
    '''
    Compute the modeled Resistence Zero based on the provided SoC (State of Charge) and SoH (State of Health) 
    for a specified cell type using a polynomial model.

    Parameters:
    - x (float): SoC (State of Charge) of the cell(s).
    - y (float): SoH (State of Health) of the cell(s).
    - cell (str, optional): The type of the cell ('Cell C' or 'Cell D'). (Defaults to 'Cell C')
    - first_order (boolean): Detect if it's used as a term in the first order function. (Defaults to False)

    Returns:
    - float: The modeled resistance for the given SoC and SoH. 

    Raises:
    - ValueError: If an unknown cell type is provided (neither 'Cell C' nor 'Cell D').
    '''
    if cell == 'Cell D':
        (coefficients, degree) = data_pack_d
    elif first_order:
        coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/R0_models/coefficients_R0_1.npy'))
        degree = 7
    elif cell == 'Cell C':
        (coefficients, degree) = data_pack_c
    else:
        print(cat, '\nError: Unknown Cell')
        return 
    
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            x_power = d - i
            y_power = i
            terms.append((x ** x_power) * (y ** y_power))
    return np.dot(terms, coefficients)


def cost(df, column_name='Model Voltage'):
    '''
    Calculate the error between the modeled voltage and the actual voltage in the given DataFrame, 
    and add the error as a new column to the DataFrame.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing the actual voltage ('Voltage') and the modeled voltage 
      (specified by 'column_name', default is 'Model Voltage').
    - column_name (str, optional): The column name for the modeled voltage data. Defaults to 'Model Voltage'.

    Returns:
    - pandas.DataFrame: The input DataFrame with an additional column 'Error' containing the error values 
      (modeled voltage - actual voltage).
    
    Raises:
    - Prints an error message and returns the original DataFrame if either 'Voltage' or the specified 
      `column_name` does not exist in the DataFrame.
    '''
    if column_name not in df.columns or 'Voltage' not in df.columns:
        print(cat, '\nError: Unknown Column name')
        return df
    else:
        error = [df[column_name].iloc[i] - df['Voltage'].iloc[i] for i in range(len(df))]
        df.loc[:,'Error'] = error
        return df

def mean_abs_cost(df):
    '''
    Calculate the mean absolute error between the modeled voltage and the actual voltage in the given DataFrame,

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing the actual voltage ('Voltage') and the modeled voltage 
      (specified by 'column_name', default is 'Model Voltage').

    Returns:
    - float: the mean absolute error between the modeled voltage and actual voltage in the given DataFrame.

    Raises:
    - Prints an error message and returns the original DataFrame if column 'Error' does not exist in the DataFrame.
    '''
    if 'Error' not in df.columns:
        print(cat, '\nPls Calculate the Error first')
        return 0
    else:
        df['abs err'] = [abs(i) for i in df['Error']]
        avg_error = df['abs err'].mean()
        return avg_error


# ------------------------------------------- Ready to Use OCV Function --------------------------------------------------------------

def OCV_f(x, y, cell='Cell C'):
    '''
    Compute the modeled Open Circuit Voltage based on the provided SoC (State of Charge) and SoH (State of Health) 
    for a specified cell type using a polynomial model.

    Parameters:
    - x (float): SoC (State of Charge) of the cell(s).
    - y (float): SoH (State of Health) of the cell(s).
    - cell (str, optional): The type of the cell ('Cell C' or 'Cell D'). Defaults to 'Cell C'.

    Returns:
    - float: The modeled voltage for the given SoC and SoH. 

    Raises:
    - ValueError: If an unknown cell type is provided (neither 'Cell C' nor 'Cell D').
    '''
    if cell == 'Cell C':
        coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/OCV_models/coefficients_OCV.npy'))
        degree = 5
    elif cell == 'Cell D':
        coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/OCV_models/coefficients_OCV_cell_d.npy'))
        degree = 3
    else:
        print(cat, '\nError: Unknown Cell')
        return
    
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            x_power = d - i
            y_power = i
            terms.append((x ** x_power) * (y ** y_power))
    return np.dot(terms, coefficients)


# -----------------------------------------------CSV FILE EXTRACTION--------------------------------------------------------

def extract_ti(df, start, end):
    '''
    Extracts a subset of the DataFrame based on a specified time interval.

    Parameters:
    df (DataFrame): The DataFrame containing time series data with a 'Total Time' column.
    start (float or int): The start time for the extraction.
    end (float or int): The end time for the extraction.

    Returns:
    DataFrame: A DataFrame containing rows where 'Total Time' is between start and end.
    '''
    return df[(df['Total Time']>=start) & (df['Total Time']<=end)]


def locate_ti(df, step, pattern=0, offset=0, offset_2=0, find_end=False):
    '''
    Identifies the start and end times of a specific step in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing step data with a 'Step' and 'Total Time' column.
    step (int or float): The step number to locate within the DataFrame.
    pattern (int, optional): The occurrence index of the step to locate (0 for first, 1 for second, etc.) (default is 0).
    offset (int, optional): Value to subtract from the start time (default is 0).
    offset_2 (int, optional): Value to add to the end time (default is 0).
    find_end (boolean, optional): Value to detect if the function is looking for the last pattern of that step (default is False). 

    Returns:
    tuple: A tuple containing the adjusted start and end times of the specified step occurrence.
    '''
    mask = df['Step'] == step
    df['start'] = mask& ~mask.shift(1, fill_value=False)
    df['end'] = mask& ~mask.shift(-1, fill_value=False)
    start_indices = df.index[df['start']].tolist()
    end_indices = df.index[df['end']].tolist()
    if find_end:
        pattern = len(start_indices)-1
    return df['Total Time'][start_indices[pattern]-offset], df['Total Time'][end_indices[pattern]+offset_2]


def pulses_extract(df):
    '''
    Extracts the region of the DataFrame where pulsed patterns appear, 
    specifically focusing on step 7.

    Parameters:
    df (DataFrame): The DataFrame containing time series data with 'Step' and 'Total Time' columns.

    Returns:
    DataFrame: A subset of the original DataFrame containing the time interval where pulsed patterns occur.
    '''
    t_s, x = locate_ti(df, 7)
    x, t_e = locate_ti(df, 7, find_end=True)
    t_s, t_e = locate_ti(df, 6, pattern=7, offset=2, offset_2=30)
    return extract_ti(df, t_s, t_e)


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
    """
    Extracts data between two specified steps (first and second) from a test dataset.

    Parameters:
    first (int or float): The start time or index for extraction.
    second (int or float): The end time or index for extraction.
    cell (str): Specifies which dataset to use ('C' or 'D').
    test (int or str): The test identifier. If provided as a string, it's converted to an integer.

    Returns:
    DataFrame: The extracted data between the specified steps.
    """
    if type(test) == str:
        test = int(test)
    if cell == 'C':
        df = dfc[test]
    elif cell == 'D':
        df = dfd[test]
    else:
        print('oops')
        return [0, 0]
    t_s, x = locate_ti(df, first, 0)
    x, t_e = locate_ti(df, second, 0)
    return extract_ti(df, t_s, t_e)


def extract_pulses(cell, test):
    """
    Extracts pulse data segments from a test dataset based on the current values.

    Parameters:
    cell (str): Specifies which dataset to use ('C' or 'D').
    test (int or str): The test identifier.

    Returns:
    list: A list of DataFrames, each representing an individual pulse.
    """
    df_pre = extract(cell, test)
    pulse_df = []
    df = df_pre[abs(df_pre["Current"]).between(
        max(df_pre["Current"])-1, max(df_pre["Current"]+1))]  # Selects all pulse indexes
    df = df[df.index.to_series().diff().gt(1)]
    for pulse_time in df["Total Time"]:
        if cell == "C":
            pulse_df.append(df_pre[(df_pre["Total Time"] >= pulse_time - 20)
                            & (df_pre["Total Time"] <= pulse_time + 15)])
        elif cell == "D":
            pulse_df.append(df_pre[(df_pre["Total Time"] >= pulse_time - 200)
                            & (df_pre["Total Time"] <= pulse_time + 150)])
        else:
            print("Invalid choice of cell")
            return None
    return pulse_df


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
    print("I am in the SOH function")
    Q_remaining = q_remaining(cell, test)
    print("I have Q remaining")
    q_init = q_remaining(cell, "00")
    print("I have q initial")
    SOH = Q_remaining / q_init
    # print(Q_remaining, q_init)
    return SOH


def general_soc(df, time=10, Q=109890.8699999999):
    # Assuming the battery was fully charged at the beginning (SoC = 1 at Total Time = 0)
    # The constant, Q, for example, was 109890.8699999999 during the first cycle of Cell C
    SoC = []
    soc_cur = 1
    for i in range(len(df)):
        SoC.append(soc_cur)
        soc_cur = soc_cur + (df['Current'][i]*time/Q)
    return SoC

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


def measure_OCV(pulse_df):
    """
    Measures the Open Circuit Voltage (OCV) from a list of pulse dataframes.
    
    Parameters:
    pulse_df (list of DataFrames): A list where each DataFrame contains pulse data with a 'Voltage' column.
    
    Returns:
    list: A list of OCV values extracted from each pulse.
    """
    spikes = []
    for pulse in pulse_df:
        voltage_diff = np.diff(pulse["Voltage"].values)

        # threshold for detection
        threshold = 0.05

        # index of spike
        spikes.append(np.argmax(np.abs(voltage_diff) > threshold))
    ocv = []
    for i in range(len(pulse_df)):
        ocv.append(pulse_df[i]["Voltage"].iloc[spikes[i]])
    return ocv


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
    #col1 = find_OCV(str(cell), str(test))["Total Time"]
    #col2 = find_OCV(str(cell), str(test))["Current"]
    #col3 = find_OCV(str(cell), str(test))["Voltage"]

    col1 = [pulse["Total Time"].iloc[1] for pulse in extract_pulses(cell,test)]
    col2 = [pulse["Current"].iloc[1] for pulse in extract_pulses(cell,test)]
    col3 = measure_OCV(extract_pulses(cell,test))

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
    

def add_ocv(cell, test):
    '''
    Parameters: cell (string) "C" or "D", test(string), cell test

    Returns dataframe containing OCV
    '''

    df = add_R0(cell, test)  # Data frame with R0
    df["OCV"] = calculate_ocv(df["SoC"], cell, test)
    return df
