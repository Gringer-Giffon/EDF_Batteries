# -------------------------------------------- INITIALISATION --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import plot as pt
import R0_identification_tianhe as rz
import math
import zeroth_order_modules as zom
from scipy.optimize import curve_fit

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

# -------------------------------------------- Files --------------------------------------------------


# Define the folder path containing the CSV files
folderPath = zom.data_file_path

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

# Filter CSV files belonging to Cell C and Cell D based on filename pattern
csvFiles_C = [f for f in csv_files if '_C_' in f]  # Files related to Cell C
csvFiles_D = [f for f in csv_files if '_D_' in f]  # Files related to Cell D

# Load CSV files into separate lists of Pandas DataFrames
dfc = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_C]  # DataFrames for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]  # DataFrames for Cell D


# ----------------------------------------------- CSV FILE EXTRACTION --------------------------------------------------------

def extract_all_steps(first, second, cell, test):
    '''
    Extracts data within a given step range for a specified cell and test.

    Parameters:
        first (int): First step in the range.
        second (int): Last step in the range.
        cell (str): "C" or "D" indicating the battery cell.
        test (str): Test number in the format "00", "01", etc.

    Returns:
        pd.DataFrame: Dataframe containing the extracted step data.
    '''
    data = extract(cell, test)  # Retrieve the full dataset for the specified cell and test
    step_data = data[data["Step"].isin(list(range(first, second+1)))]  # Filter for specified step range
    return step_data

def extract(cell, test):
    '''
    Extracts raw data from CSV files corresponding to a given cell and test.

    Parameters:
        cell (str): "C" or "D" indicating the battery cell.
        test (str): Test number in the format "00", "01", etc.

    Returns:
        pd.DataFrame: Dataframe containing the extracted data.
    '''
    
    # Find all CSV files that match the given cell and test
    file = [pd.read_csv(os.path.join(folderPath, f))
            for f in csv_files if f'_{cell.upper()}_' in f and str(test) in f]
    
    # Check if any matching files were found
    if not file:
        print("No test found for given cell and test. Cell entry must be 'C' or 'D'.")
        return None
    
    data = pd.concat(file)  # Combine multiple files into a single dataframe
    return data

def extract_step_2(first, second, cell, test):
    '''
    Extracts step data within a given range while removing duplicate step sequences.

    Parameters:
        first (int): First step in the range.
        second (int): Last step in the range.
        cell (str): "C" or "D" indicating the battery cell.
        test (str): Test number in the format "00", "01", etc.

    Returns:
        pd.DataFrame: Dataframe containing the extracted step data.
    '''
    data = extract(cell, test)  # Retrieve the full dataset
    step_data = data[data["Step"].isin(list(range(first, second+1)))]

    # Identify and remove duplicate step sequences
    step_indices = step_data.index
    for i in range(1, len(step_indices)):
        if step_indices[i] != step_indices[i - 1] + 1:  # Check for breaks in the sequence
            step_data = step_data.loc[step_indices[:i]]  # Keep only the first block
            break
    return step_data

def extract_step(first, second, cell, test):
    '''
    Extracts a specific step range from the dataset using preloaded data.

    Parameters:
        first (int): First step in the range.
        second (int): Last step in the range.
        cell (str): "C" or "D" indicating the battery cell.
        test (str or int): Test number.

    Returns:
        pd.DataFrame: Extracted step data.
    '''
    if isinstance(test, str):
        test = int(test)  # Ensure test is an integer

    # Select the correct dataframe based on the cell type
    if cell == 'C':
        df = dfc[test]
    elif cell == 'D':
        df = dfd[test]
    else:
        print("Invalid cell input. Must be 'C' or 'D'.")
        return None

    # Locate start and end of the step range
    t_s, _ = zom.locate_ti(df, first, 0)
    _, t_e = zom.locate_ti(df, second, 0)
    
    return zom.extract_ti(df, t_s, t_e)


# ---------------------------------------------------- SOC AND SOH ---------------------------------------------------

def q_remaining(cell, test):
    '''
    Computes the full discharge capacity for a given cell and test.

    Parameters:
        cell (str): "C" or "D" indicating the battery cell.
        test (str): Test number in the format "01", "02", etc.

    Returns:
        float: Full discharge capacity (Ah).
    '''
    
    # Extract full discharge data based on cell type
    if cell == "C":
        data = extract_step(26, 27, "C", test)
    elif cell == "D":
        data = extract_step(21, 23, "D", test)
    else:
        print("Invalid cell entry. Must be 'C' or 'D'.")
        return None

    # Calculate average current (I) and total discharge time (t)
    I = abs(data["Current"].mean())  # Take absolute value of current
    t = data["Total Time"].iloc[-1] - data["Total Time"].iloc[0]  # Time difference

    # Compute Q_remaining (discharge capacity) in Ah
    Q_remaining = I * t / 3600  

    return Q_remaining

def soc(cell, test):
    '''
    Computes the State of Charge (SoC) over time.

    Parameters:
        cell (str): "C" or "D" indicating the battery cell.
        test (str): Test number in the format "00", "01", etc.

    Returns:
        list: SoC values over time.
    '''
    
    data_full = extract(cell, test)  # Retrieve full dataset
    Q_remaining = q_remaining(cell, test)  # Get remaining capacity

    # Compute cumulative charge (Q_available) using numerical integration
    charge_integral = cumulative_trapezoid(data_full["Current"], data_full["Total Time"], initial=0)
    Q_available = [Q_remaining + (charge_integral[i] / 3600) for i in range(len(charge_integral))]

    # Normalize SoC values
    SOC = [(Q_available[i] / Q_remaining) - min(Q_available) / Q_remaining for i in range(len(Q_available))]

    return SOC

def soh(cell, test):
    '''
    Computes the State of Health (SoH) of a battery cell.

    Parameters:
        cell (str): "C" or "D" indicating the battery cell.
        test (str): Test number in the format "00", "01", etc.

    Returns:
        float: SoH value.
    '''
    
    print("Calculating SoH...")
    
    Q_remaining = q_remaining(cell, test)  # Get the remaining capacity
    print("Computed Q_remaining.")

    q_init = q_remaining(cell, "00")  # Get the initial capacity from test "00"
    print("Computed initial capacity (q_init).")

    SOH = Q_remaining / q_init  # Compute SoH as a ratio of current to initial capacity
    
    return SOH

# -------------------------------------------------------OCV--------------------------------------------------------------

def find_OCV(cell, test):
    """
    Identifies instances when the Open Circuit Voltage (OCV) is measured.
    Filters out duplicate readings that are too close in time.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier (e.g., "01", "02").
    
    Returns:
        pd.DataFrame: Filtered dataset with OCV measurement points.
    """
    # Set time threshold to filter duplicate measurements
    time_between_dupes = 600 if cell == "D" else 350 if cell == "C" else None
    
    if time_between_dupes is None:
        print("Invalid cell entry. Cell entry must be 'C' or 'D'")
        return None
    
    # Extract data and filter for points where current is zero (indicating OCV)
    data = extract(cell, test)[extract(cell, test)["Current"] == 0]
    
    # Remove points that are too close in time
    data_no_dupes = data.loc[~(data["Total Time"].diff().abs() < time_between_dupes)]
    return data_no_dupes

def soc_ocv(cell, test):
    """
    Plots OCV as a function of State of Charge (SoC) and returns corresponding data.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        pd.DataFrame: Data containing OCV and SoC values.
    """
    # Create a dataframe with Total Time and calculated SoC
    df_pre = pd.DataFrame({
        "Total Time": extract(cell, test)["Total Time"],
        "SoC": soc(cell, test)
    })
    
    # Extract OCV measurement points
    pulses = zom.extract_pulses(cell, test)
    col1 = [pulse["Total Time"].iloc[1] for pulse in pulses]  # Extract measurement times
    col2 = [pulse["Current"].iloc[1] for pulse in pulses]  # Extract corresponding current
    col3 = zom.measure_OCV(pulses)  # Measure OCV at extracted points
    
    # Map SoC values to OCV measurement times
    col4 = [df_pre["SoC"].loc[df_pre["Total Time"] == i].values[0] if i in df_pre["Total Time"].values else np.nan for i in col1]
    
    # Create final dataframe with OCV and SoC data
    return pd.DataFrame({"Total Time": col1, "Current": col2, "OCV": col3, "SoC": col4})

def soc_ocv_fitted(cell, test):
    """
    Fits a polynomial function between SoC and OCV for given test data.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        np.poly1d: Polynomial function mapping SoC to OCV.
    """
    df = soc_ocv(cell, test)
    soc, ocv = df["SoC"], df["OCV"]
    
    # Fit a 7th-degree polynomial to the data (it worked best)
    coefficients = np.polyfit(soc, ocv, 7)
    return np.poly1d(coefficients)

    return polynomial

def calculate_ocv(soc, cell, test):
    """
    Computes OCV values from SoC using the fitted polynomial model.
    
    Parameters:
        soc (list): List of SoC values.
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        list: List of calculated OCV values.
    """
    poly = soc_ocv_fitted(cell, test)  # Get the fitted polynomial function
    return [poly(s) for s in soc]  # Apply function to given SoC values

def add_ocv(cell, test):
    """
    Adds computed OCV values to the dataset.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        pd.DataFrame: Dataframe with added OCV column.
    """
    df = add_R0(cell, test)  # Start with dataset that includes R0 values
    df["OCV"] = calculate_ocv(df["SoC"], cell, test)  # Compute and add OCV
    return df

# -------------------------------------------------------R0---------------------------------------------------------------


soh_value = 1

def add_R0(cell, test):
    """
    Computes and adds R0 (internal resistance) to the dataset.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        pd.DataFrame: Dataset with SoC and R0 values.
    """
    global soh_value  # Use global SoH value
    df = extract(cell, test)  # Extract data
    df["SoC"] = soc(cell, test)  # Add SoC values
    df["R0"] = [zom.f(soc_val, soh_value) for soc_val in df["SoC"]]  # Compute R0
    return df

# ----------------------------------------------------MODEL 0--------------------------------------------------------------

def calculate_model_voltage_0(cell, test):
    """
    Computes the 0th order Thevenin model voltage.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        pd.DataFrame: Dataset with added Model Voltage 0 values.
    """
    global soh_value
    soh_value = soh(cell, test)  # Get SoH value
    df = add_ocv(cell, test)  # Add OCV and R0 data
    
    # Compute model voltage using the Thevenin equation
    df["Model Voltage 0"] = [df["OCV"].iloc[i] - df["R0"].iloc[i] * abs(df["Current"].iloc[i]) if df["Current"].iloc[i] < 0 
                               else df["OCV"].iloc[i] + df["R0"].iloc[i] * abs(df["Current"].iloc[i]) for i in range(len(df))]
    return df

# ---------------------------------------------------------R1------------------------------------------------------------------

def find_R1(cell, test):
    '''
    Identifies times when the capacitor acts, allowing R1 measurement.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier in the format "01", "02", etc.
    
    Returns:
        DataFrame: Contains times where R1 can be measured.
    '''
    
    # Determine step based on cell type
    if cell == "D":
        step = 6
    elif cell == "C":
        step = 7
    else:
        print("Invalid cell entry. Must be 'C' or 'D'.")
        return None
    
    # Extract data and filter for the correct step
    data = add_ocv(cell, test)[add_ocv(cell, test)["Step"] == step]
    return data

def measure_r1(cell, test):
    '''
    Calculates R1 values at identified capacitor action times.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        DataFrame: Contains measured R1 values.
    '''
    df = find_R1(cell, test)
    
    # Identify discontinuities where index jumps (indicating a break in the data)
    discontinuities = df.index.to_series().diff().gt(1)
    
    # Split DataFrame at discontinuities
    splits = []
    start_idx = 0
    for i, discontinuity in enumerate(discontinuities):
        if discontinuity:
            splits.append(df.iloc[start_idx:i])
            start_idx = i
    
    # Compute R1 for each split
    for split in splits:
        split["R1"] = abs(split["Voltage"].iloc[0] - split["Voltage"].iloc[-1]) / abs(split["Current"].iloc[-1])
    
    return pd.concat(splits)

def soc_R1_fitted(cell, test):
    '''
    Fits a polynomial relationship between SoC and R1.
    
    Parameters:
        cell (str): "C" or "D"
        test (str): Test identifier.
    
    Returns:
        np.poly1d: Polynomial function mapping SoC to R1.
    '''
    df = measure_r1(cell, test)
    soc = df["SoC"]
    R1 = df["R1"]
    
    # Fit a 7th-degree polynomial (adjustable for best fit)
    coefficients = np.polyfit(soc, R1, 7)
    return np.poly1d(coefficients)

def calculate_r1(soc, cell, test):
    '''
    Calculates R1 values using the fitted polynomial function.

    Parameters:
        soc (list): List of SoC values.
        cell (str): "C" or "D"
        test (str): Test identifier.

    Returns:
        list: Computed R1 values corresponding to given SoC values.
    '''
    # Retrieve the polynomial model fitted for R1
    poly = soc_R1_fitted(cell, test)
    return [poly(s) for s in soc]

def add_r1(cell, test):
    '''
    Adds R1 (resistance) values to the dataset based on SoC.
    
    Parameters:
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.
    
    Returns:
        DataFrame: Data with an additional column for R1 values.
    '''
    df = add_ocv(cell, test)  # Load dataset with SoC and OCV values
    df["R1"] = calculate_r1(df["SoC"], cell, test)  # Compute R1 values
    return df

# -------------------------------------------------------TAU----------------------------------------------------------------

def find_tau(cell, test):
    '''
    Filters data to identify time periods where the tau parameter can be measured.
    
    Parameters:
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.
    
    Returns:
        DataFrame: Subset of data where tau can be computed.
    '''
    df = add_r1(cell, test)  # Ensure R1 is included in dataset
    
    # Different steps correspond to tau measurement conditions for different cells
    step_condition = 6 if cell == "D" else 7 if cell == "C" else None
    if step_condition is None:
        raise ValueError("Invalid cell. Must be 'C' or 'D'.")
    
    return df[df["Step"] == step_condition]

def measure_tau(cell, test):
    '''
    Computes tau (time constant) by analyzing voltage drop over time.
    
    Parameters:
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.
    
    Returns:
        DataFrame: Data with computed tau values.
    '''
    df = find_tau(cell, test)  # Get relevant data for tau calculation
    
    # Identify discontinuities in time series to segment data
    discontinuities = df.index.to_series().diff().gt(1)
    
    splits = []  # Store segmented dataframes
    start_idx = 0
    for i, discontinuity in enumerate(discontinuities):
        if discontinuity:
            splits.append(df.iloc[start_idx:i])  # Save segment
            start_idx = i
    
    # Compute tau for each segment
    for split in splits:
        if split.empty:
            continue
        
        final_voltage = split["Voltage"].min()  # Minimum voltage in the segment
        initial_voltage = split["Voltage"].iloc[0]  # Initial voltage
        target_voltage = initial_voltage - 0.63 * abs(initial_voltage - final_voltage)  # Exponential decay threshold (1 - exp(-1))
        
        # Find the time where voltage reaches 63% of the total drop
        idx = (split["Voltage"] - target_voltage).abs().idxmin()
        split["tau"] = split["Total Time"].loc[idx] - split["Total Time"].iloc[0] if idx is not None else np.nan
    
    return pd.concat(splits)

def soc_tau_fitted(cell, test):
    '''
    Fits a polynomial between SoC and tau values to model their relationship.
    
    Parameters:
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.
    
    Returns:
        Polynomial: Fitted polynomial function for estimating tau.
    '''
    df = measure_tau(cell, test)
    
    soc = df["SoC"]
    tau = df["tau"]
    
    # Fit a polynomial to the log-transformed tau values (degree chosen to avoid overfitting)
    coefficients = np.polyfit(soc, np.log(tau), 5)
    return np.poly1d(coefficients)

def calculate_tau(soc_values, cell, test):
    '''
    Computes estimated tau values for given SoC (State of Charge) values 
    using the fitted polynomial model.

    Parameters:
        soc_values (list): List of SoC values (state of charge percentages).
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.

    Returns:
        list: Estimated tau values corresponding to SoC values.
    '''
    # Retrieve the polynomial model fitted for the given cell and test
    poly = soc_tau_fitted(cell, test)

    # Compute tau values by applying the polynomial to each SoC value
    return [poly(soc) for soc in soc_values]

def add_tau(cell, test):
    '''
    Adds computed tau values to the dataset.

    Parameters:
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.

    Returns:
        DataFrame: Data with an additional "tau" column.
    '''
    # Load the dataset with R1 values added
    df = add_r1(cell, test)

    # Compute and add the tau values to the dataset
    df["tau"] = calculate_tau(df["SoC"], cell, test)
    return df

def add_c1(cell, test):
    '''
    Computes and adds C1 (capacitance) values to the dataset using the formula C1 = tau / R1.

    Parameters:
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.

    Returns:
        DataFrame: Data with an additional "C1" column.
    '''
    # Load the dataset with tau values added
    df = add_tau(cell, test)

    # Compute and add C1 values to the dataset
    df["C1"] = df["tau"] / df["R1"]
    return df

def time_pulses_calc(df):
    '''
    Identifies and tracks time pulses where current changes significantly.

    Parameters:
        df (DataFrame): Battery dataset containing 'Current' and 'Total Time' columns.

    Returns:
        DataFrame: Data with an additional 'Time' column tracking pulses.
    '''
    # Define the threshold for detecting significant current changes
    threshold = 10  # Amperes (example threshold value)

    # Initialize the time tracker with the first total time value
    time = df['Total Time'].iloc[0]
    time_list = [time]

    # Loop through the dataset to track time pulses
    for i in range(len(df) - 1):
        # Check if there is a significant change in current
        if abs(df['Current'].iloc[i] - df['Current'].iloc[i + 1]) >= threshold:
            time = 0  # Reset time on significant current change

        # Accumulate the time difference between consecutive entries
        time += df['Total Time'].iloc[i + 1] - df['Total Time'].iloc[i]
        time_list.append(time)

    # Add the computed time pulses to the DataFrame
    df['Time'] = time_list
    return df

# -----------------------------------------------------MODEL 1------------------------------------------------------------

def calculate_model_voltage_1(cell, test):
    '''
    Calculates model voltage using the 0th and 1st order approximations.

    Parameters:
        cell (str): "C" or "D", representing the battery type.
        test (str): Test identifier.

    Returns:
        DataFrame: Dataframe including 0th order and 1st order voltage values.
    '''
    global soh_value
    soh_value = soh(cell, test)  # Compute the state of health (SOH) for the cell

    # Compute the initial model voltage (0th order)
    df = calculate_model_voltage_0(cell, test)
    print("calculated model 0")

    # Add C1, R1, and tau values to the dataframe
    df1 = add_c1(cell, test)
    print("df1", df1)

    # Merge df1 into df to align by "Total Time" directly
    df = df.merge(df1[["Total Time", "R1", "tau", "C1"]], on="Total Time", how="left")
    print("merged", df)

    # Calculate model voltage using vectorized operations
    df = time_pulses_calc(df)
    df["Model Voltage 1"] = [df["Model Voltage 0"].iloc[i] - df["R1"].iloc[i] * 
        abs(df1["Current"].iloc[i]) * (1 - np.exp(-df["Time"].iloc[i] / df["tau"].iloc[i]))
        if df1["Current"].iloc[i] < 0 else df["Model Voltage 0"].iloc[i] + df["R1"].iloc[i] * 
        abs(df1["Current"].iloc[i]) * (1 - np.exp(-df["Time"].iloc[i] / df["tau"].iloc[i]))
        for i in range(len(df))]

    print(df["R1"] * df1["Current"] * (1 - np.exp(df["Total Time"] / (df["R1"] * df["C1"]))))

    # Handle NaN results if tau was NaN or zero (optional recovery step)
    # df["Model Voltage 1"].fillna(df["Model Voltage 0"], inplace=True)

    # Save the dataframe for further analysis
    df.to_csv("model1")
    print(df[df["Step"].isin(range(6, 9))])

    return df

if __name__ == "__main__":

 # Uncomment and use these sections as needed for plotting or data testing

    # pt.soc_ocv("C", "05")
    # pt.soc_ocv("D", "01")
    plt.show()
