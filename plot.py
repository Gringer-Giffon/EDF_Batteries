import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import data as dt
from mpl_toolkits.mplot3d import Axes3D
import R0_fit as r_calc
import OCV_fit as v_calc

# ------------------------------------------------ PLOTTING CONFIGURATION ------------------------------------------------ #
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.color'] = 'darkred'  # Global text color
plt.rcParams['axes.labelcolor'] = 'darkred'  # Axis label color
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['darkred'])

# ------------------------------------------------ DIRECTORY & FILE EXTRACTION ------------------------------------------------ #
directory = "./cells_data"
centrale_red = "#AF4458"  # Custom color for plots

# Extracts all CSV file names in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# ------------------------------------------------ FUNCTION: PLOT TESTS ------------------------------------------------ #

def plot_test(cell, test):
    '''
    Plots voltage, current, and step for a given test.
    
    Parameters:
        cell (str): Cell identifier ('C' or 'D').
        test (str): Test identifier (e.g., '00', '01', etc.).
    
    Returns:
        None
    '''
    
    # Extract relevant data from the dataset
    plot_data = dt.extract(cell, test)
    time = plot_data["Total Time"]
    current = plot_data["Current"]
    voltage = plot_data["Voltage"]
    step = plot_data["Step"]
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))     # Create a figure with 3 subplots
    fig.suptitle(f"Cell: {cell.upper()}, Test: {test}", fontsize=14, fontweight='bold')
    
    # Plot Current vs Time
    axs[0].plot(time, current, "o", ms=1, color=centrale_red)
    axs[0].set_title("Current vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Current (A)")
    axs[0].grid(True)
    
    # Plot Voltage vs Time
    axs[1].plot(time, voltage, "o", ms=1, color=centrale_red)
    axs[1].set_title("Voltage vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Voltage (V)")
    axs[1].grid(True)
    
    # Plot Step vs Time
    axs[2].plot(time, step, "o", ms=1, color=centrale_red)
    axs[2].set_title("Step vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Step")
    axs[2].grid(True)
    
    plt.subplots_adjust(hspace=0.5)
    plt.show() # Display 
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

def plot_soh_cell_c_cell_d():
    """
    Plots the State of Health (SoH) of both Cell C and Cell D.
    
    - The x-axis represents the test numbers.
    - The y-axis represents the SoH as a percentage.
    """
    
    soh_c = []
    soh_d = []
    
    # Compute SoH for Cell C over tests 0 to 23
    for test in range(24): 
        soh_i = dt.soh("C", test) * 100  # Convert to percentage
        soh_c.append(soh_i)
    
    # Compute SoH for Cell D over tests 0 to 13
    for test in range(14): 
        soh_i = dt.soh("D", test) * 100  # Convert to percentage
        soh_d.append(soh_i)
    
    # Create subplots for Cell C and Cell D
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))  
    fig.suptitle("State of Health (SoH) of Cells C and D", fontsize=14)  

    # Plot SoH for Cell C
    axs[0].plot(range(24), soh_c, "g-o", markersize=4)
    axs[0].set_ylabel("SoH [%]")
    axs[0].set_xlabel("Test Number")
    axs[0].set_title("SoH of Cell C")
    axs[0].grid(True)

    # Plot SoH for Cell D
    axs[1].plot(range(14), soh_d, "g-o", markersize=4)
    axs[1].set_ylabel("SoH [%]")
    axs[1].set_xlabel("Test Number")
    axs[1].set_title("SoH of Cell D")
    axs[1].grid(True)

    # Adjust layout and show plot
    plt.subplots_adjust(hspace=0.5)
    plt.show()

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

    # Plot Current over time
    fig, axs= plt.subplots(2,1)
    axs[0].plot(df["Total Time"],df["Current"])

    # Plot SoC over Time
    axs[1].plot(df["Total Time"], df["SoC"], label="SoC Curve")

    # Annotate key points
    axs[1].annotate("Pulse Start", xy=(pulse_start, pulse_soc), xytext=(pulse_start, pulse_soc - 20),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=9)

    axs[1].annotate("Pulse End", xy=(full_charge_start, full_charge_soc),
                 xytext=(full_charge_start, full_charge_soc + 20),
                 arrowprops=dict(facecolor='green', arrowstyle='->'), fontsize=9)
    
    """
    axs[1].annotate("Discharge End", xy=(full_discharge_end-40, full_discharge_soc),
                 xytext=(full_discharge_end, full_discharge_soc - 10),
                 arrowprops=dict(facecolor='green', arrowstyle='-'), fontsize=9)
    """

    # Add labels, legend, and grid

    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Current (%)")
    axs[0].set_title(f"Current vs Time: cell {cell} test {test}")
    
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("SOC (%)")
    axs[1].set_title(f"State of Charge vs Time: cell {cell} test {test}")
    
    plt.subplots_adjust(hspace = 1)
    
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
    """
    Plots the SoC-OCV data and overlays the fitted polynomial model.

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    plot_soc_ocv(cell, test)  # Plot raw SoC-OCV data

    polynomial = dt.soc_ocv_fitted(cell, test) # Get the fitted polynomial model
    x = np.linspace(0, 1, 100)

    plt.plot(x, polynomial(x), label="Fitted OCV Model") # Plot the fitted curve
    plt.legend()

def model_data_soc_ocv_soh(cell):
    
    plot_model_data_soc_ocv(cell, 0)
    plot_model_data_soc_ocv(cell, 5)
    plot_model_data_soc_ocv(cell, 11)
    
    plt.show()
    
    
def model_data_r0_soc_soh():
    f_vectorized = np.vectorize(r_calc.f)
    
    x = np.linspace(0,1,100)
    y = np.linspace(0.9,1,100)
    x,y = np.meshgrid(x,y)
    z= f_vectorized(x,y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x,y,z,c=z,cmap='viridis')
    
    ax.set_xlabel('SoC Value')
    ax.set_ylabel('SoH Value')
    ax.set_zlabel('Resistance Zero (Ohm)')
    ax.set_title('SoC and SoH Effects on R0')
    
    ax.view_init(elev=20, azim=60)
    
    plt.show()


def model_data_ocv_soc_soh(cell):

    if cell == "C":
        print("Cell C")
    if cell == "D":
        print("Cell D")


    f_vectorized = np.vectorize(v_calc.f)
    
    x = np.linspace(0,1,30)
    y = np.linspace(0.9,1,30)
    x, y = np.meshgrid(x, y)
    z = f_vectorized(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Change from scatter to surface plot
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('SoC Value')
    ax.set_ylabel('SoH Value')
    ax.set_zlabel('Open Circuit Voltage (V)')
    ax.set_title('SoC and SoH Effects on OCV')

    ax.view_init(elev=20, azim=240)
    
    plt.show()
     

def plot_model_voltage_0(cell, test):
    """
    Plots the measured voltage and the 0th-order model voltage.

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    df1 = dt.extract(cell, test)  # Extract raw test data
    df = dt.calculate_model_voltage_0(cell, test)  # Compute 0th-order model

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    
    # Plot measured voltage
    axs[0].plot(df1["Total Time"], df1["Voltage"], color="blue")
    axs[0].set_title("Measured Voltage Over Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Voltage (V)")

    # Plot modeled voltage
    axs[1].plot(df["Total Time"], df["Model Voltage 0"], color="green")
    axs[1].set_title("0th-Order Model Voltage Over Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Voltage (V)")

    plt.tight_layout()
    plt.show()

def plot_model_voltage_1(cell, test):
    """
    Plots the measured voltage, 0th-order model, and 1st-order model voltage.

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    df = dt.calculate_model_voltage_1(cell, test)  # Compute 1st-order model

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    # Measured voltage
    axs[0].plot(df["Total Time"], df["Voltage"], color="blue")
    axs[0].set_title("Measured Voltage Over Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Voltage (V)")

    # 0th-order model voltage
    axs[1].plot(df["Total Time"], df["Model Voltage 0"], color="green")
    axs[1].set_title("0th-Order Model Voltage Over Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Voltage (V)")

    # 1st-order model voltage
    axs[2].plot(df["Total Time"], df["Model Voltage 1"], color="red")
    axs[2].set_title("1st-Order Model Voltage Over Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Voltage (V)")

    plt.tight_layout()
    plt.show()

def plot_simultaneous_0(cell, test):
    """
    Overlays the measured voltage and the 0th-order model voltage on the same plot.

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    df = dt.calculate_model_voltage_0(cell, test)  # Compute 0th-order model

    plt.figure(figsize=(8, 5)) # plot 
    plt.plot(df["Total Time"], df["Voltage"], "b", label="Measured Voltage")
    plt.plot(df["Total Time"], df["Model Voltage 0"], "g", label="Model Voltage 0")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Measured vs. 0th-Order Model Voltage")
    plt.legend()
    plt.show()

def plot_simultaneous(cell, test):
    """
    Overlays the measured voltage, 0th-order model voltage, and 1st-order model voltage.

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    df = dt.calculate_model_voltage_1(cell, test)  # Compute 1st-order model

    plt.figure(figsize=(8, 5)) # plot 
    plt.plot(df["Total Time"], df["Voltage"], "b", label="Measured Voltage")
    plt.plot(df["Total Time"], df["Model Voltage 0"], "g", label="Model Voltage 0")
    plt.plot(df["Total Time"], df["Model Voltage 1"], "r", label="Model Voltage 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Measured vs. Modeled Voltage (0th & 1st Order)")
    plt.legend()
    plt.show()

def plot_simultaneous_1(cell,test):
    df = dt.calculate_model_voltage_1(cell, test)
    plt.plot(df["Total Time"], df["Voltage"], "b")
    plt.plot(df["Total Time"], df["Model Voltage 1"], "r")
    plt.legend(["Data","Model Voltage 1"])


def plot_r0_soc(cell, test):
    """
    Plots the internal resistance R0 as a function of the State of Charge (SoC).

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    df = dt.calculate_model_voltage_0(cell, test)  # Compute 0th-order model

    plt.figure(figsize=(8, 5))
    plt.plot(df["SoC"], df["R0"], 'o', color="purple")  # Should form an L/U-shape
    plt.xlabel("State of Charge (SoC)")
    plt.ylabel("Internal Resistance R0 (Ω)")
    plt.title("R0 vs. SoC")
    plt.grid(True)
    plt.show()

def plot_r1_soc(cell, test):
    """
    Plots the first-order resistance R1 as a function of the State of Charge (SoC).

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    df = dt.calculate_model_voltage_1(cell, test)  # Compute 1st-order model

    plt.figure(figsize=(8, 5)) # plot 
    plt.plot(df["SoC"], df["R1"], 'o', color="blue")  # Should form an L/U-shape
    plt.xlabel("State of Charge (SoC)")
    plt.ylabel("First-Order Resistance R1 (Ω)")
    plt.title("R1 vs. SoC")
    plt.grid(True)
    plt.show()

def plot_tau_soc(cell, test):
    """
    Plots the time constant τ (tau) as a function of the State of Charge (SoC).

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    df = dt.calculate_model_voltage_1(cell, test)  # Compute 1st-order model

    plt.figure(figsize=(8, 5)) # plot 
    plt.plot(df["SoC"], df["tau"], 'o', color="green") 
    plt.xlabel("State of Charge (SoC)")
    plt.ylabel("Time Constant τ (s)")
    plt.title("Tau vs. SoC")
    plt.grid(True)
    plt.show()

def plot_c1_soc(cell, test):
    """
    Plots the first-order capacitance C1 as a function of the State of Charge (SoC).

    Parameters:
        cell (str): Cell identifier ("C" or "D").
        test (str): Test number.
    """
    # Extract model data for the given cell and test
    df = dt.calculate_model_voltage_1(cell, test)

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["SoC"], df["C1"], "o", color="purple")  # Expected inverted U-shape
    plt.xlabel("State of Charge (SoC)")
    plt.ylabel("First-Order Capacitance C1 (F)")
    plt.title("C1 vs. SoC")
    plt.grid(True)
    plt.show()

# ------------------------------------------------ CALLING FUNCTIONS ------------------------------------------------ #


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
    #plot_soc("D","01")
    #plt.show
    plot_simultaneous("C","01")
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
