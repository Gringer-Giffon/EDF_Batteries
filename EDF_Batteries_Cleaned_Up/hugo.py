import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plot as pt

# -----------------------------------------------VARIABLES--------------------------------------------------------

folderPath = f'./cells_data'

csv_files = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csv_files if '_C_' in f]
csvFiles_D = [f for f in csv_files if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file))
       for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]

# -----------------------------------------------FUNCTIONS--------------------------------------------------------

def plot_voltage(df):
    """Plots the full voltage response over time."""
    #plot_data = dt.extract(cell, test)

    time = df["Total Time"]
    voltage = df["Voltage"]

    plt.figure(figsize=(10, 5))
    plt.plot(time, voltage, label="Voltage Response", color="b")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Full Voltage Response")
    plt.legend()
    plt.grid(True)
    plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pulse(df, time_buffer=50):
    """Finds a pulse using current spikes (~+32 to ~-32) and plots voltage over that pulse duration with extra buffer."""
    time = df["Total Time"].values
    voltage = df["Voltage"].values
    current = df["Current"].values  

    # Define a tolerance range for current detection
    tol = 2
    current_high = np.where((current >= 32 - tol) & (current <= 32 + tol))[0]
    current_low = np.where((current >= -32 - tol) & (current <= -32 + tol))[0]

    if len(current_high) == 0 or len(current_low) == 0:
        print("No pulse found. Check current values and tolerance.")
        return

    # Identify first +32 and next -32
    pulse_start = current_high[0]
    pulse_end_candidates = current_low[current_low > pulse_start]

    if len(pulse_end_candidates) == 0:
        print("No valid pulse end found. Check data sequence.")
        return

    pulse_end = pulse_end_candidates[0]

    # Convert buffer to indices
    t_start = max(0, time[pulse_start] - time_buffer)  # Ensure we don’t go below index 0
    t_end = min(time[-1], time[pulse_end] + time_buffer)  # Ensure we don’t exceed max time

    # Find indices closest to t_start and t_end
    start_idx = np.searchsorted(time, t_start)
    end_idx = np.searchsorted(time, t_end)

    # Extract pulse time and voltage with buffer
    time_pulse = time[start_idx:end_idx]
    voltage_pulse = voltage[start_idx:end_idx]

    if len(time_pulse) == 0 or len(voltage_pulse) == 0:
        print("Pulse extraction failed. Check index selection.")
        return

    # Plot the pulse with buffer
    plt.figure(figsize=(10, 5))
    plt.plot(time_pulse, voltage_pulse, label="Voltage Pulse", color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Single Voltage Pulse with Time Buffer")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Pulse detected from t={time_pulse[0]:.3f}s to t={time_pulse[-1]:.3f}s with buffer={time_buffer}s")

# -----------------------------------------------CALLING FUNCTIONS--------------------------------------------------------

#pt.plot_test("D","00")

#print(dfc[0].columns)  # Ensure "Current" column exists
#print(np.unique(dfd[0]["Current"]))  # Check unique current values
plot_pulse(dfc[0])  # Call function on first dataset