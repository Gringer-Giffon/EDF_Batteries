import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import plot as pt
import data as dt
import pandas as pd
import os
import Tianhe_csvPlot as ti
import math


# Plot entire voltage
# plt.plot(dt.extract("D", "01")["Total Time"],dt.extract("D", "01")["Voltage"], "x")
# plt.show()

def extract_pulses(cell, test):
    df_pre = dt.extract(cell, test)
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


def measure_OCV(pulse_df):
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


def measure_R0(pulse_df):
    spikes = []
    for pulse in pulse_df:
        voltage_diff = np.diff(pulse["Voltage"].values)

        # threshold for detection
        threshold = 0.05

        # index of spike
        spikes.append(np.argmax(np.abs(voltage_diff) > threshold))
    R0 = []
    for i in range(len(pulse_df)):
        pulse = pulse_df[i]
        spike_in = spikes[i]
        R0_val = abs(pulse["Voltage"].iloc[spike_in] -
                     pulse["Voltage"].iloc[spike_in+1])/abs(pulse["Current"].iloc[spike_in+1])
        if R0_val != math.inf:
            R0.append(abs(pulse["Voltage"].iloc[spike_in] -
                      pulse["Voltage"].iloc[spike_in+1])/abs(pulse["Current"].iloc[spike_in+1]))
        else:
            R0.append(0)
    return R0

def measure_tau(pulse_df):
    spikes = []
    for pulse in pulse_df:
        voltage_diff = np.diff(pulse["Voltage"].values)

        # threshold for detection
        threshold = 0.05

        # index of spike
        spikes.append(np.argmax(np.abs(voltage_diff) > threshold))
    tau = []

    for i in range(len(pulse_df)):
        pulse = pulse_df[i]
        spike_in = spikes[i]
        if pulse["Current"].iloc[spike_in] >0:
            factor = 0.37
            volt_1 = pulse["Voltage"][pulse["Voltage"] == min["Voltage"]]
            target_voltage = pulse["Voltage"].iloc[spike_in+1] - factor * \
        abs(max(pulse["Voltage"]-min(pulse["Voltage"])))
        else:
            factor = 0.37
        target_voltage = pulse["Voltage"].iloc[spike_in+1] - factor * \
        abs(pulse["Voltage"].iloc[spike_in+1]-min(pulse["Voltage"]))
            
        idx = (pulse["Voltage"] - target_voltage).abs().idxmin()

        tau.append(pulse["Total Time"].loc[idx] - pulse["Total Time"].iloc[0])
            
    return tau

def extract_pulse(cell, test):
    '''
    Parameters: cell (string), test(string) test number "01","09",10", etc..

    Returns the dataframe of a spike in the middle of the pulse region of selected data
    '''
    df_pre = dt.extract(cell, test)
    df = df_pre[abs(df_pre["Current"]) == max(
        df_pre["Current"])]  # Selects all pulse indexes
    # Selects an index in the middle of the data
    pulse_time = df["Total Time"].iloc[math.floor(len(df)/2)]
    # Selection of frame of impulse
    if cell == "C":
        pulse_df = df_pre[(df_pre["Total Time"] >= pulse_time - 20)
                          & (df_pre["Total Time"] <= pulse_time + 15)]
    elif cell == "D":
        pulse_df = df_pre[(df_pre["Total Time"] >= pulse_time - 200)
                          & (df_pre["Total Time"] <= pulse_time + 150)]
    else:
        print("Invalid choice of cell")
        return None
    return pulse_df


def spike_index(pulse):
    '''
    Parameters: pulse (dataframe) data of your selected pulse

    Returns the index of the pulse spike
    '''
    voltage_diff = np.diff(pulse["Voltage"].values)

    # threshold for detection
    threshold = 0.05

    # index of spike
    spike_index = np.argmax(np.abs(voltage_diff) > threshold)

    return spike_index


def spike_indexes(df):
    voltage_diff = np.diff(df["Voltage"].values)

    # threshold for detection
    threshold = 0.03

    # index of spike
    spike_indexes = np.where(np.abs(voltage_diff) > threshold)[0].tolist()
    # spike_indexes = np.where(abs(df["Current"]) == 32)[0].tolist()
    return spike_indexes


def calculate_distance(voltages):
    error = 0
    errors = []
    for i in range(len(voltages)):
        error += np.sqrt(voltages[i]**2 - df["Voltage"].iloc[i]**2)
        errors.append(error)
    return errors


def model_pulse():
    # Extract one pulse
    cell = input("which cell would you like to analyse: ")
    test = input("which test would you like to analyse: ")
    df = extract_pulse(cell, test)

    # df = df[(df["Total Time"] >= 56500) & (df["Total Time"] <= 56620)]

    plt.plot(df["Total Time"], df["Voltage"], "x")
    plt.show()

    # Oth order
    spike = spike_index(df)
    U_ocv = df["Voltage"].iloc[spike]

    R0 = abs(U_ocv - df["Voltage"].iloc[spike+1]) / \
        abs(df["Current"].iloc[spike+1])  # !!!!!!!
    print("R0", R0)

    model_voltage_0 = [
        U_ocv + R0 * abs(df["Current"].iloc[i]) if df["Current"].iloc[i] > 0
        else U_ocv - R0 * abs(df["Current"].iloc[i])
        for i in range(len(df))
    ]

    print(model_voltage_0)

    # plt.plot(df["Total Time"],model_voltage_0,'b')
    plt.plot(df["Total Time"], df["Voltage"], 'r')

    # 1st order non fitted
    # R0 = 0.5*(R0+abs(min(df["Voltage"])-max(df["Voltage"]))/abs(df["Current"].iloc[spike+1]))
    # model_voltage_0 = [
    # U_ocv + R0 * abs(df["Current"].iloc[i]) if df["Current"].iloc[i] > 0
    # else U_ocv - R0 * abs(df["Current"].iloc[i])
    # for i in range(len(df))
# ]
    R1 = abs(df["Voltage"].iloc[spike+1] - min(df["Voltage"])) / \
        abs(df["Current"][df["Voltage"] == min(df["Voltage"])]).iloc[0]
    print("R1", R1)
    target_voltage = df["Voltage"].iloc[spike+1] - 0.63 * \
        abs(df["Voltage"].iloc[spike+1]-min(df["Voltage"]))

    idx = (df["Voltage"] - target_voltage).abs().idxmin()

    tau = (df["Total Time"].loc[idx] - df["Total Time"].iloc[0])

    print(tau)

    model_voltage_1 = [(model_voltage_0[i] + (R1 * abs(df["Current"].iloc[i])*(1-np.exp(-(
        df["Total Time"].iloc[i]-abs(df["Total Time"].iloc[0]))/tau)))) if df["Current"].iloc[i] > 0 else (model_voltage_0[i] - (R1 * abs(df["Current"].iloc[i])*(1-np.exp(-(
            df["Total Time"].iloc[i]-abs(df["Total Time"].iloc[0]))/tau)))) for i in range(len(df))]

    print("real voltage", df["Voltage"], "model voltage 1",
          model_voltage_1, "R0", R0, "R1", R1, "tau", tau, "Uocv", U_ocv)
    plt.plot(df["Total Time"], model_voltage_1, "b")
    plt.title("Model and data voltage comparison over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend(["Data", "Model"])
    plt.show()
    # plt.show()


def model_pulses():
    cell = input("which cell would you like to analyse: ")
    test = input("which test would you like to analyse: ")
    df = dt.extract(cell, test)
    # print(df)
    df = df[(df["Total Time"] >= 1700) & (df["Total Time"] <= 24500)]
    spike_indices = spike_indexes(df)

    for spike in spike_indices:
        if cell == "C":
            pulse_df = df[(df["Total Time"] >= df["Total Time"].iloc[spike] - 20)
                          & (df["Total Time"] <= df["Total Time"].iloc[spike] + 15)]
        elif cell == "D":
            pulse_df = df[(df["Total Time"] >= df["Total Time"].iloc[spike] - 200)
                          & (df["Total Time"] <= df["Total Time"].iloc[spike] + 150)]
        else:
            print("Invalid choice of cell")
        # Oth order
        df = pulse_df
        print(df)
        print(spike)
        print(df["Voltage"])
        U_ocv = df["Voltage"].iloc[spike]

        R0 = abs(U_ocv - df["Voltage"].iloc[spike+1]) / \
            abs(df["Current"].iloc[spike+1])
        # print("R0", R0)

        model_voltage_0 = [
            U_ocv + R0 * abs(df["Current"].iloc[i]) if df["Current"].iloc[i] > 0
            else U_ocv - R0 * abs(df["Current"].iloc[i])
            for i in range(len(df))
        ]

        # print(model_voltage_0)

        # plt.plot(df["Total Time"],model_voltage_0,'b')
        # plt.plot(df["Total Time"], df["Voltage"], 'r')

        # 1st order non fitted
        R1 = abs(df["Voltage"].iloc[spike+1] - min(df["Voltage"])) / \
            abs(df["Current"][df["Voltage"] == min(df["Voltage"])]).iloc[0]
        # print("R1", R1)
        target_voltage = df["Voltage"].iloc[spike+1] - 0.63 * \
            abs(df["Voltage"].iloc[spike+1]-min(df["Voltage"]))

        idx = (df["Voltage"] - target_voltage).abs().idxmin()

        tau = (df["Total Time"].loc[idx] - df["Total Time"].iloc[0])

        # print(tau)

        model_voltage_1 = [(model_voltage_0[i] + (R1 * abs(df["Current"].iloc[i])*(1-np.exp(-(
            df["Total Time"].iloc[i]-abs(df["Total Time"].iloc[0]))/tau)))) if df["Current"].iloc[i] > 0 else (model_voltage_0[i] - (R1 * abs(df["Current"].iloc[i])*(1-np.exp(-(
                df["Total Time"].iloc[i]-abs(df["Total Time"].iloc[0]))/tau)))) for i in range(len(df))]

        plt.plot(df["Total Time"], model_voltage_1, "b")
    plt.show()


if __name__ == "__main__":
    # model_pulse()
    pulses = extract_pulses("C", "01")

    print(measure_tau(pulses))

    # model_pulses()

    # print(extract_pulses(cell,test))

    """
        # 1st order fitted

        from scipy.optimize import curve_fit

        def battery_model(t, R0, R1, tau, U_ocv):
            current = np.abs(df["Current"].values)  # Ensure the current is positive
            return U_ocv + R0 * current + R1 * current * (1 - np.exp(-t / tau))

        print([R0, R1, tau, U_ocv])

        popt, _ = curve_fit(battery_model, df["Total Time"]-df["Total Time"].iloc[0], df["Voltage"], p0=[R0, R1, tau, U_ocv])

        R0_fit, R1_fit, tau_fit, U_ocv_fit = popt

        model_voltage_fit = battery_model(df["Total Time"]-df["Total Time"].iloc[0], *popt)
        print(popt)
        print(model_voltage_fit)
        #plt.plot(df["Total Time"], df["Voltage"], 'r', label='Data')
        #plt.plot(df["Total Time"], model_voltage_fit, 'b', label='Fitted Model')
        #plt.plot(df['Total Time'],model_voltage_0,'g')
        #plt.legend()
        #plt.show()

        # Error estimation


    plt.plot(df["Total Time"],calculate_distance(model_voltage_0),'r')
    plt.plot(df["Total Time"], calculate_distance(model_voltage_1),'g')
    plt.show()
    """
