import cell_plotting_tests_etienne as et
import matplotlib.pyplot as plt

cell = "C"
test = "08"


df = et.extract_pulse(cell,test)
spike = et.spike_index(df)


R1 = abs(df["Voltage"].iloc[spike+1] - min(df["Voltage"])) / \
    abs(df["Current"][df["Voltage"] == min(df["Voltage"])]).iloc[0]
print("R1", R1)
target_voltage = df["Voltage"].iloc[spike+1] - 0.63 * \
    abs(df["Voltage"].iloc[spike+1]-min(df["Voltage"]))

idx = (df["Voltage"] - target_voltage).abs().idxmin()

tau = (df["Total Time"].loc[idx] - df["Total Time"].iloc[0])

time = df["Total Time"]
voltage = df["Voltage"]

"""plt.plot(time, voltage, "r+")
plt.show()"""

def find_r0_model_1(cell, test):
    df = et.extract_pulse(cell,test)
    spike = et.spike_index(df)
    u_a = df["Voltage"].iloc[spike]
    u_b = df["Voltage"].iloc[spike + 1]
    u_c = min(df["Voltage"])
    u_d = max(df["Voltage"])
    i_l1 = df["Current"].iloc[spike + 1]
    i_l2 = df["Current"][df["Voltage"] == min(df["Voltage"])].iloc[0]        
    return (1 / 2) * ( (u_a - u_b) / i_l1 + (u_c - u_d) / i_l2 )

print("R0 ", find_r0_model_1(cell, test))

