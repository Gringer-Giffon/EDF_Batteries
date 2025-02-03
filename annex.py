import data as dt
import plot as pt
from scipy.integrate import cumulative_trapezoid

def ocv_voltage():
    soc_d_t_list = soc_d_time("01")
    print(soc_d_t_list)
    soc_ocv_v = {0.05*i for i in range(0, 21)}
    for element in soc_ocv_v:
        for pair in soc_d_t_list:
            if element == pair[0]:
                soc_ocv_v[element] = dt.extract("D", "01")["Voltage"][dt.extract("D", "01")[
                    "Total Time"] == pair[1]]
    print(soc_ocv_v)
    return None


def soc_d(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of charge of D cell at a given time for the discharge at the end of the test
    Returns list of SOC values in the full discharge phase for cell D
    '''

    data = dt.extract_step(21, 23, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    Q_available = [Q_remaining - I*(data["Total Time"].iloc[i] -
                                    data["Total Time"].iloc[0])/3600 for i in range(len(data["Total Time"]))]

    SOC = [Q_available[i]/Q_remaining for i in range(len(data["Total Time"]))]

    return SOC

def soc_d_time(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Calculates the state of charge of D cell at a given time for the discharge at the end of the test
    Returns list of SOC values
    '''

    data = dt.extract_step(21, 24, "D", test)

    # Calculate I and t
    I = abs(data["Current"].mean())
    t = data["Total Time"].iloc[-1]-data["Total Time"].iloc[0]

    # Calculate Q remaining and Q available
    Q_remaining = I*t/3600

    Q_available = [(Q_remaining - I*(data["Total Time"].iloc[i] -
                                     data["Total Time"].iloc[0])/3600, data["Total Time"].iloc[i]) for i in range(len(data["Total Time"]))]

    SOC = [(Q_available[i][0]/Q_remaining, data["Total Time"].iloc[i])
           for i in range(len(data["Total Time"]))]

    """
    soc_voltage_dict = {
        data["Voltage"].iloc[i]: Q_available[i] / Q_remaining
        for i in range(len(data["Total Time"]))}
    """
    return SOC

def soc_full_c(test):
    '''
    Parameters: test (string) in the form 00, 01, etc..

    Returns the list of SOC values for cell C
    '''
    data_full = dt.extract("C", test)
    data = dt.extract_step(26, 27, "C", test)

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

def extract_all_steps(first, second, cell, test):
    '''
    Parameters: first (int) first step, second (int) second step, cell (string) C or D, test (string) in the form 00, 01, etc..

    Returns dataframe for given step interval, does not remove duplicate step sequences
    '''

    data = dt.extract(cell, test)
    step_data = data[data["Step"].isin(list(range(first, second+1)))]
    return step_data

def calc_r0(test):
    r0 = [abs(add_ocv(test)["OCV"].iloc[i]-add_ocv(test)["Voltage"].iloc[i])
          for i in range(len(add_ocv(test)))]
    return r0


def calculate_r1(cell,test):
    '''
    Parameters: test (int) test number

    Returns dataframe with original data and SoC and R0

    '''
    time_between_dupes = 300 #added this
    df = extract(cell, test)
    df["SoC"] = soc(cell, test)
    df["OCV"] = calculate_ocv(soc(cell, test), cell, test)
    
    R1 = [(abs(df["OCV"].iloc[i] - df["Voltage"].iloc[i]) / abs(df["Current"].iloc[i]) if abs(df["Current"].iloc[i]) == 30 else 0)
          for i in range(len(df["Current"]))]

    #rz.R0_fill(dfc)
    # print(df, '\n')
    # R0 = dfc[int(test)]["R0"]  # complete R0 column for given test

    # df["SoC"] = soc("C", test)
    df["R1"] = R1
    R1_no_dupes = df.loc[~(
        df["Total Time"].diff().abs() < time_between_dupes)] #added this
    
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


def plot_soc_tau_r1(cell,test):
    df = pd.DataFrame(data = {"tau": measure_r1(cell,test)[1], "R1": measure_r1(cell,test)[2]})
    df["tau"] = measure_r1(cell,test)[1]
    df["R1"] = measure_r1(cell,test)[2]



    """
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
        '''
        # Calculate tau (time to reach 63% of the voltage change) voltage
        voltage_target = voltage_start - 0.63 * abs(voltage_end - voltage_start)

        # Find the closest index where the voltage meets or exceeds the target
        time_to_adapt = None
        for idx in range(start_index, end_index + 1):
            if df["Voltage"].iloc[idx] <= voltage_target:
                time_to_adapt = df["Total Time"].iloc[idx] - df["Total Time"].iloc[start_index]
                break

        tau_value = time_to_adapt if time_to_adapt is not None else 0
        '''


        # Assign tau and R1 only at the start index
        #df1.loc[df1.index[start_index], "tau"] = tau_value
        #df1.loc[df1.index[start_index], "R1"] = R1_value

        df.loc[df.index.isin(range(start_index,end_index)),"tau"] = tau_value
        df.loc[df.index.isin(range(start_index,end_index)),"R1"] = R1_value

        # Store values for debugging or further processing
        tau_values.append(tau_value)
        r1_values.append(R1_value)

        
        

    print("df!!!!!!",df)
    return None

    
    df1 = calculate_model_voltage_0(cell,test)
    if cell == "C":
        df = df1[df1['Step'] == 7].copy()  # Ensure copy to modify safely
    elif cell == "D":
        df = df1[df1['Step'] ==6].copy()
    else:
        print("Invalid cell input")
        return None
    pulse_coords = []
    pulse_start = 0
    tau_values = []
    r1_values = []

    # Initialize columns for tau and R1
    df["tau"] = None
    df["R1"] = None

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
        '''
        # Calculate tau (time to reach 63% of the voltage change) voltage
        voltage_target = voltage_start - 0.63 * abs(voltage_end - voltage_start)

        # Find the closest index where the voltage meets or exceeds the target
        time_to_adapt = None
        for idx in range(start_index, end_index + 1):
            if df["Voltage"].iloc[idx] <= voltage_target:
                time_to_adapt = df["Total Time"].iloc[idx] - df["Total Time"].iloc[start_index]
                break

        tau_value = time_to_adapt if time_to_adapt is not None else 0
        '''


        # Assign tau and R1 only at the start index
        #df1.loc[df1.index[start_index], "tau"] = tau_value
        #df1.loc[df1.index[start_index], "R1"] = R1_value

        df.loc[df.index.isin(range(start_index,end_index)),"tau"] = tau_value
        df.loc[df.index.isin(range(start_index,end_index)),"R1"] = R1_value

        # Store values for debugging or further processing
        tau_values.append(tau_value)
        r1_values.append(R1_value)
    df1.to_csv("r1 data")
    print("df1",df)
    return df
    """