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