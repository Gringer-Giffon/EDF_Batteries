import pandas as pd
import numpy as np
import zeroth_order_modules as zom
import matplotlib.pyplot as plt
import os

R1_coeffs = [ 0.00909442, -0.0215907, 0.01814499, -0.00607696, 0.00086555]
C1_coeffs = [-159168.96389611, 380939.23817068, -288456.46579797, 48044.82387891, 20063.47043403, 2047.16208581]

# ----------------------- Ready to Use R1 and C1 Function for Cell C Test 0 -----------------------------------

g = np.poly1d(R1_coeffs)
h = np.poly1d(C1_coeffs)


# ---------------------- First Order Voltage Model Example, for Cell C Test 0 -----------------------------

def SoC_to_Voltage(df):
    '''
    Compute the modeled voltage based on the SoC data (State of Charge)
    for a specified cell type using a polynomial model.

    Parameters:
    - df (pandas.DataFrame): The dataframe that contains the value of SoC (State of Charge) of the cell(s).

    Returns:
    - pandas.DataFrame: The input DataFrame with an additional column 'Model Voltage' containing the modeled voltage values.
    '''
    SoH = 1
    R1 = g(df['SoC'])
    C1 = h(df['SoC'])

    t = df['Total Time'][0]

    result_list = []
    for i in range(len(df)):
        if i < len(df)-1 and abs(df['Current'][i] + df['Current'][i+1]) > 10:
            t = 0
            #print('reset to 0')
        
        V = zom.OCV_f(df['SoC'][i], SoH) + df['Current'][i]*zom.f(df['SoC'][i], SoH, first_order=True) + df['Current'][i]*R1[i]*(1 - np.exp(-t/tau))
        result_list.append(V)

        if i < len(df)-1:
            t += (df['Total Time'][i+1] - df['Total Time'][i])
    df['Model Voltage'] = result_list

    return df


# ------------------------ First Order General Model -----------------------------

def h_1(x, y, cell='Cell C'):
    if cell == 'Cell C':
        coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/first_order_models/coefficients_C1_1.npy'))
        degree = 7
    elif cell == 'Cell D':
        coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/first_order_models/coefficients_C1_1_cell_d.npy'))
        degree = 7
    else:
        print(zom.cat, '\nError: Unknown Cell')
        return
    
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            x_power = d - i
            y_power = i
            terms.append((x ** x_power) * (y ** y_power))
    return np.dot(terms, coefficients)


def g_1(x, y, cell='Cell C'):
    if cell == 'Cell C':
        coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/first_order_models/coefficients_R1_1_cell_c_r.npy'))
        degree = 7
    elif cell == 'Cell D':
        coefficients = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/first_order_models/coefficients_C1_1_cell_c_r.npy'))
        degree = 7
    else:
        print(zom.cat, '\nError: Unknown Cell')
        return
    
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            x_power = d - i
            y_power = i
            terms.append((x ** x_power) * (y ** y_power))
    return np.dot(terms, coefficients)


def SoC_SoH_to_Voltage(df, SoH, cell='Cell C'):
    '''
    Compute the modeled voltage based on the SoC data (State of Charge)
    for a specified cell type using a polynomial model.

    Parameters:
    - df (pandas.DataFrame): The dataframe that contains the value of SoC (State of Charge) of the cell(s).
    - SoH (float): The number that contains the SoH (State of Health) status of the cell during this cycle.
    - cell (str, optional): The type of the cell ('Cell C' or 'Cell D'). (Defaults to 'Cell C')

    Returns:
    - pandas.DataFrame: The input DataFrame with an additional column 'Model Voltage' containing the modeled voltage values.
    '''
    t = df['Total Time'][0]

    result_list = []
    for i in range(len(df)):
        if i < len(df)-1 and abs(df['Current'][i] - df['Current'][i+1]) > 10:
            t = 0
            #print('reset to 0')

        R1 = abs(g_1(df['SoC'][i], SoH, cell))
        C1 = h_1(df['SoC'][i], SoH, cell)

        tau = abs(R1*C1)

        V = df['OCV'][i] + df['Current'][i]*zom.f(df['SoC'][i], SoH, cell, first_order=True) + df['Current'][i]*R1*(1 - np.exp(-t/tau))
        result_list.append(V)

        if i < len(df)-1:
            t += (df['Total Time'][i+1] - df['Total Time'][i])
    df['Model Voltage'] = result_list
    return df


# -------------------------------- Plot and Demonstration -----------------------------

def plot_model_and_error(df, cell='C', order='First'):
    if 'Voltage' not in df.columns or 'Model Voltage' not in df.columns:
        print(zom.cat, '\nColumn not found')
        return
    else:
        df = zom.pulses_extract(df)    # Extract the region where pulses locates
        df = zom.cost(df)

        print('The mean abs error is: ', zom.mean_abs_cost(df))
        

        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.suptitle(f'{order} Order Model of Cell {cell}', fontsize=16, fontweight='bold')
        ax1.set_title('Modeled Voltage')
        ax1.set_ylabel("Voltage (V)")
        ax1.set_xlabel('Time (s)')
        ax1.scatter(df['Total Time'], df['Voltage'], label = 'Raw Data', color = '#8C7194', s=1)    
        ax1.plot(df['Total Time'], df['Model Voltage'], label = 'Modeled Data', color = '#9A163A')
        ax1.legend()
        
        ax2.set_title('Error')
        ax2.set_ylabel("Voltage (V)")
        ax2.set_xlabel('Time (s)')
        ax2.plot(df['Total Time'], df['Error'], label = 'Error')
        #ax2.text(0.03, 100, f'Average Abs Error {avg_error}')
        #ax2.legend()

        plt.subplots_adjust(hspace=0.3)  # Adjust the vertical spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.show()


