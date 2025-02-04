import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import plot as pt
import R0_identification_tianhe as rz
import Tianhe_csvPlot as ti
import data as dt

R0_matrix = []
SoC_matrix = []
SoH = []

def R0_SoC_C(test):
    df = dt.calculate_model_voltage_0("C",test)
    R0, time = rz.R0_record(df)
    #print(df)
    filtered_df = df[df['Total Time'].isin(time)]
    SoC = filtered_df['SoC'].tolist()
    return R0, SoC
    #print(R0[0:len(SoC)], SoC)
    #plt.plot(SoC, R0[0:len(SoC)])

'''
for i in range(24):
    test = str(i)
    if len(test) == 1:
        test = f'0{test}'
    R0, SoC = R0_SoC_C(test)
    SoH.append(dt.soh('C', test))
    R0_matrix.append(R0)
    SoC_matrix.append(SoC)

print(R0_matrix, SoC_matrix, SoH)

print(min(len(soc) for soc in SoC_matrix))
print(min(len(r0) for r0 in R0_matrix))


print(R0_matrix, SoC_matrix, SoH)
############################################################################

X = np.array(SoC_matrix)
Y = np.array(SoH)
Z = np.array(R0_matrix)

X_flat = X.flatten()
Y_flat = np.repeat(Y, 17)
Z_flat = Z.flatten()

def generate_poly_features(x, y, degree):
    """Generate polynomial terms up to `degree` for inputs x and y."""
    features = []
    for d in range(degree + 1):
        for i in range(d + 1):
            x_power = d - i
            y_power = i
            features.append((x ** x_power) * (y ** y_power))
    return np.column_stack(features)

degree = 7  # Example; adjust based on data complexity
A = generate_poly_features(X_flat, Y_flat, degree)  # Design matrix

coefficients, residuals, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)

np.save("coefficients.npy", coefficients)

'''

coefficients = np.load("coefficients.npy")

degree = 7 
#print(coefficients)


def f(x, y):
    global coefficients, degree
    """Evaluate f(x, y) using the polynomial coefficients.
    x => SoC  y => SoH
    """
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            x_power = d - i
            y_power = i
            terms.append((x ** x_power) * (y ** y_power))
    return np.dot(terms, coefficients)


if __name__ == '__main__':
    test = '06'
    R0, SoC = R0_SoC_C(test)
    SoH = dt.soh('C', test)
    
    plt.plot(SoC, R0, 'b')
    R0_pred = [f(SoC_num, SoH) for SoC_num in SoC]
    plt.plot(SoC, R0_pred, 'r')

    plt.show()

