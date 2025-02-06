import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import cumulative_trapezoid
import plot as pt
import R0_identification_tianhe as rz
import Tianhe_csvPlot as ti
import data as dt

folderPath = f'./cells_data'

csvFiles = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csvFiles if '_C_' in f]
csvFiles_D = [f for f in csvFiles if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file))
       for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]

SoC_matrix = []
OCV_matrix = []
SoH_list = []
'''
for i in range(24):
    test = str(i)
    if len(test) == 1:
        test = f'0{test}'

    df = dt.soc_ocv('C', test)
    df = df.sort_values(by='SoC').iloc[:21]
    SoH = dt.soh('C', test)
    
    SoC_matrix.append(df['SoC'])
    OCV_matrix.append(df['OCV'])
    SoH_list.append(SoH)

print(min(len(soc) for soc in SoC_matrix))
    
print(df)

##############################################################################

X = np.array(SoC_matrix)
Y = np.array(SoH_list)
Z = np.array(OCV_matrix)

X_flat = X.flatten()
Y_flat = np.repeat(Y, 21)
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

np.save("coefficients_OCV.npy", coefficients)
'''
coefficients = np.load("coefficients_OCV.npy")
degree = 7


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
    test = '07'
    df = dt.soc_ocv('C', test)
    df = df.sort_values(by='SoC')
    SoH = dt.soh('C', test)
    
    OCV = [f(SoC, SoH) for SoC in df['SoC']]
    df['OCV_Pred'] = OCV
    plt.plot(df['SoC'], df['OCV_Pred'], 'b')
    plt.plot(df['SoC'], df['OCV'], 'r')
    plt.show()
    
