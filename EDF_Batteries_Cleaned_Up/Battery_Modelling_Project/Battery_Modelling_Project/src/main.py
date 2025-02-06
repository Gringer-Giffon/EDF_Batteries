import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zeroth_order_modules as zom
import first_order_modules as fom
import os

if __name__ == '__main__':
    
    test = '00'
    cell = 'C'
    df = zom.dfc[int(test)]
    
    df['SoC'] = zom.soc(cell, test)
    SoH = zom.soh(cell, test)
    df['R0'] = [zom.f(SoC_num, SoH, cell=f'Cell {cell}') for SoC_num in df['SoC']]
    df['OCV'] = zom.calculate_ocv(df['SoC'], cell, test)   # Use the OCV function fitted with this specific test
    #df['OCV'] = [zom.OCV_f(SoC_num, SoH, cell=f'Cell {cell}') for SoC_num in df['SoC']]   # Use the pre-fitted general OCV function. 
    
    df['Model Voltage'] = [df["OCV"].iloc[i]+df["R0"].iloc[i] * df["Current"].iloc[i] for i in range(len(df))]
    #df = fom.SoC_SoH_to_Voltage(df, SoH, f'Cell {cell}')
    
    fom.plot_model_and_error(df, cell, order='Zero')
    '''
    
    cell = "C"
    f_vectorized = np.vectorize(fom.h_1)
    
    x = np.linspace(0, 0.8, 100)
    y = np.linspace(0.92, 1, 100)
    x, y = np.meshgrid(x, y)
    z = f_vectorized(x, y, cell=f'Cell {cell}')

    z = np.where(z < 100, 100, z)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use plot_surface instead of scatter
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('SoC Value')
    ax.set_ylabel('SoH Value')
    ax.set_zlabel('Resistance One (Ohm)')
    ax.set_title('SoC and SoH Effects on C1 of cell ' + cell)

    ax.view_init(elev=20, azim=60)

    plt.show()
    '''
