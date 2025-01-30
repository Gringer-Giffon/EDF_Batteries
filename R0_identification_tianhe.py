import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import Tianhe_csvPlot as csvp

myList=[]

folderPath = f'./cells_data'

csvFiles = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csvFiles if '_C_' in f]
csvFiles_D = [f for f in csvFiles if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]

def calc_R0_cell_C(A1, B1, C1, D1, A2, B2, C2, D2, I):
    return 0.25*(A1-B1 + D1-C1 + B2-A2 + C2-D2)/I

if __name__ == '__main__':
    '''
    start = 41550
    end = 44250
    cycle=0
    number=0

    A,C = csvp.locate(dfd[cycle], 10, number, offset=1)
    B,x = csvp.locate(dfd[cycle], 10, number, offset=-30)
    D,E = csvp.locate(dfd[cycle], 5, number+1)
    
    plt.subplot(3,1,1)
    plt.plot(csvp.extract(dfd[0], start, end)['Total Time'], csvp.extract(dfd[0], start, end)['Voltage'])

    plt.plot([C, D], [dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].values[0],
                      dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].values[0]])    
    plt.plot([B, C], [dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].values[0],
                      dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].values[0]])
    plt.plot([A, B], [dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].values[0],
                      dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].values[0]])
    
    plt.subplot(3,1,2)
    plt.plot(csvp.extract(dfd[cycle], start, end)['Total Time'], csvp.extract(dfd[cycle], start, end)['Step'])
    plt.subplot(3,1,3)
    plt.plot(csvp.extract(dfd[cycle], start, end)['Total Time'], csvp.extract(dfd[cycle], start, end)['Current'])
    plt.show()
    '''
    R0 = []
    #A1, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD(dfc, 20, 17)
    pos = 0
    impulse_num = 11

    for i in range(len(dfc)):
        mask = dfc[i]['Step'] == 7
        dfc[i]['start'] = mask& ~mask.shift(1, fill_value=False)
        start_indices = dfc[i].index[dfc[i]['start']].tolist()        
        for j in range(len(start_indices)-1):
            A1, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD_n(dfc, i, j+1)
            R = calc_R0_cell_C(A1[1], B1[1], C1[1], D1[1], A2[1], B2[1], C2[1], D2[1], 32)
            R0.append(R)
            if j == impulse_num:
                plt.plot(pos, R, 'ro')
            pos+=1
    plt.plot(R0)
    plt.text(-10,0.00163,'resistance decreases at the beginning of each cycle,\nas the temperature increases')
    plt.text(-10,0.00295,'resistance increases at the end of charge-discharge stage, \ndue to the polarization of the battery')
    plt.text(200,0.00175,'overall, a logarithmic increase was witnessed as battery ages')
    plt.ylim(0.0016, 0.00305)
    plt.show()
    '''
    R0 = []
    for i in range(len(start_indices)-1):
        A1, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD_n(dfc, 0, i+1)
        R0.append(calc_R0_cell_C(A1[1], B1[1], C1[1], D1[1], A2[1], B2[1], C2[1], D2[1], 32))
    plt.plot(R0)
    plt.show()
    '''
