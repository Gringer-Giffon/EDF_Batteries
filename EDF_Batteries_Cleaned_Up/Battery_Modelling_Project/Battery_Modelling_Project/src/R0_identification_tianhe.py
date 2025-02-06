import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#import Tianhe_csvPlot as csvp
import data as dt

myList = []

data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/cells_data') # Reaching the datas in the data folder
folderPath = data_file_path

csvFiles = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csvFiles if '_C_' in f]
csvFiles_D = [f for f in csvFiles if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file))
       for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]

region = []


def calc_R0_cell_C(A1, B1, C1, D1, A2, B2, C2, D2, I):
    return 0.25*(A1-B1 + D1-C1 + B2-A2 + C2-D2)/I
    # print(A1, B1, C1, D1, I)
    # return ((abs(A1-B1))/abs(I), (abs(C1-D1))/abs(I))


def R0_calc_all(R0):
    start_indices = []
    pos = 0
    for i in range(len(dfc)):
        mask = dfc[i]['Step'] == 7
        dfc[i]['start'] = mask & ~mask.shift(1, fill_value=False)
        start_indices = dfc[i].index[dfc[i]['start']].tolist()
        for j in range(len(start_indices)-1):
            A1, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD_n(dfc, i, j+1)
            R = calc_R0_cell_C(A1[1], B1[1], C1[1], D1[1],
                               A2[1], B2[1], C2[1], D2[1], 32)  # current is constant at 32..
            R0.append(R)
            pos += 1  # need to set a time limit on recording resistance
    return start_indices


def R0_fill(dfc):
    R0 = []
    start_indices = R0_calc_all(R0)
    j = 0
    startregion = []

    for df in dfc:
        # start_indices = df.index[df['start']].tolist()
        for i in range(len(start_indices)-1):
            t_s, x = csvp.locate(df, 15, i)
            x, t_e = csvp.locate(df, 15, i+1)
            start = df.index[df['Total Time'] == t_s][0]
            end = df.index[df['Total Time'] == t_e][0]

            if i == 0:
                df.loc[0:end, 'R0'] = R0[j]
            elif i == len(start_indices)-2:
                df.loc[start:len(df)-1, 'R0'] = R0[j]
            else:
                df.loc[start:end, 'R0'] = R0[j]

            # df.loc[start:end, 'R0'] = R0[j]
            j += 1

            if i == 0:
                startregion.append(start)
            elif i == len(start_indices)-2:
                startregion.append(end)
        region.append(startregion)
        startregion = []
        # plt.plot(df['Total Time'], df['R0'])
        # plt.show()


def R0_replace(df):
    R0 = []
    start_indices = R0_calc_all(R0)
    j = 0
    startregion = []

    for i in range(len(start_indices)-1):
        t_s, x = csvp.locate(df, 15, i)
        x, t_e = csvp.locate(df, 15, i+1)
        start = df.index[df['Total Time'] == t_s][0]
        end = df.index[df['Total Time'] == t_e][0]

        df.loc[start:end, 'R0'] = R0[j]

        j += 1

        if i == 0:
            startregion.append(start)
        elif i == len(start_indices)-2:
            startregion.append(end)


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
    A1, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD(dfc, 20, 1)
    pos = 0
    impulse_num = 1

    for i in range(len(dfc)):
        mask = dfc[i]['Step'] == 7
        dfc[i]['start'] = mask & ~mask.shift(1, fill_value=False)
        start_indices = dfc[i].index[dfc[i]['start']].tolist()
        for j in range(len(start_indices)-1):
            A1, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD_n(dfc, i, j+1)
            R = calc_R0_cell_C(A1[1], B1[1], C1[1], D1[1],
                               A2[1], B2[1], C2[1], D2[1], 32)
            R0.append(R)
            if j == impulse_num:
                plt.plot(pos, R, 'ro')
            pos += 1

    plt.plot(R0)
    plt.text(-10, 0.00163, 'resistance decreases at the beginning of each cycle,\nas the temperature increases')
    plt.text(-10, 0.00295, 'resistance increases at the end of charge-discharge stage, \ndue to the polarization of the battery')
    plt.text(200, 0.00175,
             'overall, a logarithmic increase was witnessed as battery ages')
    plt.ylim(0.0016, 0.00305)
    plt.show()


############################### Calc R0 with Respect to Time ####################################
    R0_fill(dfc)
############################### Plot R0 with Respect to Time ####################################
    plt.plot(dfc[0]['Total Time'], dfc[0]['R0'])
    plt.show()

    i = 0
    for df in dfc:
        df['OCV'] = None
        i_s = region[i][0]
        i_e = region[i][1]
        df.loc[i_s:i_e, 'OCV'] = df['Voltage'][i_s:i_e] + \
            (df['R0'][i_s:i_e]*df['Current'][i_s:i_e])
        i += 1

    '''
    R0 = []
    for i in range(len(start_indices)-1):
        A1, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD_n(dfc, 0, i+1)
        R0.append(calc_R0_cell_C(A1[1], B1[1], C1[1], D1[1], A2[1], B2[1], C2[1], D2[1], 32))
    plt.plot(R0)
    plt.show()
    '''

    cycle = 0

    t_s, B1, C1, D1, A2, B2, C2, D2 = csvp.c_locate_ABCD_n(dfc, cycle, 1)
    A1, B1, C1, D1, A2, B2, C2, t_e = csvp.c_locate_ABCD_n(
        dfc, cycle, len(start_indices)-1)

    plt.subplot(2, 1, 1)
    plt.plot(csvp.extract(dfc[cycle], t_s[0], t_e[0])[
             'Total Time'], csvp.extract(dfc[cycle], t_s[0], t_e[0])['OCV'])

    plt.subplot(2, 1, 2)
    plt.plot(csvp.extract(dfc[cycle], t_s[0], t_e[0])[
             'Total Time'], csvp.extract(dfc[cycle], t_s[0], t_e[0])['Voltage'])
    plt.show()
