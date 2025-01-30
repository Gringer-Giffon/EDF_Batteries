import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

myList=[]

folderPath = f'./cells_data'

csvFiles = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

csvFiles_C = [f for f in csvFiles if '_C_' in f]
csvFiles_D = [f for f in csvFiles if '_D_' in f]

dfc = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_C]      # Dataframes for Cell C
dfd = [pd.read_csv(os.path.join(folderPath, file)) for file in csvFiles_D]      # Dataframes for Cell D

def extract(df, start, end):        # Extract data with respect to time
    return df[(df['Total Time']>=start) & (df['Total Time']<=end)]

def plotAll(dfs, name, start=None, end=None):       # Plot everything with repsect to time
    if end == None or start == None:
        start = 0
        end = dfs[0]['Total Time'].iloc[-1]
        
    for df in dfs:
        df = extract(df, start, end)
        plt.plot(df['Total Time'], df[name])

def locate(df, step, pattern, offset=0, offset_2=0):        # Extract the df with respect to steps
    mask = df['Step'] == step
    df['start'] = mask& ~mask.shift(1, fill_value=False)
    df['end'] = mask& ~mask.shift(-1, fill_value=False)
    start_indices = df.index[df['start']].tolist()
    end_indices = df.index[df['end']].tolist()
    return df['Total Time'][start_indices[pattern]-offset], df['Total Time'][end_indices[pattern]+offset_2]

def d_locate_ABCD(dfd, cycle, number):        # locate points ABCD while plot them on a graph, not sure if it works on cellc
    '''
    dfd represents the list of dataframes you're using
    cycle represents the specific cycle you're using
    number represents the impulse you're locating
    '''
    
    start, end1 = locate(dfd[cycle], 6, number, offset=1)
    start1, end = locate(dfd[cycle], 7,number, offset_2=-5)

    A,C = locate(dfd[cycle], 6, number, offset=1)
    B,x = locate(dfd[cycle], 6, number)
    D,E = locate(dfd[cycle], 7, number)
    
    plt.subplot(2,2,1)
    #plotAll(dfd, 'Voltage', start, 42000)
    plt.plot(extract(dfd[cycle], start, end)['Total Time'], extract(dfd[cycle], start, end)['Voltage'])
    plt.plot([C, D], [dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].values[0],
                      dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].values[0]])    
    plt.plot([B, C], [dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].values[0],
                      dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].values[0]])
    plt.plot([A, B], [dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].values[0],
                      dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].values[0]])

    
    plt.text(A, dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].values[0], 'A')
    plt.text(B, dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].values[0], 'B')
    plt.text(C, dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].values[0], 'C')
    plt.text(D, dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].values[0], 'D')
    
    plt.subplot(2,2,2)
    plt.plot(extract(dfd[cycle], start, end)['Total Time'], extract(dfd[cycle], start, end)['Current'])
    
    plt.subplot(2,2,3)
    plt.plot(extract(dfd[cycle], start, end)['Total Time'], extract(dfd[cycle], start, end)['Step'])

    plt.suptitle(f'Cell D Cycle {cycle} Impluse Number {number}')
    
    plt.show()
    
    return [A, dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].values[0]], [B,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].values[0]], [C,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].values[0]], [D,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].values[0]]

def d_locate_ABCD_n(dfd, cycle, number):     # locate points ABCD without plotting them on graphs, not sure if it works on cellc
    '''
    dfd represents the list of dataframes you're using
    cycle represents the specific cycle you're using
    number represents the impulse you're locating
    '''
    A,C = locate(dfd[cycle], 6, number, offset=1)
    #print(C)
    B,x = locate(dfd[cycle], 6, number)
    D,E = locate(dfd[cycle], 7, number)
    return [A, dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].iloc[0]], [B,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].iloc[0]], [C,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].iloc[0]], [D,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].iloc[0]]

################################### cell c ###########################################
def c_locate_ABCD(dfc, cycle, number):      # my guessing of how to plot ABCD in cell c
    t_s, t_e1 = locate(dfc[cycle], 7, number, offset=3)
    t_s1, t_e = locate(dfc[cycle], 14, number, offset_2=3)
    
    A1,C1 = locate(dfc[cycle], 7, number, offset=1)
    B1,x = locate(dfc[cycle], 7, number)
    D1,x = locate(dfc[cycle], 9, number, offset=1)
    B2, C2 = locate(dfc[cycle], 9, number)
    x, D2 = locate(dfc[cycle], 10, number)
    

    plt.subplot(2,2,2)
    plt.plot(extract(dfc[cycle], t_s, t_e)['Total Time'], extract(dfc[cycle], t_s, t_e)['Current'])
    #plotAll(dfc, 'Current', t_s, t_e)
             
    plt.subplot(2,2,1)
    plt.plot(extract(dfc[cycle], t_s, t_e)['Total Time'], extract(dfc[cycle], t_s, t_e)['Voltage'])
    
    plt.plot([A1, B1], [dfc[cycle].loc[dfc[cycle]['Total Time'] == A1, 'Voltage'].values[0],
                      dfc[cycle].loc[dfc[cycle]['Total Time'] == B1, 'Voltage'].values[0]])
    plt.plot([B1, C1], [dfc[cycle].loc[dfc[cycle]['Total Time'] == B1, 'Voltage'].values[0],
                      dfc[cycle].loc[dfc[cycle]['Total Time'] == C1, 'Voltage'].values[0]]) 
    plt.plot([C1, D1], [dfc[cycle].loc[dfc[cycle]['Total Time'] == C1, 'Voltage'].values[0],
                      dfc[cycle].loc[dfc[cycle]['Total Time'] == D1, 'Voltage'].values[0]])
    
    plt.plot([D1, B2], [dfc[cycle].loc[dfc[cycle]['Total Time'] == D1, 'Voltage'].values[0],
                      dfc[cycle].loc[dfc[cycle]['Total Time'] == B2, 'Voltage'].values[0]])
    plt.plot([B2, C2], [dfc[cycle].loc[dfc[cycle]['Total Time'] == B2, 'Voltage'].values[0],
                      dfc[cycle].loc[dfc[cycle]['Total Time'] == C2, 'Voltage'].values[0]])
    plt.plot([C2, D2], [dfc[cycle].loc[dfc[cycle]['Total Time'] == C2, 'Voltage'].values[0],
                      dfc[cycle].loc[dfc[cycle]['Total Time'] == D2, 'Voltage'].values[0]])

    plt.text(A1, dfc[cycle].loc[dfc[cycle]['Total Time'] == A1, 'Voltage'].iloc[0], 'A1')
    plt.text(B1, dfc[cycle].loc[dfc[cycle]['Total Time'] == B1, 'Voltage'].iloc[0], 'B1')
    plt.text(C1, dfc[cycle].loc[dfc[cycle]['Total Time'] == C1, 'Voltage'].iloc[0], 'C1')
    plt.text(D1, dfc[cycle].loc[dfc[cycle]['Total Time'] == D1, 'Voltage'].iloc[0], 'D1 / A2')
    plt.text(B2, dfc[cycle].loc[dfc[cycle]['Total Time'] == B2, 'Voltage'].iloc[0], 'B2')
    plt.text(C2, dfc[cycle].loc[dfc[cycle]['Total Time'] == C2, 'Voltage'].iloc[0], 'C2')
    plt.text(D2, dfc[cycle].loc[dfc[cycle]['Total Time'] == D2, 'Voltage'].iloc[0], 'D2')
    
    #plotAll(dfc, 'Voltage', t_s, t_e)
    plt.suptitle(f'Cell C Cycle {cycle} Impulse Number {number}')
             
    plt.subplot(2,2,3)
    plt.plot(extract(dfc[0], t_s, t_e)['Total Time'], extract(dfc[0], t_s, t_e)['Step'])
    #plotAll(dfc, 'Step', t_s, t_e)

    plt.show()

    return [A1, dfc[cycle].loc[dfc[cycle]['Total Time'] == A1, 'Voltage'].iloc[0]], [B1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == B1, 'Voltage'].iloc[0]], [C1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == C1, 'Voltage'].iloc[0]], [D1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == D1, 'Voltage'].iloc[0]], [D1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == D1, 'Voltage'].iloc[0]], [B2,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == B2, 'Voltage'].iloc[0]], [C2,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == C2, 'Voltage'].iloc[0]], [D2,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == D2, 'Voltage'].iloc[0]]

def c_locate_ABCD_n(dfc, cycle, number):      # my guessing of how to plot ABCD in cell c
    
    A1,C1 = locate(dfc[cycle], 7, number, offset=1)
    B1,x = locate(dfc[cycle], 7, number)
    D1,x = locate(dfc[cycle], 9, number, offset=1)
    B2, C2 = locate(dfc[cycle], 9, number)
    x, D2 = locate(dfc[cycle], 10, number)

    return [A1, dfc[cycle].loc[dfc[cycle]['Total Time'] == A1, 'Voltage'].iloc[0]], [B1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == B1, 'Voltage'].iloc[0]], [C1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == C1, 'Voltage'].iloc[0]], [D1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == D1, 'Voltage'].iloc[0]], [D1,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == D1, 'Voltage'].iloc[0]], [B2,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == B2, 'Voltage'].iloc[0]], [C2,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == C2, 'Voltage'].iloc[0]], [D2,
                dfc[cycle].loc[dfc[cycle]['Total Time'] == D2, 'Voltage'].iloc[0]]

def calc_R0_sample(A,B,C,D,I):        # Not Final Value
    return 0.5*(A-B+D-C)/I

if __name__ == "__main__":
    
    '''   
    for df in dfd:        # noticing that cycle 13 and 12 have only 9 impulses. 
        mask = df['Step'] == 7
        df['start'] = mask& ~mask.shift(1, fill_value=False)
        df['end'] = mask& ~mask.shift(-1, fill_value=False)
        start_indices = df.index[df['start']].tolist()
        end_indices = df.index[df['end']].tolist()
        print(len(start_indices))
   
    plt.plot(dfd[13]['Total Time'], dfd[13]['Voltage'])
    plt.plot(dfd[12]['Total Time'], dfd[12]['Voltage'])
    plt.plot(dfd[11]['Total Time'], dfd[11]['Voltage'])
    plt.show()
    '''
    R0 = []
    for i in range(len(dfd)):
        mask = dfd[i]['Step'] == 7
        dfd[i]['start'] = mask& ~mask.shift(1, fill_value=False)
        start_indices = dfd[i].index[dfd[i]['start']].tolist()
        for j in range(len(start_indices)):
            A,B,C,D = d_locate_ABCD_n(dfd, i, j)
            R0.append(calc_R0_sample(A[1],B[1],C[1],D[1],30))

    plt.plot(R0)
    plt.show()
    
    
    
    '''
    A, B, C, D = d_locate_ABCD_n(dfd, 1, 7)
    print(A,B,C,D, '\n')

    A1, B1, C1, D1, A2, B2, C2, D2 = c_locate_ABCD(dfc, 0, 7)
    print(A1, B1, C1, D1, A2, B2, C2, D2)
    '''
    
