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

def locate_ABCD(dfd, cycle, number):        # locate points ABCD while plot them on a graph, not sure if it works on cellc
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

    
    plt.text(A, dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].iloc[0], 'A')
    plt.text(B, dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].iloc[0], 'B')
    plt.text(C, dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].iloc[0], 'C')
    plt.text(D, dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].iloc[0], 'D')
    
    plt.subplot(2,2,2)
    plt.plot(extract(dfd[cycle], start, end)['Total Time'], extract(dfd[cycle], start, end)['Current'])
    
    plt.subplot(2,2,3)
    plt.plot(extract(dfd[cycle], start, end)['Total Time'], extract(dfd[cycle], start, end)['Step'])

    plt.show()
    
    return [A, dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].iloc[0]], [B,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].iloc[0]], [C,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].iloc[0]], [D,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].iloc[0]]

def locate_ABCD_n(dfd, cycle,  number):     # locate points ABCD without plotting them on graphs, not sure if it works on cellc
    '''
    dfd represents the list of dataframes you're using
    cycle represents the specific cycle you're using
    number represents the impulse you're locating
    '''
    A,C = locate(dfd[cycle], 6, number, offset=1)
    B,x = locate(dfd[cycle], 6, number)
    D,E = locate(dfd[cycle], 7, number)
    return [A, dfd[cycle].loc[dfd[cycle]['Total Time'] == A, 'Voltage'].iloc[0]], [B,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == B, 'Voltage'].iloc[0]], [C,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == C, 'Voltage'].iloc[0]], [D,
                dfd[cycle].loc[dfd[cycle]['Total Time'] == D, 'Voltage'].iloc[0]]
    
if __name__ == "__main__":
    A, B, C, D = locate_ABCD(dfd, 0, 1)
    print(A,B,C,D)
    
