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

def locate(df, step, pattern, offset=0):        # Extract the df with respect to steps
    mask = df['Step'] == step
    df['start'] = mask& ~mask.shift(1, fill_value=False)
    df['end'] = mask& ~mask.shift(-1, fill_value=False)
    start_indices = df.index[df['start']].tolist()
    end_indices = df.index[df['end']].tolist()
    return df['Total Time'][start_indices[pattern]-offset], df['Total Time'][end_indices[pattern]]

if __name__ == "__main__":
    start, end1 = locate(dfd[0], 6, 1, 1)
    start1, end = locate(dfd[0], 7, 1)
    
    plt.subplot(2,2,2)
    #plotAll(dfd, 'Voltage', start, 42000)
    plt.plot(extract(dfd[0], start, end)['Total Time'], extract(dfd[0], start, end)['Voltage'])
    
    plt.subplot(2,2,1)
    #plotAll(dfd, 'Current', 36000, 42000)
    plt.plot(extract(dfd[0], start, end)['Total Time'], extract(dfd[0], start, end)['Current'])
    
    plt.subplot(2,2,3)
    #plotAll(dfd, 'Step', 36000, 42000)
    plt.plot(extract(dfd[0], start, end)['Total Time'], extract(dfd[0], start, end)['Step'])

    '''
    plt.text(37300, 7.3, "step7:\nan instant and\nrapid discharge", fontsize=12, color="red")
    plt.text(39000, 9.3, "step9:\nan instant and\nrapid charge", fontsize=12, color="red")
    
    plt.text(41000, 10.3, "step10:\na continous discharge", fontsize=12, color="blue")
    '''

    plt.show()
    
