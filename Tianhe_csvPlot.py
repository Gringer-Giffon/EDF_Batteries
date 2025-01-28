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

if __name__ == "__main__":
    plt.subplot(2,2,1)
    plotAll(dfd, 'Voltage', 0, 35000)
    plt.subplot(2,2,2)
    plotAll(dfd, 'Current', 0, 35000)
    plt.subplot(2,2,3)
    plotAll(dfd, 'Step', 0, 35000)

    plt.show()
