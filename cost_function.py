#import numpy as np
import pandas as pd

def cost(df, column_name='Model Voltage'):
    if column_name not in df.columns or 'Voltage' not in df.columns:
        print("pls enter the column name, default as 'Model Voltage'")
        return df
    else:
        error = [df[column_name][i] - df['Voltage'][i] for i in range(len(df))]
        df['Error'] = error
        return df
