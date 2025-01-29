import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import main

folderPath = f'./cells_data'

# dataframes = [pd.read_csv(file) for file in os.listdir(folderPath)]

"""
time = data["Total Time"]
current = data["Current"]
voltage = data["Voltage"]
step = data["Step"]
"""

main.plot_test("C", "01")
plt.show()
