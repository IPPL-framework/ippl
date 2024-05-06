"""Module to plot the particle positions from a txt file."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation

# Change this path to the path of the txt file
PAR_DIST="EquidistantDistribution"
VOR_DIST="Disk"

PATH=f"{PAR_DIST}_{VOR_DIST}"

# Load data from CSV file
df = pd.read_csv(f'{PATH}/energy.csv')

plt.plot(df['energy'])
# Show animation
# plt.show()
plt.savefig(f'{PATH}/energy.png')