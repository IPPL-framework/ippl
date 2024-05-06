"""Module to plot the particle positions from a txt file."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation

# Change this path to the path of the txt file
PATH = "../build_serial/alvine"

# Load data from CSV file
df = pd.read_csv(f'{PATH}/particles.csv')

# Find the unique times and indices for particles
times = df['time'].unique()
particle_indices = df['index'].unique()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim((0,10))
ax.set_ylim((0,10))

# Initialize scatter plot with empty data and color map
scat = ax.scatter([], [], c=[], s=2, cmap='viridis_r')

def update(frame):
    current_time = times[frame]
    frame_data = df[df['time'] == current_time]
    scat.set_offsets(np.c_[frame_data['pos_x'], frame_data['pos_y']])
    scat.set_array(frame_data['vorticity'])  # Set colors based on vorticity
    return scat,

# Create animation
ani = FuncAnimation(fig, update, frames=len(times), blit=True, interval=50)

# Show animation
# plt.show()
print("Saving animation to particles.gif")
ani.save(f'particles.gif', fps=30)

