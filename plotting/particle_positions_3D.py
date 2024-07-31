"""Module to plot the particle positions from a txt file."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Change this path to the path of the txt file
PAR_DIST="../build_serial/alvine"
VOR_DIST="Disk"

PATH=f"{PAR_DIST}"

# Load data from CSV file
df = pd.read_csv(f'{PATH}/particles.csv')

# Find the unique times and indices for particles
times = df['time'].unique()
particle_indices = df['index'].unique()

# Create a figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Initialize scatter plot with empty data and color map
scat = ax.scatter([], [], [], c=[], s=2, cmap='viridis')

def update(frame):
    current_time = times[frame]
    frame_data = df[df['time'] == current_time]
    scat._offsets3d = (frame_data['pos_x'], frame_data['pos_y'], frame_data['pos_z'])
    scat.set_array(frame_data['vor_z'])  # Set colors based on vorticity
    return scat,

# Create animation
ani = FuncAnimation(fig, update, frames=len(times), blit=True, interval=50)

# Show animation
# plt.show()
print(f"Saving animation to {PATH}/particles.gif")
ani.save(f'{PATH}/particles.gif', fps=30)
