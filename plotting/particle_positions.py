"""Module to plot the particle positions from a txt file."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from matplotlib.animation import FuncAnimation

# Change this path to the path of the txt file
PATH = "../build/alvine"

# Find all rank files and combine them
rank_files = glob.glob(f'{PATH}/particles_rank_*.csv')
print(f"Found {len(rank_files)} rank files: {rank_files}")

# Load and combine data from all rank files
dfs = []
for file in rank_files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Sort by time and index to ensure proper ordering
df = df.sort_values(['time', 'index']).reset_index(drop=True)

print(f"Combined data: {len(df)} total rows")
print(f"Time steps: {df['time'].nunique()}")
print(f"Particles per time step: {df.groupby('time').size().iloc[0] if len(df) > 0 else 0}")

# Find the unique times and indices for particles
times = df['time'].unique()
particle_indices = df['index'].unique()

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim((0,10))
ax.set_ylim((0,10))

# Initialize scatter plot with empty data and color map
scat = ax.scatter([], [], c=[], s=2)

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
ani.save(f'particles_combined.gif', fps=30)
