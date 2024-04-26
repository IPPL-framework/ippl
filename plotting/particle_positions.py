"""Module to plot the particle positions from a txt file."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# Change this path to the path of the txt file
FILE = "../build_serial/alvine/data/particle_positions.txt"

# Read data from txt file
with open(FILE, 'r') as file:
    lines = file.readlines()

# Process data
data = []
# Ignore header
data.append(lines[0].strip().split(','))
# Extract time and position data
for line in lines[1:]:
    parts = line.strip().split(',')
    time = [float(parts[0])]
    position = [float(coord.strip('( ) ')) for coord in parts[1:]]
    data.append(time+position)

# Turn data into a pandas dataframe
data = pd.DataFrame(data[1:], columns=data[0])
time_steps = sorted(data['time'].unique())

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter([], [], c='b', marker='.')

# Set limits and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(data.x.min(), data.x.max())
ax.set_ylim(data.y.min(), data.y.max())
ax.set_title('Particles Animation (X vs Y)')
ax.grid(True)

def update(frame):
    """Update function for animation."""
    df_frame = data[data['time'] == frame]
    scatter.set_offsets(df_frame[['x', 'y']].values)
    ax.set_title(f'Particles at Time = {frame:.4f}')
    return scatter,

# Create animation
ani = FuncAnimation(fig, update, frames=time_steps, blit=True)
ani.save('basic_animation.gif', fps=10)
