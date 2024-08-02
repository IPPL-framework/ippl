"""Module to plot the particle positions from a txt file."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Change this path to the path of the txt file
PAR_DIST="../build_serial/alvine"

PATH=f"{PAR_DIST}"

# Load data from CSV file
df = pd.read_csv(f'{PATH}/particles.csv')

# Group the data by time
grouped = df.groupby('time')

# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter([], [], [], c=[], cmap='viridis')


def init():
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return sc,

def update(frame):
    data = grouped.get_group(frame)
    sc._offsets3d = (data['pos_x'], data['pos_y'], data['pos_z'])
    sc.set_array(data['vor_z'])
    return sc,

ani = FuncAnimation(fig, update, frames=grouped.groups.keys(), init_func=init, blit=False, repeat=True)

# Show animation
# plt.show()
print(f"Saving animation to {PATH}/particles.gif")
ani.save(f'{PATH}/particles.gif', fps=30)
