import matplotlib.pyplot as plt
import numpy as np
xs, ys = [], []
for ln in open(r"C:\Users\morit\Downloads\Semesterproject\ippl\build\renderdata\radiation_4.csv"):
    try:
        x, y = map(float, ln.replace(",", " ").split()[:2])
    except (ValueError, IndexError):
        xs, ys = [], []  # header line -> restart, so only the latest run is kept
        continue
    xs.append(x); ys.append(y)
plt.plot(xs, ys); plt.xlabel("z [m]"); plt.ylabel("radiated power [W]"); plt.tight_layout(); 
plt.yscale("log")
plt.ylim([100, 1e8])
plt.show()

np.save("radiation_data.npy", (xs, ys))
