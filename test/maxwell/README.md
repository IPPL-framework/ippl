## FDTD Solver Testcases

This directory contains testcases for both the Standard and Non-Standard FDTD solvers. There are two types of tests:

- **Image Output Testcases**: Visualize the field evolution of a gaussian pulse over a time increment of one. With periodic boundary conditions the wave is expected to return to the initial state.
- **Convergence Testcases**: Compute and record the convergence error for different grid sizes and directions, and plot the results. The L2 error between the initial field and the field after a time of one is calculated.

---

### 1. Running the Image Output Testcases

1. **Build the executables** (in existing IPPL build directory):
   ```bash
   cd ippl/build/test/solver/
   make TestStandardFDTDSolver
   make TestNonStandardFDTDSolver
   ```

2. **Create the required output folders**:
   ```bash
   mkdir -p renderdataStandard
   mkdir -p renderdataNonStandard
   ```

3. **Run the testcases**:
   ```bash
   ./TestStandardFDTDSolver
   ./TestNonStandardFDTDSolver
   ```

4. **Image Output**:
   Every 4th timestep an images of the filed will be written to:
    - `renderdataStandard/outimageXXXXX.bmp` (Standard solver)
    - `renderdataNonStandard/outimageXXXXX.bmp` (Non-Standard solver)

---

### 2. Running the Convergence Testcases

1. **Build the executables**:
   ```bash
   make TestStandardFDTDSolver_convergence
   make TestNonStandardFDTDSolver_convergence
   ```

2. **Run the testcases**:
   ```bash
   ./TestStandardFDTDSolver_convergence
   ./TestNonStandardFDTDSolver_convergence
   ```

3. **Output**:
   This will generate CSV files:
    - `StandardFDTDSolver_convergence.csv`
    - `NonStandardFDTDSolver_convergence.csv`

### 3. Plotting the Convergence Results

The following python code can be used to create the convergance plot:

```bash
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
filenames = ["StandardFDTDSolver_convergence.csv", "NonStandardFDTDSolver_convergence.csv"]

for filename in filenames:
    df = pd.read_csv(filename)

    solvername = filename.split("FDTD")[0]

    # Plot figure
    plt.figure(figsize=(8,6))

    plt.plot(df.loc[df["GaussianPulseDir"] == 'x']["NGridpoints"], df.loc[df["GaussianPulseDir"] == 'x']["ConverganceError"], 'b-', label='L_2 error for x wave')
    plt.plot(df.loc[df["GaussianPulseDir"] == 'y']["NGridpoints"], df.loc[df["GaussianPulseDir"] == 'y']["ConverganceError"], 'c--',  label='L_2 error for y wave')
    plt.plot(df.loc[df["GaussianPulseDir"] == 'z']["NGridpoints"], df.loc[df["GaussianPulseDir"] == 'z']["ConverganceError"], 'm:', label='L_2 error for z wave')
    plt.plot(df.loc[df["GaussianPulseDir"] == 'z']["NGridpoints"], df.loc[df["GaussianPulseDir"] == 'z']["NGridpoints"].astype(float)**(-2) * 2*max(df["ConverganceError"]), 'k-', linewidth=0.5, label=r'O($\Delta x^{2}$)')
    plt.plot(df.loc[df["GaussianPulseDir"] == 'z']["NGridpoints"], df.loc[df["GaussianPulseDir"] == 'z']["NGridpoints"].astype(float)**(-1) * 2*max(df["ConverganceError"]), 'k--', linewidth=0.5, label=r'O($\Delta x$)')
    plt.yscale('log', base=2)
    #plt.xscale('log', base=2)
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Error')
    plt.title(f'{solvername} Finite Differences Solver Error')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{solvername}FDTDSolver_convergence')
```

This will generate the convergence plots for both solvers and save them as image files in the current directory.
