# Working on the FEL mini-app

This guide follows the implementation in `demos/fel`. It is intended for a
student who needs to understand the data, trace one timestep and choose a useful
validation task before changing the physics. The linked API and source pages
cover this directory; IPPL, Kokkos, MPI and Catalyst remain external dependencies.

Start here: [scope](#fel_scope), [build and run](#fel_run),
[source map](#fel_map), [input and units](#fel_input),
[frames and mesh](#fel_frames), [one timestep](#fel_step),
[equations and discretization](MathematicalModel.md),
[parallel execution](#fel_parallel), [outputs](#fel_output),
[validation](#fel_validation), [student tasks](#fel_tasks),
[documentation maintenance](#fel_docs).

## What is implemented? {#fel_scope}

The program tracks a relativistic bunch through an analytic static undulator.
It evolves self-fields in a z-directed moving Lorentz frame, using a uniform
Cartesian mesh and a collocated four-potential FDTD solver. A relativistic Boris
pusher advances particles in the grid fields plus the transformed undulator.
The implementation is three-dimensional even where interfaces carry a `Dim`
template parameter.

The active `FDTDSolver_t` alias selects IPPL's `NonStandardFDTDSolver` with
absorbing boundaries. Its source is a four-vector containing charge density and
current; its unknowns are scalar/vector potentials. The boundary implementation
uses second-order Mur conditions on those potentials. This is a vacuum model:
there is no spatial permittivity field, dielectric interface model or prism in
this mini-app. Reusing its infrastructure for ChDR requires those physics
extensions as well as geometry construction.

The MITHRA-derived routines and radiation diagnostics are useful starting
points. Their presence alone does not establish discrete charge conservation,
absolute radiation accuracy or FEL gain. The validation section distinguishes
available infrastructure from checks still needed.

## Build and first run {#fel_run}

From the repository root, using your existing IPPL compiler/Kokkos/MPI setup:

```sh
cmake -S . -B build -DIPPL_ENABLE_FEL=ON -DCMAKE_CXX_STANDARD=20
cmake --build build --target FreeElectronLaser -j 4
./build/demos/fel/FreeElectronLaser ./demos/fel/config.json --info 5
```

FEL needs the Conduit API supplied by Catalyst for JSON parsing, including when
the optional Catalyst visualization path is disabled. CMake finds or fetches
that dependency when FEL is enabled.

The example is copied to `build/demos/fel/config.json`. Running
`./FreeElectronLaser --info 5` **from that directory** uses this copy. Otherwise
pass the path explicitly: the fallback `config.json` is relative to the process
working directory, with no search beside the executable. The first remaining
argument after IPPL initialization must be the path; the application only
examines `argv[1]` for this purpose.

For MPI, prefix the same command with `mpirun -np 2`, adding any launcher options
required by your machine. Set `OMP_NUM_THREADS` to suit the allocated cores for
an OpenMP build. All ranks must see the configuration and output paths.

The shipped grid is **96 x 96 x 3000**, or 27,648,000 owned cells. The principal
fields alone contain 22 doubles per cell: E (3), B (3), J (4), and three
four-potential histories (12), about **4.87 GB** before halos, particles, temporary
storage and the diagnostic history buffer. Start with a smaller grid and a short
duration for a first exercise. Preserve the transformed-grid spacing condition
below; reducing longitudinal resolution arbitrarily can violate it. A short run
checks execution, not radiation convergence.

For a reproducible startup exercise, copy the shipped JSON and change
`mesh.resolution` to `[12,12,128]`, `mesh.total-time` to `100.0` (picoseconds),
and `bunch.number-of-particles` to `1024`, keeping its other physical parameters.
Set a fresh `output.path`. This gives three timesteps and was checked on one and
two MPI ranks with the OpenMP build. Tails increase the generated count above
the requested count. Three steps are too short to assess downstream radiation.

Use a fresh `output.path` for every run: diagnostic files are opened in append
mode, and another run can append another header. Relative output paths resolve
from the working directory. `timing.dat` is written in that working directory,
independently of `output.path`.

## Source map and ownership {#fel_map}

| Read in this order | Responsibility |
|---|---|
| FreeElectronLaser.cpp | IPPL lifetime, input path, manager lifecycle, timings |
| Config.h and units.h | JSON parsing, input contract and unit conversions |
| datatypes.h | Mesh/layout/field aliases and active solver selection |
| FreeElectronLaserManager.h | Initialization, deposition, push and diagnostics |
| FELFieldContainer.hpp | Own E, B and J on the distributed layout |
| FELParticleContainer.hpp | Register particle attributes for migration |
| MithraBunch.h | Generate initial positions and normalized momenta on the host |
| LorentzTransform.h | Transform positions, momenta and fields between frames |
| Undulator.h | Evaluate the analytic external field |

The diagram shows ownership/use; the base classes and solver implementation
live in IPPL outside the local reference.

@dot
digraph fel_architecture {
  graph [rankdir=TB, bgcolor="transparent", nodesep=0.25, ranksep=0.5];
  node [shape=box, style="rounded,filled", fillcolor="#edf3fa", fontname="Helvetica", fontsize=11];
  edge [fontname="Helvetica", fontsize=10, color="#52677f"];
  main [label="main()\nIPPL lifetime + JSON input"];
  base [label="IPPL BaseManager\npre_step / advance / post_step", fillcolor="#f2f2f2"];
  manager [label="FreeElectronLaserManager\nsetup + PIC loop + diagnostics"];
  fields [label="FELFieldContainer\nUniformCartesian + FieldLayout\nE, B, J"];
  particles [label="FELParticleContainer\nR, R_nm1, gamma_beta, Q, mass\nregistered particle attributes"];
  solver [label="NonStandardFDTDSolver\nthree four-potential histories\nreconstruct E and B", fillcolor="#f2f2f2"];
  bunch [label="MithraBunch\nrank-zero host generator"];
  physics [label="UniaxialLorentzframe + Undulator\nexternal field in moving frame"];
  main -> manager [label="creates"];
  manager -> base [arrowhead=empty, label="inherits"];
  manager -> fields [label="owns"];
  manager -> particles [label="owns"];
  manager -> solver [label="owns"];
  manager -> bunch [label="initialization"];
  manager -> physics [label="uses"];
  solver -> fields [style=dashed, label="reads J; writes E/B"];
}
@enddot

The field container allocates E, B and J; the active solver allocates the three
potential histories. The optional Catalyst adaptor can
expose fields and particles during execution; it does not replace the normal
diagnostic files.

## Input values, units and bunch shape {#fel_input}

The FEL reader accepts **JSON**, as shown in `config.json`. ChDR's separate
mesh constructor has a YAML reader; that reader is not used here. The following
table highlights interpretation; read_config() documents the full parser contract.

| Input | Interpretation |
|---|---|
| `mesh.length-scale`, `mesh.time-scale` | Units of input lengths and duration; the example uses micrometres and picoseconds |
| `mesh.extents`, `mesh.resolution` | Three full input lengths and three global cell counts, in x,y,z order |
| `mesh.total-time` | Input duration, converted to internal units and later divided by frame gamma |
| `mesh.space-charge` | Whether to deposit charge density into J[0]; current is deposited in either case |
| `timestep-ratio` | Parsed, default 1; currently not used to select the timestep |
| `bunch.charge`, `bunch.mass` | Total signed charge in elementary-charge magnitudes and total mass in electron masses |
| `bunch.number-of-particles` | Requested simulation-particle count; use the actual generated count for normalization |
| `bunch.gamma` | Laboratory Lorentz factor, not kinetic energy in MeV |
| `bunch.sigma-position` | Input generator widths; z is not a Gaussian rms width in the active uniform-longitudinal generator |
| `bunch.sigma-momentum` | Dimensionless component spread in gamma beta = p/(mc) |
| `bunch.distribution-truncations` | Absolute cutoff lengths; the generator uses x for both transverse directions and z for the longitudinal cutoff |
| `undulator.static-undulator` | Dimensionless K, period and total length in the configured length units |
| `output.path` | Directory, default `../data/`; relative to working directory |

The sign of the charge is taken directly from the input. The example supplies a
positive value; the reader does not add an electron minus sign. The manager
divides total charge and total mass by the **actual** generated count, preserving
their ratio for the pusher.

The active MITHRA configuration uses random sampling, transverse Gaussian
widths, a uniform longitudinal core with half-width `sigma-position[2]`, Gaussian
tails and a prescribed initial bunching amplitude of 0.01. Shot noise is off.
These choices are set in generate_mithra_config(), not exposed as general JSON
options. Truncation, modulation and tails change measured moments; inspect the
generated distribution before identifying an input width with a physical rms
bunch length. The manager subsequently removes the global position centroid,
so the requested `bunch.position` does not retain an arbitrary final offset.

For an energy specification, first convert kinetic energy to Lorentz factor:

@f[
\gamma=1+\frac{E_{\rm kin}}{m_e c^2},\qquad
\beta=\sqrt{1-\gamma^{-2}}.
@f]

In particular, setting `gamma` to 60 would not specify 60 MeV kinetic energy.
Likewise, directly entering a desired rms pulse length into the active uniform
generator's z-width would not reproduce the desired distribution.

The internal solver equations use c=1. units.h defines the base scales
@f$L_0=1.616255\times10^{-5}\,\mathrm{m}@f$ and
@f$T_0=5.391247\times10^{-14}\,\mathrm{s}@f$, plus mass and charge scales.
Convert a length by @f$x_{\rm code}=x_{\rm SI}/L_0@f$ and recover SI by
multiplication. These are scaled units, despite accepted input aliases called
`natural` or `planck`. They differ from the SI coordinates used by ChDR mesh-1.

The parser checks required values, types, shapes, integer range and selected
finite physical values. It is not complete physical validation: for example,
zero cell counts are representable and some bunch quantities lack positivity
or finiteness checks. An unknown length/time unit warns and falls back to metres
or seconds, respectively. Check the log and use the supported spellings.

## Frames, mesh and field equations {#fel_frames}

The manager constructs a z-axis boost using

@f[
\gamma_f=\max\left(1,\frac{\gamma_b}{\sqrt{1+K^2/2}}\right).
@f]

In pre_run(), it multiplies the input z extent by @f$\gamma_f@f$ and divides
the input duration by @f$\gamma_f@f$. The transverse extents and all cell counts
remain unchanged. The resulting simulation box is centred at zero, with origin
@f$-\mathbf{L}/2@f$ and spacings @f$h_i=L_i/N_i@f$. It is nonperiodic and
decomposed along z only. These are the program's setup conventions; distinguish
them from a general Lorentz transformation of an arbitrary laboratory domain.

The active nonstandard solver sets

@f[
\Delta t=h_z,\qquad
\left(\frac{h_z}{h_x}\right)^2+
\left(\frac{h_z}{h_y}\right)^2<1
@f]

in c=1 units. Both the manager and solver enforce this strict spacing condition.
Use the **boosted** z spacing when checking an input. `timestep-ratio` does not
override this choice. The step count is the ceiling of duration/step size, so
the last step can pass the requested duration by less than one timestep.

The source and potentials have four collocated components:

@f[
J=(\rho,J_x,J_y,J_z),\qquad A=(\phi,A_x,A_y,A_z).
@f]

The solver holds three potential time levels, advances a vacuum wave stencil,
applies its absorbing boundary treatment, shifts the histories, and reconstructs
fields using the discrete counterparts of

@f[
\mathbf{E}=-\nabla\phi-\partial_t\mathbf{A},\qquad
\mathbf{B}=\nabla\times\mathbf{A}.
@f]

The spatial reconstruction is centred and the potential time derivative uses
the two most recent levels. Start a solver investigation in
`src/MaxwellSolvers/NonStandardFDTDSolver.hpp` and
`src/MaxwellSolvers/FDTDSolverBase.hpp`; current deposition is implemented in
`src/Interpolation/CurrentDeposition.hpp`. There is no
initial self-field solve: the potential histories start at zero. Separate this
startup transient from the radiation being measured.

For the complete formulas, read [Mathematical model and discretization](MathematicalModel.md).
It derives the implemented nonstandard stencil and its dispersion relation, then
gives the source weights, actual field time levels, Boris update and Mur boundary
equations, with references to MITHRA 2.0 and a map back to the source code.

## Trace one timestep {#fel_step}

After output-directory creation and mesh/field setup, the manager generates
particles on rank zero, transfers them into IPPL attributes, removes the global
position centroid and migrates them to their owners. It writes diagnostics once
before the first timestep. IPPL BaseManager then calls the following sequence:

1. **pre_step():** log completion of this otherwise empty hook.
2. **advance(), deposition:** clear J and deposit from the previous trajectory
   `R_nm1 -> R`. Deposit rho when `space-charge` is true, then accumulate source
   halos onto their owners.
3. **advance(), fields:** solve the four-potential update and reconstruct E/B.
   If enabled, Catalyst executes here, before the particle push.
4. **advance(), particles:** save R in R_nm1, fill E/B halos, and take three
   relativistic Boris substeps. Each substep gathers at the updated particle
   positions and samples the transformed external undulator at its substep
   time. The grid fields retain their current time level throughout these
   substeps. Delete particles on/outside the physical box, then migrate survivors.
5. **post_step():** increment simulation time and iteration, write the three
   diagnostic outputs, and report progress.

The initializer copies the same positions into R_nm1 and R, so the first
deposition has zero displacement and contributes no current; rho may still be
deposited when enabled. Later steps use the positions saved before the preceding
push. `R_np1` is registered in the particle container but is not used by the
active push. The separate grid2par()/gatherFields() hooks are not the gather path
used by advance(); gathering occurs within each push substep.

The collocated deposition uses midpoint CIC weighting for current and a
separate CIC charge deposition. Do not infer a discrete continuity identity
from its name or from the existence of rho: measure the continuity and Gauss-law
residuals with the same discrete operators as the field solver.

## MPI, Kokkos and storage {#fel_parallel}

Global cell counts describe the complete mesh. Each MPI rank owns a z slab
plus ghost cells; kernel view indices and global physical indices differ by
the local domain's first index and ghost width. Particle positions stay in
physical simulation coordinates. Use the existing mesh origin/spacing and
local-layout conversions when adding a deposition or diagnostic.

Kokkos kernels operate on local device views. Deposition uses atomics where
particles can update the same cell. Halo accumulation **adds** deposited
contributions to their owning cells; halo filling **copies** neighbouring field
values for stencil/gather access. They are different operations.

The initial MITHRA generator uses host storage on rank zero and then copies to
particle views. Subsequent field/particle work is distributed, and particle
`update()` migrates registered attributes together. Diagnostics reduce local
values to rank zero; the normal loop does not gather the entire 3D field.
Source/field halos and particle migration still communicate data each step.

The banded diagnostic adds a history buffer of transverse field samples. Its
window is approximately three selected cycles, and the buffer is allocated on
each rank even though only the rank owning the sampling plane contributes its
power. Include this allocation, halos, temporary views and particle attributes
when budgeting memory at scale. Three particle substeps do not relax the field
solver's mesh condition or resolve inadequate field/particle sampling.

## Read the outputs correctly {#fel_output}

| File in `output.path` | Contents and caveats |
|---|---|
| `radiation_N.csv` | Downstream plane integral of the z-directed flux from lab-transformed fields, scaled to watts |
| `radiation_band_N.csv` | Selected-frequency DFT power estimate from a sliding history window; not a full spectrum |
| `feldiag_N.csv` | Bunching, peak E, field energy, live count, bunch position/size and mean longitudinal gamma beta |

Here N is the number of MPI ranks. Headers contain commas, but data rows contain
whitespace separators. For a fresh, single-run file, read it explicitly, for example:

```python
from pathlib import Path
import pandas as pd

path = Path("renderdata/feldiag_1.csv")
with path.open() as stream:
    columns = [name.strip() for name in stream.readline().split(",")]
data = pd.read_csv(path, sep=r"\s+", skiprows=1, names=columns)
```

`labframe_z` is in metres. The implementation obtains this distance label by
transforming the point with simulation coordinate `z = extents[2]`. This is not
the geometric coordinate of the sampled downstream plane in the centred box,
whose upper boundary is `extents[2]/2`. Verify or revise this convention before
using the axis as an absolute monitor position.

In `feldiag`, `max_E`, `field_energy` and position/size columns retain internal
units. The energy is the domain sum of @f$(E^2+B^2)/2@f$ times cell volume; it is
not directly joules and excludes particle energy and boundary flux. `rms_z` is
centred about the bunch centroid, whereas `rms_perp` is
@f$\sqrt{\langle x^2+y^2\rangle}@f$, without transverse centroid subtraction.

The bunching diagnostic uses wavelength `undulator_period/(2*frame_gamma)`.
The banded-power diagnostic instead uses `undulator_period/frame_gamma`, with
lab-transformed field samples and simulation-frame time intervals. Treat the
frequency/frame convention and absolute normalization as validation questions.
The initially empty history also produces a startup window transient. A growing
curve by itself does not demonstrate physical FEL amplification.

## Validation status and useful checks {#fel_validation}

There are no FEL-specific CTest cases registered in this directory. The sources
in `test/maxwell` include standard/nonstandard vacuum pulse and convergence
drivers, but their CMake registration is **compile-only**. Building them is not
equivalent to running a convergence study. ChDR geometry tests exercise material
construction and shared containers, not FEL radiation physics.

Before interpreting a new result, choose a check tied to the change:

| Change or question | Useful evidence |
|---|---|
| Input/initialization | Known unit conversions, actual charge/mass/count, measured distribution moments and cutoffs |
| Frame/external field | Position/momentum boost round trips, known field transformation and undulator entry/exit continuity |
| Deposition | Global deposited charge and discrete continuity residual for controlled trajectories, including rank boundaries |
| Field solver | Vacuum pulse propagation, dispersion/refinement, reconstruction consistency and boundary reflection |
| Parallel changes | Same small case on one and two ranks; compare particle count and reduced diagnostics with justified floating-point tolerance |
| Radiation diagnostic | Prescribed wave with known flux, frequency and frame, plus window/monitor/domain/refinement studies |
| Energy accounting | Particle plus field energy, external-field work, open-boundary flux and lost particles |

A small executable run is useful for checking wiring and finite outputs. It does
not validate radiation, conservation, startup fields or the undulator model.
Record compiler/backend, MPI count, input, timestep, domain, particle count and
measured error when making a physical or scaling claim.

## Suggested first student tasks {#fel_tasks}

1. Trace one particle from generation through two timesteps, drawing R_nm1/R and
   the field time levels. Explain why deposit precedes push.
2. Produce a table of input, internal and moving-frame mesh extents/spacings for
   the example; check the actual timestep condition and memory estimate.
3. Measure the generated bunch's moments and total charge. Explain the difference
   between the requested z-width and the measured rms length.
4. Add a small, analytic test of a z-boost round trip or a prescribed-wave power
   diagnostic before modifying its implementation.
5. For a ChDR extension, define a material/interface formulation consistent with
   the potential equations and reconstruction, then validate a planar interface
   before introducing a staircase prism. A scalar epsilon field alone does not
   establish the interface physics.

Review implementation limits before extending helpers: transform_EB() currently
constructs its boost direction along z despite the axis template, and the
undulator's endpoint/fringe formula has strict branch boundaries and phase
assumptions. The particle container also has unused getPL()/setPL() hooks with
pointer signatures inconsistent with its stored layout value. Their source
comments identify these dormant paths; the active manager does not call them.

## Maintain this documentation {#fel_docs}

From `demos/fel`:

```sh
doxygen Doxyfile
```

Or, from the repository root after configuring FEL with Doxygen available:

```sh
cmake --build build --target fel-docs
```

Open `docs/html/index.html` in your browser. Doxygen, Graphviz, LaTeX, dvips and
Ghostscript generate local pages, SVG diagrams and formula images, usable
offline. The build target is optional and not part of the default application
build. Generated files are ignored by Git.

The explicit Doxyfile input list deliberately excludes build products, data,
handoff notes and the rest of IPPL. It enables source browsing, topic groups and
strict warnings for incomplete documentation. Keep function contracts, unit/frame
conventions, guide and actual implementation aligned when changing the code.
To share a generated copy, archive the whole `docs/html` directory so that local
images, search assets and source pages remain together.
