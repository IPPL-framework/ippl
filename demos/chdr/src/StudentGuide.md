# Working on the ChDR mesh code

@tableofcontents

[First run](#student_first_run) · [Source map](#student_source_map) ·
[Geometry](#student_geometry) · [MPI](#student_parallel) ·
[Output](#student_output) · [Tests](#student_tests) ·
[Exercises](#student_exercises) · [Troubleshooting](#student_troubleshooting)

## What this program does {#student_scope}

`mesh-1` constructs a three-dimensional Cartesian IPPL mesh containing vacuum
and one dielectric radiator. The radiator can be an oriented triangular prism
or an axis-aligned brick. Each cell receives one relative permittivity value.
The program verifies its material field and MPI halos, then writes diagnostics
and a self-contained Python viewer.

This is the **geometry and mesh stage** of the Cherenkov diffraction radiation
project. E, B, current and potential fields are allocated but remain zero.
There are no particles, Maxwell time steps, dielectric interface equations,
or electromagnetic absorbing boundaries in this executable. The existing FEL
field types provide the storage layout; they do not make the new material map
an electromagnetic solver by themselves.

All internal lengths are laboratory-frame **metres**. Output geometry and
plots use **millimetres**. The planned beam direction is **+z**; the beam itself
is not represented yet. In the shipped examples the beam-facing surface is
x=0, with dielectric at x<0 and a possible future beam at x=a>0 in vacuum.

## First successful run {#student_first_run}

Prerequisites are an IPPL build with a working C++ compiler, MPI, Kokkos and
Catalyst's Conduit parser. The parser is also used by FEL. Here `build_openmp`
means the existing local build; replace it with your configured build directory.
Run these commands from the **repository root**:

```sh
cmake --build build_openmp --target mesh-1 test-chdr-geometry --parallel 4
ctest --test-dir build_openmp -R '^chdr\.mesh\.' --output-on-failure
OMP_NUM_THREADS=2 build_openmp/demos/chdr/src/mesh-1 \
  demos/chdr/src/config-brick.yaml --output /tmp/chdr-student-brick
python3 /tmp/chdr-student-brick/mesh-1_Materials.py --pdf-only
```

For a new build, enable `IPPL_ENABLE_CHDR=ON` and `BUILD_TESTING=ON` together
with the normal IPPL toolchain options for your machine. Enabling the FEL
executable or Catalyst in-situ visualization is not required. The local
`README.md` contains the build and MPI launcher details.

The generated script needs NumPy and Matplotlib for the PDF. Add PyVista for
the interactive view; on the current workstation these are installed in
`~/.venv-h6/bin/python`, which can replace `python3` above. Run the script
without `--pdf-only` to write the PDF and open the material window.

The shipped quick cases use 40 x 40 x 64 cells, with 2.5 mm spacing:

| Case | Radiator | Occupied cells | Analytic volume | Sampled cell volume |
| :--- | :--- | ---: | ---: | ---: |
| `config.yaml` | triangular prism | 1,440 | 25,000 mm³ | 22,500 mm³ |
| `config-brick.yaml` | brick | 3,200 | 50,000 mm³ | 50,000 mm³ |

These are expected checks for the **shipped configurations**. A change to
geometry, origin or resolution can change the counts. Equal brick volumes do
not imply exact face placement; even flat faces can lie between cell edges.
The prism's 10% coarse-grid volume discrepancy is intentional and illustrates
the need for a resolution study.

`config-scale.yaml` is a resource-planning case with 400 x 400 x 640 cells.
Start with `--dry-run`: its owned fields alone require about 18.84 GB. Full
allocation, GPU execution and large-scale performance have not been validated.

## Read the source in this order {#student_source_map}

| File | Responsibility | Useful starting point |
| :--- | :--- | :--- |
| @ref ChdrMeshConfig.h | YAML schema, units and validation | @ref chdr::MeshConfig, @ref chdr::readMeshConfig |
| @ref PrismGeometry.h | Five-face prism predicate | @ref chdr::PrismGeometry |
| @ref BrickGeometry.h | Six-face axis-aligned box predicate | @ref chdr::BrickGeometry |
| @ref mesh-1.cpp | Allocation, parallel sampling, halos and output | `main`, `construct`, `previewSamples`, `writeOutput` |
| @ref MaterialViewer.h | Embed diagnostics and emit the Python entry point | @ref chdr::writeMaterialViewer |
| @ref MeshVisualization.h | Emit the Matplotlib PDF helpers | @ref chdr::writeMeshReportFunctions |
| @ref test_geometry.cpp | Unit checks for geometry and configuration | `geometryChecks`, `brickChecks`, `configChecks` |
| @ref verify_mesh_output.py | Validate output and compare MPI runs | `readAndValidate`, `checkEmbeddedDiagnostics` |

The external types come from `demos/fel/FELFieldContainer.hpp` and
`demos/fel/datatypes.h`. Their implementation and the rest of IPPL are excluded
from this focused documentation. Use the generated **Files** and **Topics**
navigation to move between the local source and its API contracts.

@dot
digraph chdr_flow {
  graph [rankdir=TB, bgcolor="transparent"];
  node [shape=box, style="rounded,filled", fillcolor="#edf4f8", color="#517487", fontname="Helvetica"];
  edge [color="#517487", fontname="Helvetica", fontsize=10];
  input [label="YAML -> MeshConfig\nvalidate and convert to SI"];
  geometry [label="PrismGeometry or BrickGeometry\ncontains(x,y,z), volume()"];
  fields [label="construct<Geometry>\nFEL fields + epsilon_r, Kokkos sampling"];
  halos [label="halo exchange + checks\nMPI counts, checksum, bounded slices"];
  output [label="rank-zero output\nmesh.json + slices.csv + Python script"];
  viewer [label="run generated Python\nPDF + optional interactive materials"];
  input -> geometry -> fields -> halos -> output -> viewer;
}
@enddot

Both geometry classes satisfy the same **compile-time interface**. There is
no inheritance or virtual dispatch. `main` chooses the class from the YAML
type, and the templated `construct` uses its `contains` predicate in kernels.

## Geometry and discretization {#student_geometry}

The domain is specified by a physical lower corner L, size S and integer cell
counts N. For each axis d:

```text
h[d] = S[d] / N[d]
cell centre[d] = L[d] + (globalIndex[d] + 0.5) * h[d]
epsilon_r(cell) = radiator epsilon if contains(centre), otherwise background epsilon
```

The mesh spacing is uniform within each axis. Spacings may differ between axes
in this geometry program; a future field solver may impose further restrictions.
Material values are sampled once at cell centres. There is no fractional filling,
subpixel averaging, adaptive refinement, or smoothing at inclined faces.

The prism's three vertices define one triangular face. A unit `axis` normal
to that face and a positive `height` define the extrusion. Either triangle
winding is accepted. A changed `axis` must remain perpendicular to the face;
otherwise rotate the face consistently as well. Reversing the axis is valid if
the resulting extrusion still fits inside the domain. The shipped prism has
45-degree faces; it is not a measured
47-degree experimental radiator.

A brick uses `lower_corner` and three positive edge lengths `size`:

```yaml
geometry:
  - name: radiator_brick
    type: brick
    material: radiator
    lower_corner: [-25.0, -20.0, 0.0]
    size: [25.0, 40.0, 50.0]
```

Both shapes include points on their boundaries within a roundoff tolerance
of 64 machine epsilons times the geometry coordinate scale. This tolerance
handles floating-point ambiguity; it is unrelated to the cell width or a
physical transition layer. The configuration reader validates the shape and
requires it to fit inside the domain before allocating fields.

Input YAML schema **1** and diagnostic JSON schema **2** are different interfaces.
The latter uses a tagged `radiator` object. Do not change the YAML schema number
to match the output file. Material values are finite positive scalar epsilon_r
and mu_r=1; dispersion, losses, tensor permittivity and multiple radiators are
outside the present implementation.

## Follow one distributed cell {#student_parallel}

`FieldLayout` partitions the global mesh. On each rank, the IPPL field view
includes halo cells around its owned block. A kernel view index i converts to
a global cell index through:

```text
globalIndex = viewIndex + firstOwnedGlobalIndex - numberOfGhostCells
```

An interior halo cell belongs to a different rank; an exterior ghost maps
outside the global domain and is skipped by the material halo check.

Using this global index is essential: the same physical point must get the
same epsilon_r regardless of which rank owns it. The geometry object contains
only fixed-size numeric state and is copied by value into Kokkos kernels.
It contains no host pointers, strings or dynamic containers.

The sequence in `construct` is:

1. Construct FEL's uniform mesh and field layout; require at least two owned
   cells per local axis for IPPL halo exchange.
2. Allocate and zero E, B, J and three potential levels; allocate epsilon_r.
3. Fill owned material cells in a Kokkos reduction, accumulating occupied count
   and an unsigned checksum of global occupied-cell IDs.
4. Exchange epsilon_r halos and check owned/in-domain halo values against the
   same geometric predicate. Exterior nonperiodic ghosts do not define an
   electromagnetic boundary condition.
5. Reduce diagnostics. Build identical bounded preview-index lists on all ranks.
6. Sample the actual field on each owning rank. Nonowners contribute zero;
   MPI_SUM assembles each sample on rank zero.
7. Write files on rank zero and broadcast an output failure to all ranks.

Only bounded slice arrays are copied from device to host. The program never
gathers a full 3D material field. The per-plane bound is the configured preview
cap squared; MPI rank-box metadata adds storage proportional to the rank count.

The owned-field estimate is 23 doubles per cell: E(3), B(3), J(4), three
four-potential levels(12) and epsilon_r(1). Reported allocated bytes also include
field halos and actual vector sizes; they exclude general process overhead and
preview buffers. Construction time is the maximum over ranks through the field
checks and allocation accounting, excluding preview generation and file I/O.

Input/output errors are coordinated between ranks. Unexpected rank-local
allocation or kernel failures reach an MPI abort rather than leaving peers
waiting indefinitely at a collective. Keep collective call order identical
across ranks when adding code.

## Understand the generated output {#student_output}

`writeOutput` closes `mesh.json` and `slices.csv` before calling the viewer
emitter. The generated `mesh-1_Materials.py` embeds both snapshots. It remains
usable after copying just that file to another directory.

The C++ program writes the script; **running the script creates the PDF**.
By default the PDF is placed beside the script and then PyVista opens the 3D
materials. `--pdf-only` needs no graphics display or PyVista. `--save image.png`
writes the PDF plus a 3D screenshot, and `--pdf report.pdf` changes the PDF path.

| View | What is shown | What it can establish |
| :--- | :--- | :--- |
| Interactive 3D | Analytic radiator and transparent outer vacuum boundary | Position, shape, orientation and domain fit |
| PDF overview | Analytic radiator, domain and MPI rank boxes | Geometric layout and decomposition overview |
| PDF slices | Actual epsilon_r samples with analytic interface outlines | Cell-centre material assignment and visible staircasing |

Slice planes pass through the cells containing the analytic centroid. They are
snapped to cell centres; reported plane coordinates may differ from the centroid.
If a slice is decimated, the display bins between samples are not additional
simulated cells. Thin features may be missed. Increase
`output.preview_max_points_per_axis` when needed, within its 2..1024 range.

Edit @ref MaterialViewer.h or @ref MeshVisualization.h to change future generated
viewers, then rebuild and rerun `mesh-1`. An edit to a generated script changes
only that snapshot and will be overwritten when the same output is regenerated.
The emitted Python helpers are:

| Helper | Responsibility |
| :--- | :--- |
| `materialSurfaces()` | Closed PyVista analytic radiator and outer-domain surfaces |
| `buildScene()` | Actors, camera and material visibility controls |
| `radiatorMesh()` / `analyticCut()` | Analytic polygons and intersections for the PDF |
| `saveMeshReport()` | Four-panel report from embedded metadata and actual samples |
| `main()` | CLI, automatic PDF, then optional 3D rendering |

## Tests and what passing means {#student_tests}

`test_geometry.cpp` is the unit-test executable. Known points check interior,
exterior and boundary classification, volumes, prism winding and one rotated
prism. Additional cases check units and invalid YAML. The Kokkos checks execute
on the configured backend; the validated local backend is OpenMP CPU.

The integration tests run the real `mesh-1` executable with one and two MPI
ranks. `verify_mesh_output.py` reads each output and compares the runs. It uses
only the Python standard library, so the test suite does not require plotting
packages. Its independent geometric reference covers the shipped compact prism
and axis-aligned bricks; arbitrary prisms currently receive consistency and
cross-run checks without an independent classification reference.

The compact prism's independent total count is checked for the shipped
40 x 40 x 64 grid. Brick counts use factorized interval counts for any resolution.
The Python checker checks exported slices plus global count/checksum; it does
not independently inspect every 3D field value. The executable's internal halo
check reuses its production predicate. The viewer integrity check parses embedded
data without executing the viewer, so rendering is separate manual QA.

| CTest names | Purpose |
| :--- | :--- |
| `chdr.mesh.geometry` | Geometry and configuration unit checks |
| `chdr.mesh.serial`, `chdr.mesh.brick.serial` | One-rank construction |
| `chdr.mesh.mpi2`, `chdr.mesh.brick.mpi2` | Two-rank construction and halo checks |
| `chdr.mesh.compare`, `chdr.mesh.brick.compare` | Independent output checks and serial/MPI agreement |

The comparison tests use CTest fixtures, so CTest first prepares the required
outputs. A changed test tolerance needs numerical justification. Passing this
suite establishes the implemented geometry workflow; it does not establish
mesh convergence, radiation accuracy or energy conservation in a future solver.

## Suggested first contributions {#student_exercises}

These are exercises for the student, not implemented features.

1. **Resolution study.** Copy the quick prism YAML, double the cells along each
   axis, and compare sampled volume and slices. Predict the eightfold growth in
   owned field storage. Record results in a table and plot error against spacing.
   Staircase errors can change non-monotonically as faces cross cell centres.
2. **MPI invariance.** Run the same small case on one, two and four ranks, then
   pass all output directories to `verify_mesh_output.py`. Change only the enabled
   decomposition axes and repeat, retaining at least two local cells per axis.
3. **Geometry validation.** Add a consistently rotated prism case and an
   independent reference for its slice intersections before generalizing shapes.
   Include points on faces and just outside them, plus a Kokkos classification test.

For a new radiator type, update configuration validation, a device-copyable
geometry predicate, centroid selection and dispatch in `mesh-1.cpp`, JSON
serialization, both visual representations, independent output checks, examples
and tests. Search for `GeometryType`, `radiator["type"]` and `geometryType` to find
the current selection points. Preserve SI units, explicit boundaries, and
MPI-invariant global indexing.

Connecting a Maxwell solver is a separate physics task: define the spatially
varying material operator and interface conditions, compatible source deposition,
boundary conditions and validation cases. Simply allocating epsilon_r does not
apply those equations. Begin with the planar reference before interpreting
radiation from a rotated radiator.

## Troubleshooting and rebuilding these docs {#student_troubleshooting}

| Symptom | First check |
| :--- | :--- |
| Rejected YAML | Read the reported key path; check units, shape-specific keys and domain containment |
| Too few local cells | Reduce MPI ranks or change decomposition axes |
| OpenMPI slot/mapping failure | Check launcher configuration; see the local README's workstation example |
| PDF contains no feature | Check analytic position and whether preview decimation skips a thin feature |
| Correct-looking 3D shape, unexpected slice | The 3D shape is analytic; the slice is the sampled field |
| MPI comparison fails | Compare global cell indices, origin, halo offsets, checksum and sample ownership |
| Edited viewer changes disappear | Change its C++ emitter rather than the generated snapshot |

From `demos/chdr/src`, run:

```sh
doxygen Doxyfile
```

Open `docs/html/index.html`. The inputs are explicitly restricted to these local
sources and this guide. Generated output, build trees and the larger IPPL code
are excluded. Doxygen warnings fail the generation command. The optional CMake
target `chdr-docs` runs the same command without building the simulation.

The documentation build uses Doxygen, Graphviz `dot`, and LaTeX tools
(`latex`, `dvips`, Ghostscript) for local formula images. The generated HTML,
diagrams and formulas can then be read offline. Generated `docs/` is ignored by
Git; commit the comments, guide and Doxyfile rather than the HTML.
