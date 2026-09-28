# ChDR mesh-1: distributed mesh and dielectric radiator

`mesh-1.cpp` builds a static dielectric prism or brick for geometry inspection and
later scale tests. It reuses FEL's `FELFieldContainer`, `UniformCartesian`,
`FieldLayout`, three-component E/B fields and four-component source/potential
fields. It allocates three potential levels and one scalar epsilon field:
**23 doubles per owned cell**, plus halos and MPI/diagnostic overhead.
No Maxwell update or particle initialization is performed.

## Student documentation

Start with [StudentGuide.md](StudentGuide.md) for the source reading order,
program flow, indexing/units, MPI data movement, test coverage, and suggested
first contributions. Public geometry/configuration APIs and the orchestration
functions have Doxygen contracts alongside their implementations.

Generate the focused HTML documentation from this directory:

```sh
doxygen Doxyfile
```

Open `docs/html/index.html`. Doxygen, Graphviz and LaTeX/Ghostscript formula
tools are needed to generate it; the resulting HTML is readable offline.
After CMake configuration, `cmake --build build_openmp --target chdr-docs` from
the repository root is an alternative. Documentation is optional for simulation
builds. Its explicit input list excludes generated data, handoff notes and the
rest of IPPL; generated `docs/` is ignored by Git. Warnings fail the documentation
build so broken references are visible.

To share an offline copy with a student, package the generated HTML and assets
from this directory:

```sh
python3 -m zipfile -c docs/chdr-student-docs.zip docs/html
```

After extracting the archive, open `html/index.html`. Include the entire HTML
directory so diagrams, equations, navigation and linked source pages remain available.

## Build and run

Enable the demo in an IPPL CMake build using the normal compiler, MPI and
Kokkos configuration for your machine:

```sh
cmake -S . -B build -DIPPL_ENABLE_CHDR=ON -DBUILD_TESTING=ON
cmake --build build --target mesh-1 test-chdr-geometry --parallel 4
ctest --test-dir build -R '^chdr\.mesh\.' --output-on-failure
```

The ChDR target needs Catalyst's Conduit parser, like FEL. CMake finds or
fetches it automatically. Enabling ChDR does not require enabling the full
FEL demo or Catalyst in-situ visualization. The standalone executable defaults
to the quick YAML copied beside it by CMake.

```sh
OMP_NUM_THREADS=2 build/demos/chdr/src/mesh-1
OMP_NUM_THREADS=2 mpiexec -n 2 build/demos/chdr/src/mesh-1 \
  demos/chdr/src/config.yaml --output /tmp/chdr-mesh-mpi2
~/.venv-h6/bin/python /tmp/chdr-mesh-mpi2/mesh-1_Materials.py --show
~/.venv-h6/bin/python /tmp/chdr-mesh-mpi2/mesh-1_Materials.py --save /tmp/chdr-materials.png
```

Replace `build` by your build directory (the local validation uses
`build_openmp`). `--help` lists the application options. `--output` overrides
the YAML path; relative output paths are relative to the working directory.

## Cases and units

- `config.yaml`: **40 x 40 x 64 cells**, 2.5 mm spacing, about **18.84 MB**
  owned field storage. A deliberately coarse inspection case.
- `config-scale.yaml`: **400 x 400 x 640 cells**, 0.25 mm spacing, about
  **18.84 GB** owned field storage. Use a suitable machine for allocation.
- `config-brick.yaml`: the quick mesh with an **axis-aligned brick** instead of
  the prism. Its 25 x 40 x 50 mm size matches the prism's bounding box.

Check the scale case before allocating it:

```sh
build/demos/chdr/src/mesh-1 demos/chdr/src/config-scale.yaml --dry-run
```

The prism cases use the previously discussed 100 x 100 x 160 mm box and the assumed
25 x 40 x 50 mm triangular prism bounding box. The triangle has 45-degree
faces and its analytic volume is 25,000 mm^3. These remain illustrative
geometry choices. The domain includes the earlier allowance for future
absorbing boundaries; the current program only constructs nonperiodic fields.
An absorber has not been implemented by specifying this box.

The YAML structure follows the shared-case discussion. `domain.lower_corner`
is a physical boundary; `domain.size` and `mesh.cells` determine spacing.
`mesh.decompose` selects the axes available for MPI decomposition (FEL's
z-only layout is the default). Grid-aligned faces and inclined faces use the
same analytic point-inside predicate. `mesh.material_sampling` must be
`cell_center_staircase`.

The prism's three `vertices` define its bottom triangular face. `axis` is a
unit extrusion vector and `height` its extrusion length, not the triangle's
depth. In the examples the vertices lie at y=-20 mm and extrusion ends at
y=+20 mm; the beam direction remains +z. An arbitrary oriented triangular
prism can be given by changing vertices and axis consistently. Geometry must
fit inside the configured box.

Select the rectangular radiator from the LaTeX setup figure with `type: brick`:

```yaml
geometry:
  - name: radiator_brick
    type: brick
    material: radiator
    lower_corner: [-25.0, -20.0, 0.0]
    size: [25.0, 40.0, 50.0]
```

Lengths follow `units.length`. This brick occupies x=-25..0 mm, y=-20..20 mm,
z=0..50 mm. Its beam-facing surface is x=0, consistent with a future beam
travelling along +z at x=a>0 in vacuum. The LaTeX figure is schematic and gives
no physical dimensions; these sizes remain configurable example assumptions.
The brick has six axis-aligned faces; rotation is not part of this input type.
Exactly one radiator is supported, with either `type: prism` or `type: brick`.
Type-specific keys are checked; a brick cannot silently accept prism vertices.

Run the prepared brick case separately:

```sh
OMP_NUM_THREADS=2 build/demos/chdr/src/mesh-1 \
  demos/chdr/src/config-brick.yaml --output demos/chdr/src/data/brick
```

`units.length` converts all input lengths to SI metres internally. The code
uses laboratory coordinates and does not apply FEL's bunch-frame boost.
Geometry output uses millimetres. E/B/current/potentials start at zero and
have no physical source or solution in this mesh inspection stage. Real
positive scalar permittivities and mu_r=1 are the initial supported material
model. The active nonstandard FDTD solver is never instantiated, so its
anisotropic-spacing restriction does not apply to this geometry program.
It must be checked when a solver is connected later.

FEL itself currently reads JSON. ChDR uses the same Catalyst Conduit dependency
with its YAML protocol, already exercised elsewhere in IPPL. This is a
ChDR geometry schema, not the old FEL job schema: unrelated bunch/undulator
settings are not accepted as silent inputs.

## Output and quick inspection

The C++ program writes:

- `mesh-1_Materials.py`: one self-contained script with the interactive PyVista
  view and the Matplotlib mesh report. Geometry, metadata and bounded sampled
  slices are embedded. No sibling files or IPPL installation are needed.
  PDF generation requires NumPy and Matplotlib; the interactive view and 3D
  screenshots additionally require PyVista.
- `mesh.json`: physical geometry, global grid, rank boxes, material cell count,
  analytic/voxelized dielectric volumes, allocation and construction diagnostics.
  Output schema version 2 uses a `radiator` object with `type: prism` or `brick`;
  input YAML remains schema version 1.
- `slices.csv`: actual epsilon samples from x-z, x-y and y-z planes through
  the radiator centroid, snapped to cell centres.

For the same workflow as OPALX's generated `ElementPositions.py`, run the
generated material script directly from your output directory. A local example
has been generated in `demos/chdr/src/data`:

```sh
cd ~/git/ippl-chdr/demos/chdr/src/data
~/.venv-h6/bin/python mesh-1_Materials.py --show
```

Each run first generates **`mesh-1_Materials.pdf` beside the script**, containing
the former `visualize_mesh.py` overview and three material slices. It then opens
the interactive window. Running without `--show` has the same behavior.
The PDF is refreshed on every run from the embedded simulation snapshot.
Drag to rotate, scroll to zoom,
and Shift-drag to pan. Checkboxes hide either material. The transparent blue
box marks the outer boundary of the vacuum region, and the orange solid shows
the dielectric radiator. Vacuum occupies the domain outside the radiator.
Coordinates are in millimetres. `--save materials.png` writes the PDF and renders
a 3D screenshot offscreen.

This follows OPALX's `MeshGenerator::write` pattern of embedding geometry in a
generated PyVista script. The implementation is inspired by
`/Users/adelmann/git/opalx/src/Structure/MeshGenerator.cpp`. Only analytic
material surfaces are drawn interactively. Embedded data size scales with the
bounded preview and rank count, not the full three-dimensional mesh.

To generate just the PDF without PyVista or a graphics display, or change its
destination:

```sh
~/.venv-h6/bin/python mesh-1_Materials.py --pdf-only
~/.venv-h6/bin/python mesh-1_Materials.py --pdf-only --pdf report.pdf
```

`visualize_mesh.py` is no longer generated or required. The combined script
also accepts `--dpi` and `--max-rank-boxes` for PDF formatting. The report shows
analytic radiator outlines over actual staircase slices for either geometry.
`output.preview_max_points_per_axis` caps the number of samples in each slice
axis. Every plotted material value comes from the allocated IPPL field; a
coarse preview does not recreate or average skipped voxels. The figure labels
decimated previews, and thin features may be missed. The 3D pane displays the
analytic radiator and MPI boxes, not every voxel. Large rank lists are thinned
only for display. There is no full 3D field gather or host mirror.

## Parallel behavior and checks

Rasterization uses a Kokkos kernel in the configured execution/memory space.
Cell coordinates are `lower + (globalIndex + 0.5)*h`, independent of MPI
partitioning. Owned cells are filled, epsilon halos exchanged, and all
in-domain owned/halo samples checked against the analytic predicate. External
nonperiodic ghost cells are not electromagnetic boundary conditions.
A rank's local dimensions must each have at least two cells for IPPL's halo
exchange; incompatible decompositions fail before field allocation.

Unsigned counts and a checksum over global occupied-cell IDs provide
partition-independent diagnostics. Reported allocation includes field halos;
its size can increase with rank count. Construction time is the maximum over
ranks and includes allocation, initialization, rasterization and halo checks;
preview generation and file output are outside that interval. Small validation
runs are not a performance benchmark.

Only bounded preview indices/values are transferred between device and host;
MPI reductions assemble slices on rank zero. Root-only output failures are
broadcast, and rank-local construction failures abort the communicator to
avoid leaving peers in collectives. No existing solver's conservation or time
integration is changed. No numerical tolerance in FEL is modified.

CTest covers geometry/configuration, one-rank construction and two-rank halo
exchange. `verify_mesh_output.py` can compare resulting metadata and slices,
including independent analytic checks of the shipped compact prism and brick,
and exact agreement between embedded viewer data and output diagnostics.
The geometry test also exercises a rotated prism and both device predicates.

Validated locally with LLVM 21, Kokkos 5.2 OpenMP and OpenMPI 5.0.8: all seven
CTest cases pass. Additional four-rank/all-axis and anisotropic-grid runs give
identical material counts, checksums and sampled fields to their single-rank
counterparts, with zero internal halo mismatches. The coarse example has
1,440 dielectric cells and 22,500 mm^3 voxel volume versus 25,000 mm^3 analytic
volume (10% lower); it is an inspection example, not a converged representation.
The brick example has 3,200 dielectric cells and 50,000 mm^3 voxel and analytic
volume. Equal volume does not imply exact face placement on the coarse grid.
The 102.4-million-cell case was validated by dry run only. GPU execution and
large-scale performance have not been measured.

On this Mac, OpenMPI needed explicit local slots and slot mapping. The local
CTest configuration uses `MPIEXEC_PREFLAGS` with
`--host localhost:4 --map-by slot --bind-to none --oversubscribe`. These are
local test-launcher settings, not defaults imposed on cluster jobs.

The new face-classification tolerance is 64 machine epsilons times the geometry
coordinate scale. It only resolves floating-point ambiguity at a face and does
not grow with mesh spacing. Boundary samples belong to the dielectric.

## Meep compatibility

The physical geometry schema retains Meep's polygon-plus-extrusion concepts.
A future Python adapter can read the same YAML, translate all positions to a
zero-centred computational cell and construct `mp.Prism`. Meep uses a scalar
pixels-per-length resolution; arbitrary anisotropic IPPL grids therefore need
an explicit Meep resolution choice. Staircase comparison also requires an
explicit subpixel-smoothing setting and attention to Yee versus collocated
sample positions. A Meep YAML adapter is not part of this implementation.

Sources: [Meep geometry interface](https://meep.readthedocs.io/en/latest/Python_User_Interface/#prism)
and [subpixel smoothing](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/).
