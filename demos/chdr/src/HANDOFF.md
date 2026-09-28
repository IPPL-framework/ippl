# mesh-1 implementation state

Goal (18 September 2026): construct a YAML-configured distributed FEL-layout
mesh and staircase prism, inspired by OPALX's generated Python beamline viewer.
Scope excludes field evolution and particle initialization.

Implemented: mesh-1.cpp reuses FELFieldContainer, its UniformCartesian and
FieldLayout, E/B/J aliases and three potential time levels, plus scalar epsilon.
23 doubles/cell preserve the earlier field-memory estimate. Internal coordinates
are lab-frame SI metres; output geometry uses mm. Rasterization and halo checks
run in Kokkos; only bounded sampled slices transfer to host/root. Separate quick
and scale YAML cases, dry-run resource estimate, metadata and Python viewer.

Files: ChdrMeshConfig.h, PrismGeometry.h, mesh-1.cpp, MeshVisualization.h,
config.yaml, config-scale.yaml, test_geometry.cpp, verify_mesh_output.py,
README.md and CMake integration (root option, Catalyst guard, demos/chdr target).
Existing FEL, OPALX and solver source files were not edited. Geometry accepts
either winding and arbitrary unit extrusion, scalar positive epsilon/mu=1.
Preview cap is 2..1024 points/axis. At least two local cells per axis are
required for IPPL halo exchange. Boundary tie tolerance is 64*double-epsilon
times geometry coordinate scale; no existing numerical tolerance changed.

Validation completed: Release build passes with LLVM21, external OPALX Kokkos
5.2 OpenMP, OpenMPI5.0.8 and fetched Catalyst2.1. All four CTest cases pass:
geometry/config/device predicate, serial mesh, MPI2 mesh and cross-rank output
comparison. Additional MPI4 with all decomposition axes enabled matches serial
counts/checksum/slices; anisotropic40x40x128 grid matches on1/2ranks. Zero
material/halo mismatches. Default40x40x64 gives1440 dielectric cells,22500mm^3
voxel volume versus25000 analytic (-10%, deliberately coarse). Scale config
400x400x640 dryrun reports102400000cells/18.8416GB owned field storage; no full
allocation. No GPU run or performance scaling claim.

Build location: build_openmp. Kokkos_DIR points to
/Users/adelmann/git/opalx/omp-build/_deps/ippl-build/_deps/kokkos-build.
Compiler /opt/homebrew/opt/llvm@21/bin/clang++; OpenMP libomp from Homebrew.
Initial compile lacked omp.h: fixed local CMAKE_CXX_FLAGS with
-I/opt/homebrew/opt/libomp/include. OpenMPI default core mapping failed in this
environment; local MPIEXEC_PREFLAGS now set --host localhost:4 --map-by slot
--bind-to none --oversubscribe. No production CMake launcher defaults changed.
Third-party Catalyst builds emit two existing unused-symbol warnings; no new
ChDR compiler warnings were found. Build/test logs are under build_openmp.

Generated viewer rendered with ~/.venv-h6 and inspected visually. Current
preview: build_openmp/demos/chdr/src/test-serial/mesh.png, alongside emitted
visualize_mesh.py, mesh.json and slices.csv. OPALX inspiration was
src/Structure/MeshGenerator.cpp::write, which emits ElementPositions.py.
No full 3D field gather is performed; preview data are bounded. The 3D pane
shows analytic geometry; slice panels contain actual material samples.

Final review: source/diff and independent subagent review; git diff --check.
Remaining limits are documented: material/mesh construction only, no field
evolution or particles, no Meep YAML adapter, no absorber, no large allocation.

## OPALX-style material viewer (18 September 2026)

User clarification: generate a script like OPALX's ElementPositions.py, runnable
directly in the data directory, to interactively inspect only the two materials.
Completed in MaterialViewer.h, called by mesh-1.cpp on the output rank. Each run
now emits mesh-1_Materials.py with all geometry embedded in millimetres. It uses
NumPy/PyVista, opens interactively by default (also accepts --show), supports
--save for an offscreen PNG, trackball navigation and two visibility checkboxes.
The transparent blue surface marks the vacuum region's outer domain boundary;
the orange prism is dielectric. Vacuum excludes the prism. These are analytic
surfaces, not voxels; the existing visualize_mesh.py still shows actual field
slices. Output size and render cost are independent of simulation cell count.

Changed: added MaterialViewer.h and local .gitignore for generated data/cache;
updated mesh-1.cpp, README.md and this state file. No physics, material sampling,
parallel kernels, tolerances or OPALX sources changed. Prepared user output in
demos/chdr/src/data/mesh-1_Materials.py, with materials.png as a preview.

Validation: rebuilt mesh-1 and all four existing CTest cases pass again
(geometry, serial, MPI2 and output comparison). Generated script works as an
isolated copy in a temporary directory without metadata. Independent surface
checks pass for original/reversed winding and arbitrary rotated prisms: closed
manifold surfaces, outward normals, expected bounds, 25,000 mm^3 prism volume
and 1,600,000 mm^3 domain volume. Initial VTK WEDGE winding was corrected so its
base normal points opposite extrusion. Both actor visibility controls passed;
offscreen PNG rendered and visually inspected after moving the orientation
widget away from material labels. Local PyVista version is 0.48.4. macOS VTK
rendering aborted inside the restricted sandbox; the same render succeeded
with approved macOS graphics access. git diff --check and clang-format checks
pass. No remaining viewer blocker; next step is user geometry inspection.

## Combined viewer with automatic PDF and brick (completed, 18 September 2026)

User request: integrate visualize_mesh.py into mesh-1_Materials.py and generate
its overview/slice content automatically as PDF. Implementation now embeds
mesh.json metadata and bounded slices.csv data in the single emitted viewer.
MeshVisualization.h emits reusable Matplotlib report functions; MaterialViewer.h
calls them before the existing interactive PyVista scene. Default PDF is
mesh-1_Materials.pdf beside the script; --pdf overrides it and --pdf-only skips
PyVista/display. --save still writes a 3D screenshot as well as the PDF.
mesh-1.cpp no longer emits a separate visualize_mesh.py. No sampling, fields,
parallel kernels or physics changed. Next: build, test serial/MPI and standalone
PDF generation, render/inspect PDF, verify combined screenshot, update README.

Steering: also cover an axis-aligned brick like LaTeX Figure 1. The figure in
ideas-lit-reseach/chdr_literature_review.tex:82 has x<0 dielectric, face x=0,
beam +z at x=a>0; no physical dimensions. Added config-brick.yaml using the
existing prism bounding box: lower[-25,-20,0] mm, size[25,40,50] mm. These are
example assumptions. ChdrMeshConfig.h now supports strict type: prism|brick;
BrickGeometry.h uses six inequalities with the existing 64-epsilon boundary
policy. mesh-1 templates the existing construction over the geometry predicate,
keeping Kokkos/device sampling and bounded data movement unchanged. Metadata
output schema 2 uses tagged radiator object; input YAML still version1. Viewer
and PDF support both geometries. verify_mesh_output.py checks independent brick
intervals/counts plus exact embedded diagnostic data; added brick serial/MPI2/
compare CTest cases. Current build mesh-1 passed; geometry tests build ongoing.

Final validation: mesh-1 and test-chdr-geometry builds passed, and all seven
CTest cases pass, including serial/MPI2/cross-rank comparison for each shape.
Prism remains 1,440 occupied cells; brick is 3,200, with analytic/voxel volume
50,000 mm^3 (equal volume does not prove exact coarse-grid face placement).
Combined scripts were copied individually to empty temporary directories and
ran --pdf-only from /tmp without metadata files, producing valid PDFs. Both
material surfaces are closed with correct volumes. Combined --save mode
generated PDFs and PyVista PNGs for both cases with macOS graphics access.
PDFs were rendered with Poppler and visually inspected; both contain the
overview, x-z/x-y/y-z actual samples, analytic interfaces and diagnostics.
All embedded sample/metadata checks passed in serial and MPI2 outputs.
Formatting and whitespace checks pass. Independent integration review found
no functional issue. README documents geometry selection, automatic report,
headless mode, new metadata schema and limitations. No GPU/scale run added.

Prepared outputs (ignored data directory): prism data/mesh-1_Materials.py and
data/mesh-1_Materials.pdf; brick data/brick/mesh-1_Materials.py and corresponding
PDF. Running either script normally first refreshes its PDF, then opens the
interactive material window. No task blocker remains.

## Student documentation (completed, 19 September 2026)

Goal: review the local source documentation, add a src-local Doxyfile and
student guide, and generate focused HTML docs. Scope is docs/comments plus an
optional documentation build target; no physics/numerical behavior changes.
Plan: document geometry/configuration contracts, distributed execution and
generated viewer lifecycle; explain unit/integration tests and their limits;
add architecture diagram, first-run walkthrough and student exercises. Exclude
generated data and the rest of IPPL from Doxygen input. Installed tools found:
Doxygen 1.14.0 (/opt/local/bin/doxygen), Graphviz (/opt/homebrew/bin/dot).
Agents document geometry/config headers and test sources independently; root
owns main/viewer comments, guide, Doxyfile, README and generation/validation.
Next: build HTML with clean diagnostics, check links/layout, rebuild and run
existing seven tests, review changes for documentation-only intent.

Completed: added src-local Doxyfile and Documentation.dox topic groups;
StudentGuide.md provides first-run commands, architecture diagram, linked
source map, geometric/discrete conventions, global/halo indexing, MPI and
memory accounting, output lifecycle, test limits, exercises and extension
checklist. Updated all local C++ source/header documentation and Python checker
comments/docstrings; README explains the docs build. Added optional chdr-docs
CMake target and /docs/ ignore rule. Runtime algorithms, data schemas, numerical
tolerances and generated-viewer behavior were not changed.

Validation: executable C++ token checks for agent-owned headers/tests and Python
AST comparison excluding docstrings confirm unchanged code. Both executables
rebuilt and all seven existing tests pass (including both geometries, serial and
MPI2). Doxygen 1.14.0 generates with zero warnings using local formula images;
the chdr-docs target also succeeds. Checked 52 generated HTML pages, 2,494 local
links/assets and 28 formula images; all referenced files/anchors exist. Independent
source/guide review corrected precision of halo indices, valid axis reversal and
absolute/relative output paths. Formatting and whitespace checks pass.

Browser visual preview of file:// HTML was blocked by the browser security
policy. No workaround was attempted; validation used local HTML parsing and
asset checks. Docs are at docs/html/index.html; docs/chdr-student-docs.zip is an
offline bundle for sharing with a student. Generated docs are ignored by Git.
The local generation requires Doxygen, Graphviz and latex/dvips/Ghostscript;
reading the resulting HTML needs no server or external formula CDN. No blocker
remains for the requested documentation; browser visual inspection was not done.
