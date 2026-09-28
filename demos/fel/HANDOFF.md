# FEL student documentation state

Goal (19 September 2026): review demos/fel source comments and generate focused,
local Doxygen documentation comparable to demos/chdr/src. This is a documentation
task: preserve executable C++ tokens, algorithms, defaults and numerical tolerances.

Scope: local C++ comments, Doxyfile, Documentation.dox, StudentGuide.md, README,
optional fel-docs CMake target and generated ignored HTML/offline archive.
Preserve all pre-existing repository changes outside demos/fel.

Review findings to explain: JSON input and working-directory-relative config
fallback; scaled internal units; nonstandard collocated four-potential solver;
moving-frame mesh and time conversion; rank-zero bunch generation and actual
particle count; deposition, three Boris substeps, migration and absorbing
boundaries; mixed-format diagnostic files and their interpretation limits.
Do not claim a dielectric model, charge-conserving deposition or validated
radiation normalization. No FEL-specific runtime tests are currently registered.

Parallel review: Config/units/datatypes; manager/field/particle containers;
MITHRA initialization/Lorentz transforms/undulator. Root owns the entry point,
guide, Doxygen configuration, build integration and final validation.

Validation planned: strict Doxygen generation; local HTML links/assets/formulas;
token comparison against HEAD; compile FEL and relevant existing checks; inspect
diff. Browser file preview is unavailable under this session's tool policy; use
filesystem validation and deliver local documentation links without a workaround.
Completed: documentation improvements in all ten local C++ source/header files;
student guide with ownership diagram, reading map, configuration/units, particle
shape, boosted grid/timestep, step ordering, distributed execution, outputs,
validation limits and exercises. Added Doxyfile, five topic groups, README
instructions, ignored docs directory and optional fel-docs target. Independent
source review corrected guide statements about ownership, pre_step and the
initial zero-displacement current deposit. No runtime implementation changed.

Validation so far: Doxygen 1.14.0 generated HTML twice with zero warnings;
67 pages, 5654 local links/assets, 67 formula images and all eleven guide anchors
pass filesystem checks. Ownership diagram rendered and inspected separately.
Offline docs/fel-student-docs.zip passes archive integrity check. Clang raw-token
comparison against HEAD passes for all ten C++ files (comments/whitespace only).
Scripts/logs are in /tmp/fel-doc-review; git diff --check passes.

Build complication: default `cmake` resolves to MacPorts, while build_openmp was
configured with /opt/homebrew/bin/cmake. Reconfiguration to enable FEL caused
FetchContent's Catalyst git-clone stamp to change and removed its old source
checkout before attempting to refetch. gitlab.kitware.com currently times out,
including a direct archive request outside the sandbox. Stopped our configure
process; existing libraries/binaries remain. Recover exact Catalyst v2.1.0 from
an available verified public cache before reconfiguring with Homebrew CMake and
an explicit local source override. No repository dependency code was changed.

Completed recovery: restored Catalyst v2.1.0 from the Spack source mirror with
SHA-256 verified against its libcatalyst package recipe:
1db07593c2f0203f53dfa39445d6f9d7a7fff78e0ce024b17af66ca7bce78abc.
The release source is in build_openmp/_deps/catalyst-src. nlohmann/json v3.11.3
also finished fetching. Local CMake cache now enables FEL and explicitly points
FETCHCONTENT_SOURCE_DIR_CATALYST and FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON at
their respective build_openmp/_deps source directories, avoiding another fetch.
Use /opt/homebrew/bin/cmake for this build tree. Repository dependency settings
remain unchanged. Initial fetch/configuration logs are retained in /tmp.

Final validation: FreeElectronLaser, mesh-1 and test-chdr-geometry compile with
LLVM21/Kokkos OpenMP; the optional fel-docs target succeeds with no warnings.
All seven existing ChDR CTest cases pass, covering geometry/configuration plus
serial/MPI2 mesh construction and output comparison for prism and brick.
Three-step FEL smoke runs on one and two ranks pass using 12x12x128 cells,
requested 1024 particles, shipped physical parameters and 100 ps input duration.
Both generate and retain 1176 particles; all four diagnostic rows are finite.
Printed output agrees except initial centroid roundoff (maximum difference
9.33e-15 internal lengths). Radiation outputs are zero over this short startup
window; this is an execution/MPI check, not radiation validation. Inputs and
logs are in /tmp/fel-doc-review/smoke. No tolerances or executable tokens changed.
The guide includes the checked smoke recipe and records physics limitations.

Status: documentation complete; final HTML/archive refreshed after review.
No outstanding task blocker. No full shipped run, GPU/scaling study, radiation
convergence or browser-page visual inspection was performed. The standalone
ownership diagram was rendered/inspected and local HTML references checked.

## Mathematical documentation extension (19 September 2026)

User request: explain the solved equations and their discretization, drawing on
MITHRA. Scope is documentation only. Plan: add an integrated mathematical page
with continuum potentials/gauge/units, exact IPPL nonstandard update and dispersion,
source deposition and time levels, reconstruction, Boris substeps, and Mur
boundaries; cite MITHRA 2.0 section/equation numbers and identify differences.
Local primary source: /Users/adelmann/git/mithra/manual/MITHRA_FDTDPIC/MITHRA_FDTDPIC.tex
(checkout 42fcef8ca0b15327845b86e2b2e2beec42191160); public version
https://arxiv.org/html/2009.13645. Parallel read-only reviews cover literature,
particle/deposition math and boundary formulas. Root derives the active stencil.
Do not copy the manual's apparent expanded-stencil source-sign/repeated-dx typos;
derive expressions from current code. Current field reconstruction differs from
MITHRA's temporal averaging/staggering, and exact discrete continuity is unproven.
Next: write formulas, independently verify their algebra, regenerate strict
Doxygen and check local assets/archive. No solver/runtime edits or new physical
validation claim; no need to rerun unchanged executable smoke checks.

Completed: MathematicalModel.md is integrated into Doxyfile, the main student
guide, README and the fields topic group. It contains SI and normalized potential
equations, Lorenz gauge and compatibility, cell/time notation, compact and expanded
15-point NSFD update, the implemented coefficient and source-free dispersion
analysis, CIC/chord-current sources, literal reconstruction time levels, three
Boris substeps, and exact compact Mur face/edge/corner equations. Includes primary
MITHRA links with equation numbers and equation-to-code map. Explicitly distinguishes
IPPL's reconstruction/deposition from MITHRA's description. No C++ files changed
in this extension.

Validation: independent math/source review passed. Reproducible scripts in
/tmp/fel-math-review compare compact versus literal coefficient expressions:
10,000 randomized NSFD, Fourier-symbol and axial-limit cases (max scaled error
1.54e-15); 10,000 each face/edge/corner Mur cases (max scaled error 2.49e-14).
These are algebraic transcription checks, not numerical convergence or boundary
reflection/MPI validation. Doxygen generated with zero warnings after replacing
unsupported amsmath shorthand; expanded stencil, Boris and Mur formula images
were visually inspected. Local HTML verification passes: 69 pages, 5798 local
references, all main/math section anchors and 142 formula images. Offline archive
was refreshed and integrity-checked. git diff --check passes. Existing executable
checks from the previous documentation pass remain applicable; none were rerun
because executable source and numerical behavior are unchanged. Status: complete.
