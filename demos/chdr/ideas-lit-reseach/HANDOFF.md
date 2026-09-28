# ChDR literature and coordinate review

## Citations restored to user-edited implementation plan, 28 September 2026

User edited chdr_implementation_plan.tex and asked to add earlier citations.
Read their current source directly; preserved A Adelmann & Hugo, their initial
E/B source-free test sentence and their other wording/line references. Added
13 cited works using natbib/unsrtnat and standard BibTeX (TeXShop pdflatexmk).
Reused chdr_references.bib; appended only three missing entries: Chew 2014,
Ryu et al. 2016 (DOI checked against Crossref), and Fallahi/MITHRA 2.0 2020.
Short attribution notes distinguish potential/staggered literature and
charge-conserving constructions from the unvalidated IPPL collocated operator.
The local Meep numerical results remain attributed to VALIDATION.md.

Compiled in tmp/pdfs/implementation-plan and again beside the source for
TeXShop with SyncTeX. Both PDFs have identical extracted content: 5 pages,
13 references, no LaTeX/BibTeX warnings, missing citations or box overflows.
All five pages visually reviewed. Updated output/pdf/chdr_implementation_plan.pdf
and appended build guidance to README_latex.md. Explicit reversible-edit audit
proves all pre-existing document text and all pre-existing BibTeX content
were preserved. Backups and edit manifest are under the location recorded in
/tmp/chdr-plan-citations-backup-path.txt. No C++ or numerical tests changed.

## Standalone LaTeX implementation plan, 28 September 2026

User requested LaTeX for the implementation plan. Created editable,
self-contained chdr_implementation_plan.tex (pdfLaTeX / TeXShop) and compiled
output/pdf/chdr_implementation_plan.pdf (4 pages). Preserved all three work
groups, 26 explicit source references and 11 secondary line references; converted
the long test table to subsections and rendered mathematical notation in LaTeX.
Original Markdown, HTML, literature review and user-edited TeX files unchanged.
Independent content audit passed. latexmk -pdf -interaction=nonstopmode
-halt-on-error -outdir=tmp/pdfs/implementation-plan chdr_implementation_plan.tex
succeeded without box/font/reference warnings. Rendered and visually inspected
all four pages; no clipping or overlap. No C++ or physics tests run (documentation
only). Temporary QA files are isolated in tmp/pdfs/implementation-plan.
Next step, if requested, is the material-aware solver implementation.

## Source-referenced solver and test details, 28 September 2026

User requested more detail and line numbers for work-plan sections 1 and 2.
Re-read the current source and expanded chdr_implementation_checklist.md with
verified implementation sites, existing versus required test coverage and
observable-based acceptance tasks. Retained exactly three top-level work groups;
section 3 is unchanged. Geometry/material-field construction remains marked done.
An independent read-only test audit confirmed COMPILE_ONLY Maxwell examples and
existing runtime geometry/MPI checks. No C++ or numerical results changed; no
physics tests rerun for this documentation edit. Next step remains material-aware
solver integration. Final review: all 26 source links resolve to nonblank lines; exactly three
work groups retained and section 3 verified byte-for-byte unchanged.

## Consolidated ChDR work plan, 28 September 2026

User corrected that the material field and geometry already exist in mesh-1.cpp
and requested three groups: solver extension, tests, full particle simulations.
Rewrote chdr_implementation_checklist.md accordingly. Material-field creation is
complete; integration into a material-aware potential solver remains. Retained
validation limits, distributed 64-bit initialization for 1:1 bunches and the
later collision-inclusive evolution requirement. No C++ or other reports changed.
Reviewed the documentation changes; numerical tests are not applicable to this
revision. Next implementation step is the material-aware solver integration.

## IPPL/FEL ChDR implementation checklist, 28 September 2026

User asked for a list of required changes, not implementation. Re-audited
current core/FEL/ChDR sources with independent core and application audits.
Created chdr_implementation_checklist.md: retain four-potentials and uniform
MPI/Kokkos mesh; material operator/gauge/interface, consistent deposition and
E/B reconstruction, timestep, initialization/absorbers, lab-frame manager,
geometry/YAML/unit integration, beam initialization, diagnostics, validation.
Mesh-1 already has staircase brick/prism+epsilon and viewer, no field evolution.
Meep/reference agreement applies only to tested planar Fourier modes/bands.
New actionable detail: 1 nC is6.24e9 electrons, exceeding MithraBunch unsigned
int count; 1:1 runs need distributed64-bit initialization (currently rank0).
Other FEL defaults to remove in ChDR mode: automatic boost (evenK=0), beam
recentering,1% modulation, uniform longitudinal sampling, undulator bands.
Current FEL input is JSON; CHDR mesh input is YAML. Initial geometry is SI,
FEL runtime normalized: explicit conversion boundary required.
Collision-inclusive transport/P3M is a later separate requirement for the
scientific collision objective, not a prerequisite for prescribed-source ChDR.
No C++, existing reports or webpage text changed; no new numerical tests run.

## AWA charge range update, 24 September 2026

User requested a quick literature check and update of the webpage text.
Verified drive-beam envelope 0.1-100 nC from Neveu IPAC2018 THPMF048/049;
common operating points then were 1,4,10,40 nC, with measured CTR cases
0.3,0.7,1.3,30 nC. Lu LCWS2025 slide7 corroborates a65 MeV drive line,
100 nC single-bunch maximum and600 nC train maximum. Frame IPAC2022
WEPOPT065 describes <=1 nC and eight50 nC high-charge operation. Train
total and witness-gun limits were excluded from the single-bunch statement.
Created awa_charge_range_review.md with primary links and interpretation.
Updated awa_research_summary.txt: proposed initial0.1-10 nC scan near60MeV,
1 nC /1 mm rms reference retained; later10-100 nC exploration conditional
on delivered beam quality, transmission, detector response and clearance.
The facility envelope is not a guarantee of those parameters simultaneously.
Three plain-text sections and dissemination paragraph preserved. No model,
LaTeX, PDF or baseline numerical parameter files changed.

## AWA webpage research summary, 24 September 2026

Read the user-supplied funded ResearchPlan.pdf (21 pages), particularly pp1,
7-15, with complete text extraction and visual inspection of p12. It defines
WP1 macroscopic perturbations, WP2 microscopic/collisional transition and WP3
noise control/self-organization (stretch goal). The plan specifies TR/OTR;
ChDR is the current diagnostic extension, not an original hardware commitment.
Use the user's updated statement that the four-year grant supports two PhD
students and one postdoc, rather than the older PDF's two-year postdoc duration.

Created awa_research_summary.txt with the requested three plain-text sections.
Current run is diagnostic commissioning and macroscopic validation supporting
WP1 and preparing WP2, not a promised observation of microscopic correlations
or strong coupling at the 60 MeV/1 nC/1 mm Gaussian reference point.
Proposed first-run targets (not verified installed capabilities): 20-120 GHz,
SNR >=5 in >=3 channels, relative spectral calibration <=20%, >=100 accepted
pulses per point, scans with >=3 charges and gaps and >=2 focusing settings,
independent bunch-length comparison within20% at3 points, and a calibrated95%
confidence sensitivity estimate. Scans need not form a complete Cartesian grid.
Later phase compares joint radiation/energy-spread/emittance data against
collisionless and collision-inclusive models. Optical feasibility and keV slice
resolution are not promised for this first run. Independent audit checked
phase mapping and quantitative targets. Main LaTeX/PDF and author edits retained.

## Prat energy-spread discussion, 24 September 2026

User requested discussion of Sections 4.3/4.4 with reference [3] energy spread;
LaTeX, PDF and packaged sources are unchanged. Re-read Prat 2022 Sections II
and IV.B: 5.9 +/- 0.1 keV rms at 19 A with zero R56, 4.2 +/- 0.2 keV with
weaker focusing, and 14.2 +/- 0.3 keV with nominal dispersion. The latter
includes unresolved microbunching and cannot automatically be called thermal.
These are conditional SAME ABSOLUTE spread inputs for our 60 MeV, 1 nC,
Gaussian rms xyz=1 mm beam, not predictions obtained by transferring SwissFEL
beamline dynamics. Longitudinal variance conversion:
kBT_parallel' = sigma_E^2/(gamma^2 beta^2 m_e c^2). Scalar Debye/HNC results
additionally assume isotropy and local equilibrium at that temperature.

For sigma_E = 4.2, 5.9, 14.2 keV respectively:
kBT_parallel' = 2.462, 4.858, 28.142 meV;
Gamma = 0.1410, 0.07147, 0.01234;
lambda_D' = 6.376, 8.957, 21.557 um; ND = 3.634, 10.07, 140.44.
ND at the lower spreads is not asymptotically large. Frozen-pattern vacuum
crossover wavelengths = 0.338, 0.475, 1.144 um, conditional broad screening
crossovers rather than predicted lines. At these wavelengths, the best-coupled
vacuum field decay length equals lambda_D', so micron-scale clearance remains
necessary. Density spacing and convected 1/omega_p' = 10.88 m are unchanged.

Replace cold initial DIH energy balance in any future revision with
3/2 kB(Tf'-Ti') = u_corr,i - u_corr,f. Cold endpoint 0.1558 meV corresponds
to only 1.0565 keV rms lab spread under isotropy; it is not a universal extra
variance to add to an already warm beam. For weak coupling, initially random
positions and fixed density, Debye-Huckel gives leading fractional heating
DeltaT/T = Gamma^(3/2)/sqrt(3), about 1.1% at 5.9 keV and 0.079% at 14.2 keV.
Do not extrapolate the strong-coupling OCP energy fit to these warm cases.
IBS redistribution in anisotropic beams remains possible; transverse random
velocity distribution and density/focusing history are still needed.

Calculated reproducibly in /tmp/chdr_energy_spread_screening.py (CSV alongside).
Checked screening identities, relativistic velocity derivative, spread scaling;
independent scientific audit agreed. Next step only if requested: incorporate
the finite-spread scenarios into the report, preserving author edits.

## Diagnostic-regime table, 24 September 2026

Retrieved the earlier three-row table following "For the diagnostic, we can
distinguish:" from this conversation and inserted it at the end of Section 2
of chdr_collision_diagnostics.tex. It is Table 2 on page 2: Gaussian envelope
(1 mm rms, 48 GHz), collective search interval (25--100 um, 3--12 THz),
and selected microscopic probe (0.5 um, 600 THz). Retained the qualification
that the latter rows do not establish a collision-generated wavelength.
Verified that removing the inserted block reproduces the current author-edited
source exactly, including the author's commented-out Section 1 paragraph.
PDF and source ZIP refreshed; 13 pages, no build warnings or unresolved
references, table placement visually checked, ZIP contents match source.

## Wording clarification, 24 September 2026

Replaced the compressed response-weighted-density-correlation sentence in
chdr_collision_diagnostics.tex with the agreed explanation of electron-pair
separations and trajectory/spectral/angular/polarization response. Clarified
the following sentence's subject. Rebuilt the PDF and source ZIP; no build
warnings. Only page 1 text changed; its layout was visually checked.

## Collision diagnostic proposal, 24 September 2026 (completed)

Goal: standalone LaTeX/PDF in this directory consolidating feature scales,
spectrometer resolution, radiation statistics and an experimental strategy for
measuring electron collision effects. Preserve the user-edited literature
report and slides. Baseline: 60 MeV kinetic, 1 nC, Gaussian rms xyz=1 mm;
1 eV is laboratory rms energy spread, not a rest temperature. Unknown gap,
transverse phase space and density/transport history prevent a unique
collision-feature or photon-yield prediction.
Delivered chdr_collision_diagnostics.tex/.pdf (13 pages), standalone .bib
(15 references), build_collision_diagnostics.sh, README_collision_diagnostics.md,
plot_collision_diagnostics.py, output/collision_diagnostics/ numerical data
and two vector figures, plus output/pdf/chdr_collision_diagnostics_sources.zip.
The PDF is also copied to output/pdf/. Uses standard BibTeX and TeXShop
pdflatexmk. Covers fixed-N radiation moments, connected fluctuations,
transverse response, all discussed beam/correlation scales, Debye/HNC/DIH
limits, instrument resolution and counts, gap feasibility, measurement
precedents and a collision-attribution experimental programme.
Independent audits checked spectral-width derivations and collision
attribution. A prescribed full-envelope sinusoidal modulation has intensity
FWHM79.445 GHz; 0.5 um period gives R7547 for equal instrument/source width,
or R15094 for instrument half-width. These are conditional line examples,
not requirements for broad screening spectra or a predicted collision line.
Checks passed: Gaussian widths/crossover, area-conserving convolution,
agreement with prior Debye calculator, 15 resolved citation keys, source
hygiene, no LaTeX/BibTeX warnings, all-page visual review with corrected
figure annotations/legend, ZIP CRC/exact-content checks and a clean build
from extracted sources with identical 13-page text. pdftotext was absent
from PATH; bundled pypdf supplied PDF text validation. No solver/MPI changes.
Remaining physics inputs: transverse velocity distribution, density/transport
history, normal clearance/halo, finite-radiator yield and predicted collision
contrast. No new HNC or collision-dynamics result is claimed.

## Debye sphere and collision feature scales, 24 September 2026 (completed)

Added screening/collision/diagnostic discussion to
chdr_correlation_heating_review.md and reproducible estimate_debye_scales.py.
Outputs are output/debye_scales/conditional_debye_scales.csv and parameters.json.
For the Gaussian 60 MeV, 1 nC, sigma_xyz=1 mm case, proper peak density is
3.347e15/m^3 and a_WS'=4.147 um. Conditional isotropic rest temperatures span
the cold DIH endpoint to 100 meV. At the DIH endpoint Gamma=2.22874, the formal
Debye count is 0.05784: weak Debye screening fails, requiring pair correlations.
Distinguished sphere occupancy from collision rate; added Coulomb encounter
and relaxation scaling, local plasma response clock (10.88 m convected length),
intensity dependence, weak-equilibrium S(k) and conditional frozen-pattern
projection to a laboratory spectral crossover. Added Stupakov's 2025 IBS
preprint, explicitly marking its additional long-range conclusions preliminary.
The user's 1 eV lab rms energy spread specifies only a longitudinal moment;
transverse emittance/divergence and transport/density history remain unknown.
An optional question requests these; no beam-specific collision rate is claimed.
Calculator ran with ~/.venv-h6 and passed identity, thermal-velocity, density
scaling and cold OCP energy checks. No HNC/particle simulation or MPI changes;
main LaTeX, bibliography and edited slides preserved.
Independent scientific review found no material issues. Text/control-character
and math-delimiter checks, output/schema checks and patch whitespace review
passed. Next physics step: constrain the rest-frame velocity distribution and
density/transport history, then validate HNC correlations against particle
dynamics before folding them through the ChDR response.

## Beam intensity correction, 24 September 2026

User noted no intensity enters the proposed collective-substructure scale.
Clarified in chdr_correlation_heating_review.md and its proposed statement:
53 um is the geometric impedance response scale, not a predicted feature.
At 1 nC and sigma_z=1 mm the peak current is119.6 A. Induced bunching gain
in the cited linear drift/dispersive model explicitly depends on current,
interaction length, R56 and energy spread. Actual feature prediction requires
that intensity-dependent dynamics; the proposed THz band is provisional.
Checked primary Eq1 and peak-current arithmetic. No model/slide changes.

## Diagnostic feature-scale statement, 24 September 2026 (completed)

User asks for physically motivated feature sizes besides density spacing.
Extended chdr_correlation_heating_review.md with the finite-transverse-size LSC
scale k*sigma_perp/gamma~1: at sigma_perp=1 mm and 60 MeV, reduced length
8.445 um, modulation period53.06 um and frequency5.650 THz. Candidate search
band25--100 um /3--12 THz is conditional on actual transport, not a prediction
of existing microbunching. Primary references: Schneidmiller/Yurkov2010
Sec2.3 and IPAC2011 Eq1. Distinguished this from Gaussian envelope47.71 GHz,
microscopic g/S correlations, thermal mixing and instrumental gap acceptance.
Clarified period versus Gaussian rms feature width (2pi factor). Computed
ChDR vacuum coupling lengths and gap scales; 1 mm single-track gap must not
be mistaken for safe clearance of a 1 mm rms beam. Calculations and primary
source interpretation independently checked. Also corrected a missing LaTeX
thin-space escape in the prior note. No model, slide or main-report changes.
No numerical solver/MPI tests required for this research discussion.

## HNC and correlation-heating theory review, 24 September 2026 (completed)

User pointed to equilibrium heating of a cold random spherical cloud in a
constant focusing channel, with temperature oscillations. Reviewed Gericke /
Murillo, explicit HNC/OZ equations, Maxson's confined electron-cloud benchmark,
Chen's kinetic-energy oscillations, Dubin/O'Neil's trapped OCP theory, and the
local P3MHeating / OPALX ConstantFocusing implementations. Added
chdr_correlation_heating_review.md with sources, equations, limits and next
validation targets. HNC plus energy conservation gives equilibrium temperature
and pair correlations; dynamical P3M/MD is needed for the transient oscillations.
The OCP energy-fit benchmark gives Gamma_f=2.22874 for the cold random confined
reference. Corrects the scope of the previous purely ballistic 1 eV lab-spread
answer: self-consistent correlation heating can change that spread. Feature
predictions require g(r), S(k), actual transport and ChDR response weighting.
Conditional beam-frame scales calculated with scipy.constants; equation and
numerical independent review completed. Local audit confirms existing DIH
benchmark uses cm/s units and raw temperature variances can contain breathing.
No HNC solve or particle simulation run, no field-solver/MPI changes, no edited
slides/LaTeX/BibTeX changes. Only research note and task-state files changed.

## Measurement literature LaTeX integration, 23 September 2026

User clarified that the detailed research should extend the canonical
chdr_literature_review.tex. Added Section 5, "Measured Cherenkov bands and
the current 1D study", pages 10--13, and nine bibliography entries (25 cited
references total). Existing article text, original bibliography entries and
the old 5 nC / 3 mm setup are preserved. The new section explicitly uses
60 MeV kinetic, 1 nC and sigma_z=1 mm, with sigma_x=sigma_y=1 mm and gap
undecided. The report now has 15 pages. Added measurement tables, comparison
figure, form-factor/gap calculations and instrument-response interpretation.

The presentation has three added slides (11,15,19), using a wide variant of
the same figure. The comparison generator and presentation Makefile maintain
that variant. README_latex.md now lists the required external PDF and the
updated source ZIP includes it, its generator, TeX/BibTeX, README and build
script. All original source content was checked against pre-edit snapshots.
Both builds have no warnings; changed pages and bibliography were visually
reviewed; source archive CRC/content and a clean build after extraction pass.
No solver/numerical model changes or MPI/GPU tests. No remaining work.

## Measurement bandwidth literature comparison, 23 September 2026 (completed)

Request: research electron Cherenkov measurements and compare their measured
bandwidths with the current presentation cases. Current parameters supersede
the older mesh-budget assumptions below: 60 MeV kinetic energy, 1 nC,
sigma_z = 1 mm rms (sigma_t = 3.33576 ps), with sigma_x = sigma_y = 1 mm
from the earlier geometry discussion. Actual beam--surface gap is undecided.
Keep the user-edited slides and existing LaTeX report unchanged.

Primary-source review covers external/hollow-prism ChDR at Tomsk, CLEAR,
CLARA, CESR and ATF2, with a vacuum-channel THz source as a separate geometry.
Distinguish filters, integrated detector bands, usable spectral reconstruction
intervals and train-induced linewidths. Compare the coherent GHz envelope
and 0.5 micrometre microstructure case separately.

Added `chdr_measurement_bandwidth_review.md` and
`compare_measurement_bands.py`, with CSV/JSON and PNG/PDF outputs under
`output/measurement_bandwidth/`. Main findings: Gaussian 1/e form-factor scale
47.71175 GHz; 20--120 GHz is a proposed initial study band, supported by the
Tomsk/CLEAR precedents. CLARA's 300--800 GHz reconstruction band and CLEAR2025's
400--600 GHz detector target shorter bunches. Optical measurements establish
incoherent ChDR, not optical microbunching performance. At 60 MeV/500 nm the
least-evanescent amplitude scale is 9.423 micrometres; the earlier 1 mm rms
transverse beam is incompatible with a nonintercepting 10 micrometre centre gap.
Finite geometry, material dispersion, transverse coherence and detection
response remain necessary for absolute measured-signal predictions.

Checks: regenerated analytic outputs with ~/.venv-h6; verified monotonicity,
wavelength/time conversions, fixed-N pair/self crossover and modal power
scaling. Independently reviewed primary-source band definitions, source
species and numerical outputs; visually inspected the figure. Diff whitespace
check passed. No field solver, MPI or GPU changes/tests apply. Existing slides,
LaTeX/BibTeX report and solver files were preserved. No outstanding work for
the requested research comparison.

## mesh-1 implementation, 18 September 2026

The subsequent user request authorized construction. Implemented
`demos/chdr/src/mesh-1.cpp`, YAML cases, a staircase-prism geometry and generated
Python viewer inspired by OPALX. Reuses FEL mesh/layout/field types. See
`../src/README.md` and `../src/HANDOFF.md` for scope, build and validation.
Four CTest cases pass; additional four-rank/all-axis and anisotropic tests
match serial material counts/checksums/slices. No solver evolution or particle
initialization, no changes to this literature report's LaTeX/BibTeX/PDF.

## Mesh/prism input discussion, 18 September 2026

User requested a new `src` directory and discussion of shared Meep/FEL-style
input before constructing the computational domain. Created `demos/chdr/src`
with a design-only `README.md`; no parser, mesh, material field or solver was
implemented. Proposed a common declarative YAML file with physical units,
domain lower corner/size, integer mesh cells, materials, and a prism defined
by bottom-face vertices plus extrusion axis/height. Sample reproduces the
conditional compact budget, not an approved final geometry. Meep Python
adapter and IPPL C++ reader can consume the same data; Meep's executable
Python/Scheme input is not directly a C++ configuration file.

Verified FEL actually loads JSON through Catalyst Conduit (`Config.h:307`),
while IPPL already loads/parses YAML with the same dependency in
`ProxyWriter.cpp:1011,1026`. Meep uses scalar pixels-per-length resolution;
IPPL supports independent hx/hy/hz. Require explicit handling of anisotropic
grid differences. Proposal translates all positions into a zero-centred Meep
box, preserves prism bottom-face semantics, and switches off Meep smoothing
for an initial staircase comparison without claiming identical Yee/collocated
coefficient arrays. Existing planar Meep benchmark remains unchanged.
Independent source audit confirmed lower+(i+1/2)h matches collocated PIC
sampling and warned against treating getVertexPosition as a cell-centre API.
Checks for this discussion are source/documentation review, YAML example
syntax/arithmetic and diff review; no C++/MPI/Meep simulation is warranted.
Next step: discuss schema/geometry choices with the user, then implement the
geometry-only stage if requested.

## Radiator dimensions and AWA mesh budget, 18 September 2026

Completed literature sizing request. User confirmed sigma_x = sigma_y = 1 mm; retain K = 60 MeV, Q = 5 nC and sigma_z = 3 mm. Gap remains unspecified: a = 5 mm is explicitly illustrative. Checked primary papers and rendered figures: Curcio CLARA reports 50 mm base and 40 mm width but omits triangle height/angle; CLEAR's 25 mm length/50 mm base/5 mm bore is a different geometry. Shevelev/Konkov uses a 45 mm triangle leg with infinite extrusion; Tyukhtin's inverted prism is wavelength-scaled. Supplementary Potylitsyn/Popov/Sukhikh (2010), DOI 10.1088/1742-6596/236/1/012025, specifies a right-isosceles PTFE prism with 247 mm beam-facing hypotenuse and 74 mm extrusion; derived depth 123.5 mm.

Added `chdr_radiator_mesh_estimate.md`, `estimate_chdr_mesh.py` and `output/mesh_budget/{mesh_budget.csv,mesh_inputs_and_scales.json}`. Conditional compact geometry assumes (25,40,50) mm; it is not the fully recovered CLARA device. For +/-4 sigma source support, full source sweep across the interaction region, 20 mm vacuum plus 10 mm absorber allowance per side, rounded domains are (100,100,160) mm and (200,140,360) mm. Idealized epsilon_r = 2.13 and chosen 40 GHz band give lambda_d = 5.135 mm; h = 0.25 mm gives ~20.5 cells/wavelength, 4 cells/transverse rms. Cubic budgets: 102.4 million cells/18.84 GB and 645.12 million/118.70 GB for core fields only (23 doubles/cell). Current active nonstandard solver rejects cubic meshes: (0.25,0.25,0.125) mm passes its mesh check and doubles these counts; no claim of dielectric stability. Standard dt ~0.417 ps; 4 ns is workload planning only and needs consistent beam outflow.

Validation: calculator executed using ~/.venv-h6, independent reviewer assertions checked domains/padding, spectral scales, grid products, memory/refinement scaling, standard dt and nonstandard condition. Final local checks cover regenerated outputs, alternate CLI inputs and diff whitespace. No solver, particle, MPI or GPU run; no changes to existing LaTeX/BibTeX/PDF or C++ files. Open physical decisions: actual gap, compact prism height/face angle, material response and bandwidth. Padding/absorber/source initialization, outflow, resolution and duration remain convergence questions. Report results to user with conditional estimates and source links.

## Two-media representation on a uniform mesh, 18 September 2026

Discussion only: keep the potential formulation and distinguish material geometry from field resolution. Inspected UniformCartesian: separate constant hx/hy/hz are supported, coordinate-dependent spacing is not part of this mesh. Proposed analytic prism half-spaces plus a scalar epsilon_r/material-ID field and interface-only metadata (normals, stencil intersections, fractions); precompute material-aware potential coefficients. Distance-weighted harmonic averaging is a concrete 1D scalar normal-flux example, not a full vector-potential or oblique-interface method. Recommend aligned baseline, then exact geometry/interface corrections and grid/translation convergence. Subcell interface data cannot repair unresolved gap/source/wavelength scales. Nested uniform grids and coordinate mapping remain larger optional extensions; both need new operators/coupling/source treatment. Asked which limitation matters most (tilted-surface accuracy, gap resolution or vacuum volume); no answer received at time of this note, so comparison remains conditional. No solver/source/LaTeX edits or numerical tests; only this task-state entry was added.

## Potential formulation reassessment, 18 September 2026

User asks why the existing four-potentials cannot be retained with a changed stencil. They can: revised the design note to prioritize a potential-based planar-interface prototype and retain the Yee design as an alternative. Independently checked the weighted generalized Lorenz-gauge equations against Chew (2014), Sections 2–3, and Ryu et al.'s potential FDTD formulation. Uniform epsilon requires speed/source scaling; discontinuous epsilon requires weighted scalar and vector operators and transmission/gauge consistency. Phi and A can be decoupled by gauge choice, although A components remain coupled. Preserve existing storage/time levels/MPI/Kokkos; audit reconstruction, discrete operator compatibility, sources and stability. No C++ or LaTeX/PDF changes, no numerical implementation or solver tests. This reassessment supersedes the preference for a new direct Yee backend in the previous entry.

## Prism solver design, 18 September 2026

Request: inspect IPPL/FEL and propose discretization, material and solver changes with a class diagram. Reviewed branch `chdr`, commit `ad468dbdd`; no C++ edits or simulation runs. Added `chdr_prism_solver_design.md` with code references, proposed classes, material equations, integration details and staged validation. Key decision: a lab-frame ChDR manager plus separate direct D/B Yee/material backend, preserving the existing collocated vacuum four-potential FEL solver. Existing Field centering/generic curls and `assemble_current_yee` are not sufficient for drop-in Yee PIC: explicit offsets/operators, density normalization and general 3D continuity are required. Retain immutable distributed material coefficients, initially scalar/staircased then a stable tensor interface update; add CPML, source initialization and broadband diagnostics. Existing Maxwell examples are COMPILE_ONLY. Checks for this task are source/design/diff review only; numerical, MPI and GPU validation remain proposed implementation work. Current LaTeX/BibTeX/PDF files are unchanged. Next step, if implementation is requested: Yee layout/operator and vacuum tests before material and electron coupling.

## Python prototype implemented, 15 September 2026

User requested directory `1d` and a working script varying gap, kinetic energy and rms pulse duration. Completed in `../1d/`: `chdr_1d.py`, tests, README and dependency list. Full transverse integration, SI field/spectrum normalization, Gaussian bunch response, parameter sweeps and CSV/JSON/PNG/SVG outputs are provided. All 11 physical/CLI tests pass; vacuum Bessel fields, interface matching and flux/work agreement checked. Eleven example cases are under `../1d/output`; maximum flux/work relative difference 6.18e-13. See `../1d/HANDOFF.md` for design and validation, including the justified zero-component numerical tolerance. The report and C++ solver were not edited for this implementation. No 3D FDTD comparison has yet been run.

## Python prototype discussion, 15 September 2026

Current request is discussion, not implementation. Proposed an independent frequency-domain half-space reference in SI with beam +z, electron at x=a>0, vacuum x>0 and dielectric x<0. Begin with k_y=0 (TM fields E_x, E_z, H_y), analytic homogeneous-region solutions and matching H_y and H_y'/epsilon at x=0; then handle arbitrary k_y and integrate to reconstruct the full 3D point-electron field. A single k_y=0 mode is not the point-electron field at y=0. Benchmark complex E/H, interface jumps, normal phase/decay, modal gap dependence and eventual energy loss/flux per path length. Account for finite-domain/startup effects, sampling/Fourier normalization and the PIC particle shape in comparisons. Fixed-trajectory source isolates the material solver from particle feedback. Plane-wave Fresnel checks can precede this. Checked Tyukhtin et al., Section III, via the primary arXiv paper. Computed illustrative 16 GHz scales for 60 MeV and epsilon_r=2.13: lambda0=18.7370 mm, lambda_d=12.8384 mm, kappa0(k_y=0)=2.83192 /m (353.118 mm decay length), k_y cutoff=356.455 /m, internal angle=46.7476 degrees. Gap remains a parameter, not a newly assumed physical value. No Python prototype or report changes have been made; only a direct scalar calculation was run. Present the proposed stages and validation limits for discussion.

## Section 4 integration, 15 September 2026

Completed request: add the FEL capability assessment under Section 4 and add Taflove references plus the "Moving electron source" discussion. Baseline source SHA-256: `cc620aad8107e000975b75dd55141ad58ce7a31449dbbe6355877362ab645007`; source/bibliography/README backups are in `/tmp/chdr-section4-g08u0uv_`. Replaced the overly categorical charge-conservation paragraph in Section 4.1 with existing PIC capabilities, zero-potential startup and required continuity/Gauss verification. Added the moving-source example from Oskooi/Johnson, Chapter 4, Section 4.7 of the Taflove co-edited 2013 volume, with correct chapter authorship, plus Taflove/Hagness (2005). Added three entries to the main BibTeX database (16 total) and updated README; Section 3.4 only gains a cross-reference label. User prose, figure, formulas, coordinates, document title and working BibTeX configuration are otherwise preserved. Removed the forced break before Section 4.3 and reduced bibliography item spacing from 7pt to 3pt to avoid nearly empty pages. Final report remains 9 pages. BibTeX/pdfLaTeX build passes without warnings, all 16 references resolve, all nine rendered pages were visually checked, and the complete source diff was reviewed. Original figure, displayed equations and 13 existing bibliography entries are byte-preserved. PDF copies match and the refreshed four-file source ZIP passes CRC and content comparison. No solver changes or simulation claims; no outstanding work for this request.

## Section 3.4 implementation check, 15 September 2026

User asks whether Section 3.4 is fully covered by the FEL mini-app. Read the current manager, collocated deposition, potential initialization, field reconstruction and deposition tests. Existing components: relativistic Boris push, field gathering, trajectory-segmented collocated CIC current deposition, CIC charge deposition conditional on `space_charge`, and particle initialization. Qualifications: `NonStandardFDTDSolver.hpp:124` zeros all three potential time levels; no consistent pre-existing electron self-field initialization is called in the manager's `pre_run`. `TestCurrentDeposition.cpp` checks prescribed deposition values, not the discrete continuity/Gauss residual stated in Section 3.4. Thus the particle/source machinery is reusable, but complete coverage of initialization and exact discrete conservation is not established by this source review. Report this distinction in response to the user's "correct?" question. No document or solver changes and no simulations were performed for this check; the initial intention to revise Section 3.4 was deferred while answering the explicit confirmation question.

## Taflove literature check, 15 September 2026

Latest request: check Allen Taflove's works against our ChDR problem. Added `chdr_taflove_review.md` and `chdr_taflove_references.bib`; the author's main `.tex`, existing `.bib`, PDFs and source archive are unchanged. Identified the explicit moving-source/Cherenkov example in Oskooi and Johnson, Chapter 4, Section 4.7, pp. 89–91, in Taflove/Oskooi/Johnson (eds., 2013), plus Chapters 5 (PML) and 6 (subpixel smoothing), Taflove/Hagness (2005), and Joseph/Hagness/Taflove (1991). Chapter 4 was inspected directly; remaining access scope is stated in the note. Preserve author versus editor attribution. Flagged the source chapter's footnote 10: no extra gamma multiplier applies to an invariant total point-particle charge; the note derives the delta-function Jacobian cancellation. The earlier analytical vacuum-gap benchmarks remain necessary. No solver modifications or simulation claims. Completed: note and bibliography reviewed; all six new entries and the existing thirteen process together with BibTeX/unsrtnat, with no duplicate keys, missing entries or warnings. The scratch build was automatically removed. No PDF rebuild was needed for this separate research note. Ready to report findings; no outstanding work for this request.

## Previous coordinate consistency review

Latest request: add the explanations of the Fourier transform in y and the dielectric subscript d to Section 2.2. Added material uniformity/independent k_y modes, fixed k_z, the localized electron/all k_y reconstruction, the meaning of d and an unnumbered dispersion relation deriving Eq. (2.5). Existing equation numbers are preserved. Removed the forced page break before the FDTD section because the added text otherwise left a page containing only four lines. Source baseline: f8c3db8137eb7888e5141fb6a71371c6b7e8953bd23d1d1fc390c117c5970e7e. Completed: final BibTeX build passes without warnings; the report remains 9 pages with all 13 references. Eq. (2.5) retains its number. All pages visually reviewed; source diff confined to Section 2.2 and the one page-break removal. PDF copies match, bibliography is unchanged, and ZIP CRC/contents match the current source. Scratch files removed after verification. No further work remains for this request.

Current request: the author changed the beam direction to +z; check the report and update its formulas accordingly. Work from the latest source, preserve unrelated author edits and the working natbib/BibTeX configuration.

Convention: Cartesian order (x,y,z), beam +z, vacuum x>0, dielectric x<0 in the half-space reference, gap a along x, remaining transverse coordinate y. Use sigma_z for bunch length, r_e=(a,0,vt), J=v rho z-hat, k_z=omega/v, Fourier variable k_y and dielectric normal wave number k_{x,d}. Check source continuity, dispersion, bunch form factor, interface/tensor component mapping and code coordinate mapping. Scalar physical values are invariant under this relabelling.

Changes applied in `chdr_literature_review.tex`: removed obsolete s/u/w notation in favor of explicit Cartesian coordinates; corrected trajectory, charge/current source, phase matching, transverse Fourier wave number, normal dielectric wave number, bunch length/duration/form factor/domain span, finite-body symmetry direction and job-file mapping. Added the x-normal component/tensor interpretation and corrected two TikZ comments; rendered figure geometry and labels are preserved. `README_latex.md` documents the convention. Bibliography and build script are unchanged.

Checks: reviewed the complete source diff; no residual old-axis formulas found. Direct distributional differentiation gives partial_t rho + partial_z(v rho)=0 for the new source. The dispersion identity and form-factor scale were checked numerically; gamma=118.4170710, sigma_t=10.0073 ps, f_*=15.904 GHz and theta=46.7476 degrees are unchanged. SymPy was unavailable in ~/.venv-h6; no environment changes were made, and the continuity/tensor algebra was checked directly. The BibTeX build passes without warnings, with 9 pages and 13 references. All nine rendered pages were visually inspected.

Source baseline before this change: `a4c6266499aa8c61e38f7a44565618545484d7bc926006f4a9f13868b49986b0` (SHA-256). Unchanged bibliography: `df7cc2f1abe31f7681dc3c6e0f64f9b425052d4a67851f30e1c64499fed1d218`. No solver code was changed or simulated.

Completed: source archive refreshed and byte-for-byte contents/CRC checked; the two PDF copies are identical. Final source audit confirms all old-axis formulas are removed, rendered figure content and bibliography are unchanged, and the 9-page PDF has all 13 references. Scratch files removed after verification. Deliver the updated PDF and source ZIP.

## Previous completed bibliography repair

Latest request (15 September 2026): fix the missing literature while preserving the author's latest edits. The user uses TeXShop and explicitly requested BibTeX.

## Final implementation

- `chdr_literature_review.tex`: changed bibliography setup from biblatex/Biber to natbib with numeric, square, sorted/compressed citations and `unsrtnat`. Adapted bibliography spacing and retained the numbered "Literature and software references" heading. No changes to author/title, figure, axis labels, prose, equations or reference keys.
- `chdr_references.bib`: added `year = {n.d.}` to the three undated Meep entries so BibTeX prints no date explicitly and emits no missing-year warnings. All other metadata is unchanged.
- `build_literature.sh`: executable, portable pdfLaTeX/BibTeX/pdfLaTeX/pdfLaTeX build; writes PDF next to source and copies it to `output/pdf/`.
- `README_latex.md`: TeXShop BibTeX workflow, automatic pdflatexmk option, command-line build and notes on the three undated software entries.
- PDFs and source ZIP: completed after visual QA.

## Diagnosis and decisions

The user's build ran BibTeX against a source requiring Biber, producing an empty `.bbl`. An initial Biber rebuild succeeded, then the user explicitly asked for BibTeX. An obsolete user-local biblatex.bst (format 2.5) also shadowed TeX Live's current style (format 3.3). Testing the current style locally recovered entries but biblatex's BibTeX fallback retained spurious citation-order rerun warnings. Final solution uses standard natbib/BibTeX, avoiding this compatibility path entirely. Temporary local biblatex.bst was removed. No user TeX installation files or TeXShop preferences were changed.

## Preservation and checks

Original author `.tex` SHA-256: `42f3e79b342490c166c74c098806176e940eb1a32e1da4eae23579e6f04940cd`.
Original `.bib` SHA-256: `0553c99119aac58e7dd13202ed228a8796662284823e60615a2cfddbc2199b36`.

Source diff reviewed: only bibliography configuration, spacing and printing changed. Full pdfLaTeX/BibTeX build succeeded, 9 pages, all 13 bibliography entries generated. Final LaTeX log has no warnings, undefined references, or overfull/underfull boxes. BibTeX also finishes without warnings after explicitly marking the three undated entries `n.d.`; existing access dates are preserved and no year was invented. No solver tests or simulations apply to this document-only repair.

Completed: all nine pages visually inspected; the bibliography was inspected again after adding the no-date markers. Source diff reviewed, all 13 citation keys match bibliography entries, both PDF copies match, and ZIP CRC/contents verified against the final source files. Final outputs: `chdr_literature_review.pdf`, identical `output/pdf/chdr_literature_review.pdf`, and `output/pdf/chdr_latex_sources.zip`. No open implementation work remains; deliver the report and TeXShop BibTeX workflow.
