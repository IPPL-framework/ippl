# Planar ChDR Python prototype

## Correlation-heating qualification (2026-09-24)

The user identified disorder-induced heating in a confined cold random cloud.
The previous 1 eV laboratory energy-spread estimate below is only a ballistic
phase-mixing calculation, not a self-consistent prediction of a fixed spread.
See ../ideas-lit-reseach/chdr_correlation_heating_review.md for the HNC/OCP
review, equilibrium energy balance, oscillations, structure factor and local
P3M benchmark. Correlations can change the independent-particle noise reference
and must be folded through the ChDR response. No radiation model/slides changed.

## Energy spread and feature-scale discussion (2026-09-23, completed)

User clarified that the proposed 1 eV is a LABORATORY energy spread, not
isotropic rest-frame thermal energy. Discussion assumes rms uncorrelated
Gaussian sigma_E=1 eV at 60 MeV kinetic. Analytic derivative of relativistic
velocity gives sigma_vz=3.53324e-4 m/s; fixed-energy free drift gives rms
longitudinal blur=1.17860 pm per metre. A prescribed coherent modulation
is multiplied in amplitude by exp[-(2*pi*blur/lambda_b)^2/2], not the Poisson
shot-noise floor. General linear transport uses blur=abs(R56)*sigma_delta,
sigma_delta=sigma_E/(beta^2*gamma*mc^2)=1.65271e-8. Temperature/energy spread
sets transport-dependent survival, not a selected generated wavelength;
n^(-1/3) is unchanged at fixed density. The variance-equivalent longitudinal
rest thermal energy is only 1.39567e-10 eV; it does not establish isotropic
thermal equilibrium or justify a scalar Debye length. Unknowns for structure
formation remain transverse emittance, upstream transport and collective
forces. Primary reference: Marinelli & Rosenzweig, PRSTAB 13, 110703 (2010),
Eqs. (19), (23), DOI 10.1103/PhysRevSTAB.13.110703. Numerical values computed
with scipy.constants and independently checked. No model, slides or report
changed. No numerical solver/MPI tests apply to this discussion.

## Phase-sum explanation (2026-09-23, completed)

Read the user's latest slide edits and added a short explanation of T_N on
slide 8: dimensionless complex phase sum for one actual bunch, unit-magnitude
terms with arrival-time phases, and |T_N|^2 as the single-electron spectrum
multiplier. All equations and the rest of that slide are preserved. Rebuilding
also exposed overflow from the user's longer slide 3 note; removed its empty
centre environment and reduced list spacing without changing any words.
The other 17 slides are byte-preserved, with 19 pages total. Final build has
no warnings or overfull/underfull boxes; pages 3 and 8 were rendered and
visually checked. Independent wording review and source diff checks pass.
No numerical model, figures, report or bibliography changes.

## Measurement literature integration (2026-09-23, completed)

Request: add the measurement-bandwidth research to the 1D LaTeX material,
and add the figure, concise context and literature to the presentation.
User selected the main report in ../ideas-lit-reseach rather than a new
article. Added its Section 5 on pages 10--13 and nine BibTeX entries (25 total).
The report now has 15 pages. Its source archive includes the external PDF
figure and passes a clean build after extraction.

Presentation: added pages 11 (band comparison), 15 (optical measurements and
gap), and 19 (measurement references), for 19 slides total. All original
16 frame bodies and the preamble are byte-preserved. The comparison script
now also emits a wide version; the presentation Makefile regenerates and
copies it automatically. README files document the new external figure and
the distinction between the old report baseline and current 60 MeV kinetic,
1 nC, sigma_z=1 mm parameters. No model or solver changes.

Validation: both final PDFs compile with no warnings or layout errors.
New pages/slides and reference pages were rendered and visually reviewed.
Initial slide overflow and a Beamer leading-group parsing issue were fixed.
Original report content and bibliography entries are preserved; main/output
PDF copies match. ZIP CRC/content and extracted-source build pass. Source
diff and whitespace reviewed. No numerical/MPI/GPU run applies to this
document-only integration. No outstanding work.

User request: create directory `1d` and a Python prototype varying gap a, electron kinetic energy and pulse duration in ps. Location: `demos/chdr/1d`. No changes to the author's LaTeX report or C++ solver.

Design: independent SI frequency-domain half-space solution, beam +z at x=a>0, vacuum x>0 and real positive constant epsilon_r in x<0. Match TE electric and TM magnetic amplitudes analytically per (omega,k_y); integrate k_y automatically. Output positive-frequency radiated energy per path length and per Hz, Gaussian-bunch coherent/incoherent contributions, and complex single-electron/mean-bunch fields at a dielectric probe. Pulse duration is rms; default 10 ps, 60 MeV kinetic energy, 5 nC, illustrative a=1 mm and epsilon_r=2.13. CLI supports parameter sweeps.

Normalization: time/y forward transform has no prefactor; inverse has (2pi)^-2. Fourier source rho_hat=q/v delta(x-a), Jz_hat=q delta(x-a), with invariant q=-e. Plane-wave fields reconstructed in SI using Lorenz-gauge vacuum potentials and Maxwell interface conditions. Radiated spectrum per Hz is (1/pi) integral dk_y of minus Re(E_hat cross H_hat*)_x; independently verify against -(q/pi) integral Re(Eref_z at source) dk_y. Only propagating modes carry normal flux for the lossless half-space; complex probe fields include all evanescent modes as well.

Validation planned: exact vacuum Bessel-function fields, tangential/normal interface conditions, outgoing dispersion, zero radiation in vacuum/below threshold, modal gap scaling, flux/work balance, quadrature convergence, Gaussian pulse and charge scaling, CLI output checks and visual plot review. Use ~/.venv-h6; numpy/scipy/matplotlib available; pytest absent, so use unittest. No MPI/GPU kernels change; MPI tests not applicable.

Implemented: `chdr_1d.py`, `test_chdr_1d.py`, `README.md`, `requirements.txt` and `.gitignore`. Supports Cartesian-product scans of gap, kinetic energy and rms duration; no-fields mode skips probe reconstruction; pulse scans cache single-electron solutions. Exports SI spectra, full complex E/H and mean-bunch fields, parameter JSON, and PNG/SVG plots. Default run plus gap/energy/pulse examples generated under ignored `output/`. No large-prism approximation is used.

Completed validation: all 11 unittest tests pass. Default flux/work relative difference over 0.2-200 GHz is 6.18e-13. Absolute vacuum fields agree with the Bessel-function solution, including normalization. Initial test failure was solely the analytically zero H_z component (8.6e-31 A s/m) from TE/TM cancellation, approximately 1e-13 relative to a nonzero field. Its absolute tolerance now derives from the independent field scale (5e-12 times max expected H); nonzero-component relative tolerance remains 2e-7. This tolerance change addresses floating-point cancellation, not a changed physics model. Final default/profile and gap/energy/pulse plots visually reviewed; added epsilon/charge labels and writable font caches. All 11 exported example cases checked for finite values, nonnegative spectra, bunch decomposition and expected monotone gap/duration dependence. `output/validation.json` records those checks. Fresh-cache plotting CLI smoke test passes without the earlier fontconfig warning. Source additions and whitespace reviewed; no C++/LaTeX changes, no MPI tests applicable. The requested first prototype is complete. Future work is a 3D comparison, finite geometry and measured dispersive material response.

Reference: Tyukhtin/Galyamin/Vorobev arXiv:2105.01111 Section III (its printed formulas use Gaussian units); code uses a separately derived SI formulation documented in README. SciPy quad_vec API checked in official documentation. Material dispersion, finite edges, optical extraction and self-consistent trajectory changes are outside this first prototype.

## Shot-noise follow-up (2026-09-23)

Question: whether the cube-root density scale for a 1 nC bunch in a uniform
`(1 mm)^3` box, `n^(-1/3)=0.543 um`, means that ChDR shot-noise radiation
should be studied at 0.5 um.

Conclusion: the arithmetic is correct, but the conclusion is not.  For a
fixed physical count and independent arrival times, the field-fluctuation
spectrum is `N*(1-|F|^2)*W1`, which approaches the conventional `N*W1` self
term only above the smooth bunch form-factor bandwidth.  At optical frequency
they are equal to machine precision.  The spectrum is broadband; `d` does not
select a spectral line.  A regular lattice would suppress ordinary Poisson
noise and create Bragg features instead.  The model only represents identical
transverse tracks and cannot derive 3-D correlations from the density scale.

Implementation: added `UniformDensityEstimate`, `--uniform-box-mm`, and
`--wavelength-um`; `spectrum.csv` now includes vacuum wavelength, the
least-evanescent coupling length, and the fixed-N fluctuation spectrum.
The README documents the statistical interpretation and a 0.5-um command.
For 60 MeV at 0.5 um, the coupling length is 9.42 um; a 1-mm gap suppresses
even the least-evanescent modal energy by `6.6e-93`.  The optical check uses a
10-um gap only as a half-space reference.  It still needs measured complex
optical epsilon and finite-radiator extraction before it can predict a camera
signal.

Validation: all 13 `unittest` checks pass after the fixed-N noise update. They
cover the density conversion, the coupling-length formula, the optical CLI
path, invalid derived inputs, duplicate-wavelength coalescing, singleton-band
metadata, and equality of the explicit optical noise output with the existing
incoherent term. No C++/MPI/GPU code was changed.

## Presentation follow-up (2026-09-23)

Latest clarification: after reading the user's edited source (including author,
AWA-campaign notes and shortened variable lists), revised only slides 7 and 8.
Slide 7 now defines F as the profile-weighted complex amplitude of unit waves
and explains that Eq. (3) averages unknown microscopic arrival times at fixed
N and g, not measured bunches. Slide 8 gives the exact single-bunch phase sum,
the ordered-pair derivation, and why a smooth mean field alone misses shot noise.
Eq. (4) moved to slide 8; all 16 page numbers and other user-edited content stay
intact. A comparison against a pre-edit snapshot verified that only frames 7/8
changed. The PDF compiles without layout/reference warnings; the full deck and
the two revised pages were rendered and visually checked. Independent review
found no physics issues; no numerical model or figure data changed.

Latest revision completed: the presentation now uses common 60 MeV, 1 nC,
sigma_z=1 mm parameters. The two cases are full-bunch length diagnostics and
the main optical fine-structure/microbunching application. The 16-slide deck
defines symbols, explains spectral flux/work and fixed-N statistics (still
pages 6/7), adds a phase-sum interpretation and a Fourier-normalization appendix,
and removes the "It does not provide" block. Precise sources: Tyukhtin Sec. III,
Bosch/Bosch Eqs. (29)-(30) and Summary, Schmidt et al. Sec. 3, Oskooi/Johnson
Sec. 4.7. Schmidt's section was checked and corrected from an initial 2.4 to 3.

Added a Gaussian form-factor plot with f*=47.71175 GHz; the existing coherent
cross-term/self-term crossover remains 226.59 GHz. The optical Gaussian plot
is explicitly the unmodulated noise reference for a future microbunching test.
The figure generator derives sigma_t from exactly 1 mm and the model velocity,
and writes proper TeX scientific notation. Existing model defaults are untouched.

Validation: figure regeneration/numerical sanity checks pass; the PDF compiles
with no overfull/underfull boxes or unresolved references. All 16 slides were
rendered and visually reviewed, with detailed inspection of the revised formula,
case, variable, plot, and appendix slides. Initial layout overflows were resolved.
An independent source/physics review found no remaining blockers. Changes are
confined to presentation sources/plots, its README, and this task log.

Goal: provide a TeXShop-ready presentation under `presentation/` that explains
the analytic 1-D reference and shows reproducible plots from the model.

Implementation: `presentation/chdr_1d_presentation.tex` is a standalone
12-slide Beamer deck. It distinguishes the baseline finite-radiator target
(`60 MeV`, `5 nC`, `sigma_z=3 mm`) from the separate `1 nC` density and
optical half-space diagnostics. `presentation/make_figures.py` regenerates the
coherence, optical-spectrum and gap-dependence plots plus `figures/numbers.tex`
from `../chdr_1d.py`. `presentation/Makefile` supports `make figures` and
`make pdf`; the source uses a manual bibliography, not BibTeX.

Physics decisions: the density scale is explicitly marked as a diagnostic, not
a radiation line. The plotted optical and gap spectra use the exact fixed-N
fluctuation expression `N*(1-|F|^2)*W1`. The 0.5-um gap slide calls
`exp(-2a/ell_perp)` the least-evanescent exponential coupling factor, rather
than a complete finite-interface energy prediction.

Validation: `make pdf` completed with no TeX overfull/underfull boxes. The
12-page PDF was rendered with Poppler and visually checked, including the
equations, the case separation, both spectral plots, and the references.
Existing Python model tests remain the appropriate regression suite; no C++,
MPI or GPU code is changed by this presentation.
