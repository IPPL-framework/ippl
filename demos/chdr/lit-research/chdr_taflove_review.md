# Allen Taflove's work relevant to the ChDR simulation

Literature check, 15 September 2026. Geometry: electrons travel along **+z**, remain in vacuum at gap **a**, and excite a dielectric across the plane **x = 0**. Beam parameters remain 60 MeV kinetic energy, 3 mm rms longitudinal length and 5 nC.

**Finding:** Taflove's work supplies useful FDTD foundations and material models. His 2013 edited volume also contains an explicit Cherenkov moving-source example and chapters addressing tilted dielectric interfaces and absorbing boundaries. These are numerical-method references; the analytical vacuum-gap benchmark remains necessary.

The companion `chdr_taflove_references.bib` contains conventional BibTeX entries. This supplementary note does not change the author's working LaTeX report.

## Reading priorities and access

| Priority | Reference | Relevance and access |
|---|---|---|
| 1 | A. Oskooi and S. G. Johnson, **Electromagnetic Wave Source Conditions**, Chapter 4, pp. 65–100, in A. Taflove, A. Oskooi and S. G. Johnson (eds.), *Advances in FDTD Computational Electrodynamics: Photonics and Nanotechnology* (2013). | Read **Section 4.7, pp. 89–91, especially Fig. 4.10**. It demonstrates Cherenkov radiation from a moving current and the artifacts caused by rounding its position to grid points. Interpolation reduces these artifacts; numerical dispersion remains. The homogeneous-medium example is not our vacuum-gap benchmark. [Open author manuscript](https://arxiv.org/pdf/1301.5366). |
| 2 | A. Oskooi and S. G. Johnson, **Accurate FDTD Simulation of Discontinuous Materials by Subpixel Smoothing**, Chapter 6, pp. 133–148, in the same volume. | Addresses interfaces that do not align with a Cartesian grid, including material averaging, field-component interpolation and convergence. This is the most relevant chapter for the eventual 47° radiator. [Book preview: contents and editorial summary](https://api.pageplace.de/preview/DT0400.9781608071715_A24132289/preview-9781608071715_A24132289.pdf). |
| 3 | A. Oskooi and S. G. Johnson, **Rigorous PML Validation and a Corrected Unsplit PML for Anisotropic Dispersive Media**, Chapter 5, pp. 101–132, in the same volume. | Covers absorber validation, anisotropic/dispersive materials and failure for oblique waveguides. Relevant when designing the outer boundary, especially where material intersects an absorber. The physical vacuum–dielectric interface is a separate issue. [Publisher and contents](https://us.artechhouse.com/Advances-in-FDTD-Computational-Electrodynamics-P1567.aspx). |
| 4 | A. Taflove and S. C. Hagness, **Computational Electrodynamics: The Finite-Difference Time-Domain Method**, 3rd ed., Artech House (2005). | The foundational reference: Yee's method, numerical dispersion and stability, PML, near-to-far-field transformations, dispersive materials and subcell geometry. [Publisher, contents and accompanying code](https://uk.artechhouse.com/Computational-Electrodynamics-Third-Edition-P1916.aspx). |
| 5 | R. M. Joseph, S. C. Hagness and A. Taflove, **Direct time integration of Maxwell's equations in linear dispersive media with absorption for scattering and propagation of femtosecond electromagnetic pulses**, *Optics Letters* **16**, 1412–1414 (1991). | Treats pulse propagation and reflection from a Lorentz-medium half-space, with analytical comparisons. Relevant when replacing constant permittivity with a causal dispersive material model and testing its interface response. It does not provide the material parameters for our radiator or a vacuum-gap electron calculation. [Publisher abstract and DOI](https://opg.optica.org/ol/abstract.cfm?uri=ol-16-18-1412). |

**Attribution:** Taflove is a co-editor of the 2013 volume. Chapters 4, 5 and 6 are authored by **Oskooi and Johnson**; they should be cited under those authors, with Taflove among the book's editors. Chapter authorship and page ranges are verified in the book's own contents. The moving-source chapter was inspected directly; Chapters 5 and 6 were assessed through the contents and editorial summaries, not a complete reading of those chapters. The 2005 book was checked through its publisher's contents, and the 1991 paper through its publisher's abstract.

## Consequences for our simulation

The following are our implementation recommendations, based on the references above and the already reviewed ChDR literature.

1. **Start in the laboratory frame with a prescribed electron trajectory.** This isolates the moving source, material response and radiation diagnostics. The FEL mini-app's existing boosted-frame potential solver requires a separate derivation of the material equations; a textbook Yee update cannot simply be inserted into it.
2. **Validate the source in vacuum at the actual beam speed.** For 60 MeV kinetic energy, gamma is about 118.42 and beta about 0.999964. A uniformly moving charge has a bound field, but no physical Cherenkov emission in homogeneous vacuum. Check numerical radiation and start/stop transients before attributing a signal to the dielectric. Smooth interpolation alone does not establish discrete charge conservation; use the deposition/field-operator consistency discussed in the existing review.
3. **Test a grid-aligned material interface independently of the electron.** Compare plane-wave reflection and transmission against Fresnel coefficients. Then add the electron at gap a and compare fields or spectra with the half-space spectral solution already identified in the review. A homogeneous-medium Cherenkov cone is an additional check, not a substitute for the gap-dependent benchmark.
4. **Add the tilted radiator after those comparisons pass.** A tilted surface is feasible in Cartesian FDTD. Discontinuous material coefficients require careful component-dependent treatment. The existing Farjadpour/Kottke references provide the detailed smoothing theory; Chapter 6 offers a consolidated account. Derive the tensor update and verify stability for the chosen solver. Sharp corners can limit convergence even when planar interface treatment is accurate.
5. **Verify extraction and absorption separately.** Test mesh refinement, source-to-boundary distance, absorber thickness and observation time. Compare spectra and integrated flux, not only field images. Optical extraction from a finite prism remains a different observable from emission inside an infinite dielectric half-space.

These numerical checks apply equally when each computational particle represents one physical electron.

## Relativistic source normalization

For one electron of invariant total charge q = -e, our laboratory-frame source is

\[
\rho(x,y,z,t)=q\,\delta(x-a)\delta(y)\delta(z-vt),
\qquad
\mathbf J(x,y,z,t)=v\rho(x,y,z,t)\,\hat{\mathbf z}.
\]

This source obeys the continuity equation and integrates to q at every time. It needs **no additional gamma multiplier**. The source chapter's footnote 10 on p. 90 suggests such a multiplier; that statement must not be applied to the invariant total point-particle charge used here. [Source chapter](https://arxiv.org/pdf/1301.5366).

Our normalization check is explicit: under a boost along z, the transformed rest-frame density contains the product

\[
\gamma q\,\delta(x-a)\delta(y)\delta\!\left(\gamma(z-vt)\right)
=q\,\delta(x-a)\delta(y)\delta(z-vt).
\]

The density transformation and the delta-function Jacobian cancel. Multiplying this laboratory-frame expression by gamma again would incorrectly change the total electron charge.

## Assessment

The strongest addition to the present literature review is the combination of **Chapter 4, Section 4.7** for moving sources, **Chapter 6** for dielectric geometry, and **Chapter 5** for absorber validation. Taflove and Hagness (2005) supplies the broader FDTD derivation; Joseph, Hagness and Taflove (1991) supplies an early dispersive half-space validation example.

In the sources checked, I did not identify a Taflove-authored analytical solution specifically for an electron remaining in vacuum at gap a. The previously identified half-space and finite-prism papers remain the more direct physical benchmarks. No solver changes or simulation results are claimed by this literature check.
