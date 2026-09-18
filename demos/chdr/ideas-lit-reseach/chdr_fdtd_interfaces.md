# FDTD Cherenkov radiation: vacuum–dielectric interfaces

Literature review, 14 September 2026. Target: a 60 MeV electron travelling in vacuum at gap \(a\) beside a dielectric, with a later extension to the specified bunch and finite radiator.

**Answer:** FDTD can model this process using Maxwell's equations with a material constitutive law and a consistent moving-charge source. The physical vacuum–dielectric interface is represented inside the grid. An absorbing boundary belongs at the outside of the computational domain.

## Primary literature and usable examples

| Reference | Relevant result | How to use it |
|---|---|---|
| A. F. Oskooi et al., *Meep: A flexible free-software package for electromagnetic simulations by the FDTD method*, CPC **181**, 687–702 (2010). [Author PDF](https://math.mit.edu/~stevenj/papers/OskooiRo10.pdf), [DOI](https://doi.org/10.1016/j.cpc.2009.11.008) | Describes the field/material discretization and moving-source interpolation. Figure 5 demonstrates Cherenkov radiation and the effect of discretizing source motion. | A concrete FDTD starting point. The demonstration concerns a homogeneous medium and a 2D source; it is not a validated 3D single-electron result for our interface geometry. |
| A. Farjadpour et al., *Improving accuracy by subpixel smoothing in the finite-difference time domain*, Optics Letters **31**, 2972–2974 (2006). [Paper](https://doi.org/10.1364/OL.31.002972) | Develops interface averaging that removes the leading interface error for suitable geometries. | Especially relevant to a sloping face. Arbitrary scalar averaging is not equivalent to the derived method. |
| C. Kottke, A. Farjadpour and S. G. Johnson, *Perturbation theory for anisotropic dielectric interfaces, and application to subpixel smoothing of discretized numerical methods*, PRE **77**, 036611 (2008). [Author PDF](https://math.mit.edu/~stevenj/papers/KottkeFa08.pdf), [DOI](https://doi.org/10.1103/PhysRevE.77.036611) | Explains the discontinuity-aware averaging and its extension to anisotropic materials; see Section VI. | Supports a tensor treatment of cut cells. Sharp edges and corners can prevent full second-order convergence. |
| T. Zh. Esirkepov, *Exact charge conservation scheme for Particle-in-Cell simulation with an arbitrary form-factor*, CPC **135**, 144–153 (2001). [DOI](https://doi.org/10.1016/S0010-4655(00)00228-9), [earlier preprint](https://arxiv.org/abs/physics/9901047) | Constructs current deposition satisfying a discrete continuity equation. | A reference for source consistency. Its operators and staggering must match those of the chosen field solver. |
| X. Xu et al., *On numerical errors to the fields surrounding a relativistically moving particle in PIC codes*, JCP **413**, 109451 (2020). [Preprint](https://arxiv.org/abs/1910.13529), [DOI](https://doi.org/10.1016/j.jcp.2020.109451) | Analyses unphysical fields and Cherenkov-like wakes around a relativistic particle, including in vacuum. | Motivates a vacuum control at the actual beam speed and a numerical-dispersion study. A visible cone alone does not establish physical ChDR. |
| S. G. Johnson, *Notes on Perfectly Matched Layers (PMLs)* (2021). [Notes](https://arxiv.org/abs/2108.05348) | Explains absorbing layers through complex coordinate stretching and discusses their limits. | Use for the outer boundary, separately from the material-interface treatment. |

The maintained [Meep moving-source example](https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py) is useful for an independent prototype. Its source units, omitted dimension and charge continuity need checking before using it for absolute electron radiation.

For a closer physical example, Curcio et al. use MAGIC for ChDR from a hollow conical radiator and compare its frequency and gap response with theory. Those calculations are axisymmetric electromagnetic simulations, not a specification of the interface algorithm needed by IPPL. [PRAB 23, 022802, Section III.A](https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.23.022802)

## Representing the physical interface

A useful formulation, written here in SI units, is

\[
\partial_t\mathbf D=\nabla\times\mathbf H-\mathbf J_{\rm free},
\qquad
\partial_t\mathbf B=-\nabla\times\mathbf E,
\qquad \mathbf B=\mu_0\mathbf H.
\]

For a lossless nondispersive proof of concept,

\[
\mathbf D=\epsilon_0\epsilon_r(\mathbf r)\mathbf E,\qquad
\epsilon_r=
\begin{cases}
1&\text{in vacuum},\\
2.13&\text{in the illustrative radiator}.
\end{cases}
\]

The electron's field polarizes the material. No separately prescribed “Cherenkov radiation source” is needed.

With no free surface-charge sheet or free surface current, Maxwell's interface conditions are

\[
[\mathbf E_t]=[\mathbf H_t]=0,\qquad [D_n]=[B_n]=0.
\]

Consequently \(E_n\) generally jumps. A scheme that forces every component of \(\mathbf E\) to be continuous would describe the wrong interface. Bound polarization charge is already represented by the constitutive relation.

In a Yee implementation, material coefficients must be associated with the locations and averaging volumes of the electric components. Begin with a grid-aligned plane; this isolates constitutive and source errors from errors in representing a tilted surface.

For a cell cut by a locally planar interface between two isotropic, nondispersive materials, let \(f\) be its dielectric fraction and use relative permittivities. The corresponding normal and tangential averages are

\[
\epsilon_{\parallel}=f\epsilon_d+(1-f)\epsilon_v,\qquad
\epsilon_{\perp}=\left(\frac{f}{\epsilon_d}
+\frac{1-f}{\epsilon_v}\right)^{-1}.
\]

For unit interface normal \(\hat{\mathbf n}\),

\[
\boldsymbol{\epsilon}_{\rm eff}
=\epsilon_{\parallel}(\mathbf I-\hat{\mathbf n}\hat{\mathbf n}^{T})
+\epsilon_{\perp}\hat{\mathbf n}\hat{\mathbf n}^{T}.
\]

This is the isotropic-material case discussed by Kottke et al.; implementing its off-diagonal entries on a staggered grid requires consistent interpolation. It represents a sharp interface within a cell, not a deliberately thick physical transition layer.

A face at approximately \(47^\circ\) can therefore be represented on a Cartesian grid. The challenge is interface accuracy and convergence, rather than a fundamental restriction to coordinate-aligned geometry.

For dispersive material, introduce a causal polarization model, for example auxiliary equations for Lorentz or Debye responses, with
\(\mathbf D=\epsilon_0\epsilon_\infty\mathbf E+\mathbf P\).
The simple nondispersive averaging recipe is not a complete dispersive implementation. Meep currently smooths the instantaneous permittivity, with explicit limitations for dispersive contributions. [Meep subpixel documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)

## Source, boundaries and diagnostic pitfalls

For the moving electron, require the solver's discrete version of

\[
\frac{\rho^{n+1}-\rho^n}{\Delta t}
+\nabla_h\cdot\mathbf J^{n+1/2}=0.
\]

Together with consistent initial fields, this preserves the free-charge Gauss constraint \(\nabla_h\cdot\mathbf D=\rho\). Starting an already moving charge with zero fields, or abruptly turning its current on and off, produces transients. For a half-space benchmark, initialize consistently or demonstrate that start/end transients do not enter the measurement.

The numerical particle shape must fit within the vacuum gap and resolve the frequencies being measured. Otherwise its deposited source overlaps the dielectric or filters the spectrum. This remains relevant when each simulated particle represents one electron.

Place absorbing layers outside the physical observation region. Their placement and thickness require convergence checks. If dielectric reaches a PML, extend the material consistently; variation along the stretching direction, including certain oblique structures, can invalidate ordinary PML matching. [Meep PML implementation guidance](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/)

At \(\beta=0.99996434\), mesh dispersion can produce a spurious vacuum wake. Test the same source and mesh with \(\epsilon_r=1\), inspect its spectrum, and refine the grid. Numerical radiation from one prescribed charge should also be distinguished from collective numerical Cherenkov instability in a beam.

Measure radiation separately from the electron's bound near field. Compare complex field spectra at specified locations or outgoing flux away from the trajectory. Align electric and magnetic fields in space and time before calculating Poynting flux. A scalar difference of two near-field powers is not generally the power of the difference field.

## Implications for the current FEL mini-app

The following observations come from a read-only inspection of the local source on 14 September 2026, rather than from the cited literature:

| Current code | Consequence for ChDR |
|---|---|
| [Solver selection](/Users/adelmann/git/ippl/demos/fel/datatypes.h:50) uses NonStandardFDTDSolver with absorbing Mur boundaries. | Material transmission and outer absorption need separate implementation and validation. |
| [Nonstandard update](/Users/adelmann/git/ippl/src/MaxwellSolvers/NonStandardFDTDSolver.hpp:28) evolves four-potential fields with a vacuum stencil; its initialization sets \(\Delta t=h_z\) in normalized units. | A variable dielectric is more than a geometry mask or replacement of \(c\) by \(c/n\). The constitutive/interface equations and stability analysis must be consistent with the formulation. |
| [Current deposition](/Users/adelmann/git/ippl/demos/fel/FreeElectronLaserManager.h:147) already calls assemble_current_collocated; charge density is conditional on space_charge. | Audit discrete continuity and Gauss's law for the selected material solver. Charge-conserving deposition is already present in the design, but compatibility with a new staggering cannot be assumed. |
| [Frame initialization](/Users/adelmann/git/ippl/demos/fel/FreeElectronLaserManager.h:47) ties a Lorentz boost to bunch and undulator parameters. | Prefer a laboratory-frame ChDR mode with a stationary radiator. A stationary isotropic dielectric in the lab cannot be represented as the same stationary scalar material after a nonzero boost. |

Two possible formulations deserve evaluation: a direct \(\mathbf D,\mathbf B\) Maxwell solver with a material response, or a potential solver coupled consistently to polarization charge and current. The latter needs \(\rho_{\rm bound}=-\nabla\cdot\mathbf P\) and \(\mathbf J_{\rm bound}=\partial_t\mathbf P\), together with a stable, gauge-consistent coupling. The review does not establish which is the smaller implementation change.

## Recommended validation sequence

1. **Passive interface:** launch plane waves at a flat vacuum–dielectric boundary; compare Fresnel reflection/transmission for both polarizations and verify lossless energy balance.
2. **Vacuum electron:** prescribed 60 MeV trajectory, consistent charge/current and initialization; measure unwanted radiation and Gauss/continuity residuals.
3. **Infinite-interface approximation:** compare the complex spectra and gap dependence with the independent half-space calculation from the companion review. Check source-shape, mesh, box-size and absorbing-boundary convergence.
4. **Finite body:** add leading/trailing edges and then the tilted extraction face. Compare angular spectra and polarization; refine the interface representation.
5. **Bunch response:** apply the longitudinal and transverse distributions in the selected band, once the single-electron response is established.

For a conventional explicit Yee solver, the vacuum region imposes the usual condition
\[
c\Delta t\sqrt{h_x^{-2}+h_y^{-2}+h_z^{-2}}\leq1.
\]
Material auxiliary equations and alternative stencils may impose additional conditions. This is not the time-step rule for the existing nonstandard potential solver. Resolve the shortest material wavelength and the relevant gap/source scales; no fixed number of cells alone certifies an accurate spectrum.

These are proposed checks, not tests already run. No solver files or existing LaTeX content were changed. See the [single-electron review](/Users/adelmann/git/ippl/demos/fel/research/chdr_single_electron.md) and [BibTeX bibliography](/Users/adelmann/git/ippl/demos/fel/research/chdr_references.bib).
