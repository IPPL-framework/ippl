# Single-electron Cherenkov diffraction radiation: analytical and reduced models

Literature review, 14 September 2026. Geometry confirmed by the user: the electron remains in vacuum, moving parallel to the radiator at gap \(a\). It may enter and leave the longitudinal region beside a finite radiator; it does not enter the dielectric.

**Answer:** an infinite planar interface has an analytical spectral solution requiring only a one-dimensional numerical integral at each frequency. This provides a useful independent benchmark for the PIC calculation. A finite radiator needs additional treatment of its ends, extraction faces and diffraction; a spatially one-dimensional simulation cannot represent the complete setup.

## Most relevant primary literature

| Reference | What it provides | Applicability and limits |
|---|---|---|
| A. V. Tyukhtin, S. N. Galyamin and V. V. Vorobev, *Radiation of Charge Moving along Face of Inverted Prism*, arXiv:2105.01111, submitted 2021, revised 2022. [Paper](https://arxiv.org/abs/2105.01111) | Section III, Eqs. (5)–(8), explicitly gives the exact dielectric-half-space field as a single transverse-wave-number integral. | Best starting point for the planar benchmark. Its later finite-prism construction uses an aperture approximation; that part is not an exact solution for an arbitrary finite body. |
| M. V. Shevelev and A. S. Konkov, *Peculiarities of the generation of Vavilov–Cherenkov radiation induced by a charged particle moving past a dielectric target*, JETP **118**, 501–511 (2014). [DOI](https://doi.org/10.1134/S1063776114030182), [author PDF](https://portal.tpu.ru/SHARED/e/EKWINUS/Research/Tab1/Shevelev_Konkov_JETP.pdf) | A polarization-current calculation for a particle outside a prismatic dielectric; includes radiation extraction and target orientation. | A useful finite-target model. Its Fresnel extraction assumes an exit face much larger than the wavelength and omits multiple internal rereflections; see p. 510. The target is infinite in one transverse direction. Their gap is \(b\); their \(a\) is a target dimension. |
| A. Curcio et al., *Noninvasive bunch length measurements exploiting Cherenkov diffraction radiation*, PRAB **23**, 022802 (2020). [Paper](https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.23.022802) | Theory, electromagnetic simulations and measurements of coherent ChDR. Section III.A compares MAGIC simulations with an analytical radiator transfer function. | The simulation uses an axisymmetric hollow cone and a modulated beam. It supports the modelling strategy, but its geometry and normalization differ from our planar, single-electron benchmark. |

These references were selected for their direct connection to the stated geometry, rather than for radiation from a particle crossing a material interface. This is a focused review, not an exhaustive bibliography.

## What “one-dimensional computation” means here

The following reduction is derived from the assumed planar geometry. To avoid ambiguity about the drawing's axis names, let \(s\) denote the beam direction, \(u\) the surface normal and \(w\) the other transverse coordinate:

\[
u>0:\ \text{vacuum},\qquad u<0:\ \text{dielectric},\qquad
\mathbf r_e(t)=(vt,a,0).
\]

For an electron,

\[
\rho=-e\,\delta(s-vt)\delta(u-a)\delta(w),\qquad
\mathbf J=v\rho\,\hat{\mathbf s}.
\]

With time convention \(e^{-i\omega t}\), the source fixes
\(k_s=\omega/v\). Fourier-transforming in \(w\) leaves a boundary-value problem in \(u\) for each \((\omega,k_w)\). In each homogeneous region the normal wave numbers are

\[
\kappa_0=\sqrt{k_w^2+\frac{\omega^2}{\gamma^2v^2}},
\qquad
k_{u,d}^{\,2}=\epsilon_r(\omega)\frac{\omega^2}{c^2}
-\frac{\omega^2}{v^2}-k_w^2 ,
\]

assuming \(\mu_r=1\). The vacuum field is evanescent normal to the interface. Choose the dielectric branch to carry energy away from the interface, or decay into passive lossy material.

For each transverse wave number, solve the two polarization matching problems at the interface. The remaining inverse transform is one integral over \(k_w\); reconstructing a time waveform also requires integration over frequency. The full transverse integral includes both propagating and evanescent dielectric components.

This is a **spectral reduction of a three-dimensional point-source problem**. Ordinary 1D FDTD cannot resolve both the gap and the propagation direction. Ordinary Cartesian 2D FDTD represents a source uniform in the omitted direction—a line charge—with a per-unit-length normalization.

The explicit coefficients and Fourier normalization needed for implementation are given in the Tyukhtin paper cited above. Its formulas use Gaussian electromagnetic units; convert the complete source and field normalization consistently before comparing an absolute amplitude with SI or IPPL units.

## Consequences for our parameters

For 60 MeV kinetic energy, direct calculation gives

\[
\gamma=1+\frac{60}{0.51099895}=118.4171,\qquad
\beta=0.99996434.
\]

With an illustrative lossless, nondispersive \(\epsilon_r=2.13\),
\(n=1.45945\). A propagating dielectric mode exists when

\[
k_w^2<\frac{\omega^2}{c^2}
\left(\epsilon_r-\beta^{-2}\right).
\]

Thus the threshold is \(n\beta>1\), and phase matching gives the internal angle

\[
\cos\theta_{\rm Ch}=\frac{1}{n\beta},
\qquad \theta_{\rm Ch}=46.7476^\circ.
\]

These are internal wave-vector angles for the homogeneous planar benchmark, not a prediction of the camera angle after extraction.

At each \(k_w\), translating the trajectory away from the interface supplies a field factor \(e^{-\kappa_0a}\), and the corresponding modal energy has the factor \(e^{-2\kappa_0a}\). An integrated observable combines many \(k_w\) values, so it need not follow one exponential in \(a\). At \(k_w=0\), the field coupling length is

\[
\kappa_0^{-1}=\frac{\gamma\beta\lambda_0}{2\pi}.
\]

For example, it is approximately \(9.4\,\mu\mathrm m\) at \(\lambda_0=500\) nm. This is a coupling scale, not a hard cutoff or a prescription for the computational box.

For the previously specified Gaussian bunch, \(\sigma_s=3\) mm gives
\(\sigma_t=\sigma_s/v=10.0073\) ps. If the electron responses differ only by arrival time, linear superposition and independent arrival statistics give

\[
\left\langle\frac{dW_N}{d\omega}\right\rangle
=\left[N+N(N-1)|F(\omega)|^2\right]
\frac{dW_1}{d\omega},\qquad
|F|^2=e^{-(\omega\sigma_t)^2},
\quad N=\frac{5\,\mathrm{nC}}{e}\simeq3.12\times10^{10}.
\]

This expression follows by expanding the squared sum of the single-electron amplitudes. It assumes prescribed trajectories and a common transverse coupling; a distribution of gaps needs an amplitude average over those trajectories.

The squared form factor reaches \(e^{-1}\) at \(15.904\) GHz. **That is not automatically the maximum of the detected spectrum.** The single-electron spectrum, finite-radiator response and collection optics also contribute. The dielectric constant must be specified over the selected band; an optical value should not be carried into a microwave calculation without material data.

## The finite radiator and the approximately 47-degree geometry

An infinite interface is invariant along the beam. Consequently, the coupled modes have \(k_s=\omega/v>\omega/c\); their tangential wave number exceeds the vacuum propagating-wave limit. They cannot emerge as propagating radiation through that same infinite planar interface. An inclined extraction face or a finite edge changes that constraint. This is a direct phase-matching consequence of the equations above.

It follows that a rectangular radiator can demonstrate excitation inside the dielectric, while producing a different external angular distribution from the prism. Distinguish the angle of the beam-facing surface, the extraction-face normal and the internal Cherenkov angle.

Entering the longitudinal region beside a finite body breaks translation symmetry along \(s\). Edge transients, interference and internal reflections then require a finite-geometry calculation. If the material remains invariant in \(w\), a Fourier transform in that coordinate produces a family of two-dimensional field problems (“2.5D”). A radiator of finite width generally requires 3D treatment.

The prism aperture models are attractive when relevant face dimensions greatly exceed wavelength. A centimetre-scale radiator at the bunch's approximately 16 GHz roll-off has dimensions comparable to the roughly 19 mm vacuum wavelength; the large-face assumption must therefore be checked rather than presumed.

## Recommended benchmark

Implement the infinite-half-space spectral calculation first, with a prescribed electron trajectory and an explicit frequency band. Compare complex field components and polarization, their gap dependence, and the dielectric propagation angle. For an infinite track, compare a spectral quantity per unit path length rather than a finite total radiated energy.

Then introduce finite leading and trailing faces in the numerical model. Compare external radiation with a finite-target analytical approximation only within that approximation's stated regime. The remaining physical inputs are gap \(a\), radiator dimensions, actual \(\epsilon_r(\omega)\), transverse beam profile, and the desired observable: local fields, energy loss or collected spectral energy.

No field simulation was run for this review. See the companion [FDTD/interface review](/Users/adelmann/git/ippl/demos/fel/research/chdr_fdtd_interfaces.md) and [BibTeX bibliography](/Users/adelmann/git/ippl/demos/fel/research/chdr_references.bib).
