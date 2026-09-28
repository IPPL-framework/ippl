# One-dimensional spectral ChDR prototype

Run a planar vacuum/dielectric reference calculation while varying the vacuum gap **a**, electron **kinetic energy**, and Gaussian **rms pulse duration in ps**. No manual choice of transverse wave number is required.

Each frequency and transverse Fourier mode is a one-dimensional boundary-value problem in x. Analytical matching is followed by numerical integration over the transverse modes. This produces the field and radiated spectrum of a 3D point electron near an infinite interface. It is not a time-stepping 1D PIC model or a finite-prism calculation.

## Run

From this directory:

```sh
~/.venv-h6/bin/python chdr_1d.py
~/.venv-h6/bin/python chdr_1d.py --gap-mm 0.5 --energy-mev 60 --pulse-ps 10
```

Defaults: **a = 1 mm (illustrative), kinetic energy = 60 MeV, rms duration = 10 ps, charge magnitude = 5 nC, epsilon_r = 2.13**. The electron travels in vacuum along +z at `(x,y)=(a,0)`. Vacuum is x>0, dielectric x<0. The charge sign is negative. The 10 ps default corresponds to about 3 mm rms length at 60 MeV; when energy changes, the entered time duration stays fixed and the derived spatial length changes.

`--pulse-ps` is **rms**, not FWHM. For a Gaussian, FWHM = 2.35482 times rms. For example, a 10 ps FWHM bunch has `--pulse-ps 4.24661`.

Parameter scans accept lists and run their Cartesian product:

```sh
~/.venv-h6/bin/python chdr_1d.py --gap-mm 0.2 1 3 --no-fields --output output/gap_scan
~/.venv-h6/bin/python chdr_1d.py --energy-mev 0.1 1 10 60 --no-fields --output output/energy_scan
~/.venv-h6/bin/python chdr_1d.py --pulse-ps 3 10 30 --no-fields --output output/pulse_scan
~/.venv-h6/bin/python chdr_1d.py --gap-mm 0.5 2 --pulse-ps 5 20 --no-fields --output output/combined_scan
```

Pulse scans reuse the single-electron calculation. Use distinct output directories to retain runs with different settings; rerunning a case in the same directory replaces its files.

Other options:

```sh
~/.venv-h6/bin/python chdr_1d.py --help
~/.venv-h6/bin/python chdr_1d.py --charge-nc 5 --epsilon-r 2.13 --fmin-ghz 0.2 --fmax-ghz 200 --points 120
~/.venv-h6/bin/python chdr_1d.py --probe-depth-mm 1 --profile-frequency-ghz 16 --profile-depth-mm 20
```

The frequency interval is an **analysis band**, not a monochromatic source. The default band is 0.2-200 GHz. Change it for substantially different pulse durations. The reported band integral is only over this interval, using the sampled spectrum; refine `--points` to check that integral separately from the adaptive transverse quadrature.

For selected optical or infrared wavelengths, use vacuum wavelengths directly. This example evaluates the fixed-total-charge, independent-particle **shot-noise fluctuation spectrum** at 0.5 micrometres for the stated 1 nC, 1 mm thought experiment. At this optical frequency the smooth bunch form factor is negligible, so it equals the usual `N*W1` self term. The 10 micrometre gap is chosen only to avoid the overwhelming 1 mm-gap suppression discussed below.

```sh
~/.venv-h6/bin/python chdr_1d.py \
  --charge-nc 1 --energy-mev 60 --pulse-ps 3.33576 --gap-mm 0.01 \
  --wavelength-um 0.5 --uniform-box-mm 1 1 1 \
  --no-fields --no-plots --output output/shot_noise_0p5um
```

`--wavelength-um` replaces the frequency range and point-count options; duplicate wavelength samples are coalesced. `--uniform-box-mm X Y Z` only reports the density scale in the terminal and `parameters.json`; it does not introduce individual particles, a spatial lattice, or a transverse bunch distribution into the radiation calculation.

Dependencies: Python 3.10+, NumPy 2+, SciPy, Matplotlib. The local `~/.venv-h6` environment already provides them. Tests use the standard-library `unittest` module.

## Outputs

The default directory is `output/` beside the script. Numerical arrays use SI units.

- `spectra.png` / `.svg`: comparisons of the single-electron spectrum, expected bunch spectrum and Gaussian squared form factor. Plot spectra use GHz or THz to match the sampled band; CSV spectra are per Hz.
- Each `case_.../spectrum.csv`: positive-frequency **energy entering the dielectric per metre of electron trajectory and per Hz**, in J/(m Hz); vacuum wavelength; the least-evanescent vacuum coupling length; a separately calculated work-on-electron check; the conventional self and coherent cross terms; the fixed-`N` shot-noise fluctuation spectrum; and the total bunch spectrum.
- `case_.../probe_fields.csv`: complex time-Fourier E and H at `(x,y,z)=(-probe_depth,0,0)`, for one electron and the mean Gaussian bunch field. E has units V s/m, H has units A s/m. Mean fields are not the incoherent noise field, and their squared magnitude is not the full ensemble-averaged intensity.
- `case_.../field_profile.csv` and `.png` / `.svg`: complex single-electron fields versus x at the selected profile frequency, at y=z=0. The profile includes both sides of the interface and stops before the electron plane. Shading marks the dielectric.
- `case_.../parameters.json`: all settings, gamma, beta, electron count, equivalent rms spatial length, internal angle, form-factor scale, sampled-band integral when at least two distinct frequencies were used, and flux/work agreement.

Use `--no-fields` for faster spectrum-only scans or `--no-plots` for numerical output only. Complex fields can also be obtained from the importable `reconstructFields` function at arbitrary broadcastable coordinate arrays. This first implementation excludes probe points on x=a, where a separate treatment of the source-plane integral would be needed.

The default plots do not show a camera signal. For an infinite track the total emitted energy is infinite; the useful finite observable is energy **per path length**. Finite-prism extraction, leading/trailing edges, detector optics, material loss/dispersion, transverse bunch size and trajectory feedback are outside this initial model. Epsilon_r=2.13 is an illustrative constant, not a measured material model.

## Density scale and Poisson shot noise

For a charge magnitude `Q` uniformly occupying a rectangular volume `V`, the physical electron count and the commonly quoted cube-root volume-per-electron scale are

\[
N=|Q|/e,\qquad n=N/V,\qquad d_{\mathrm{cube}}=n^{-1/3}.
\]

For 1 nC in `(1 mm)^3`, the diagnostic reports

\[
N=6.2415\times10^9,\qquad
n=6.2415\times10^{18}\ \mathrm{m}^{-3},\qquad
d_{\mathrm{cube}}=0.543\ \mathrm{\mu m}.
\]

This arithmetic is correct, but it does **not** predict a shot-noise line at a vacuum wavelength of 0.543 micrometres. Independent random positions have broadband microscopic noise; a Poisson point process has structure factor one until physical correlations modify it. A perfectly regular lattice has no ordinary Poisson shot noise; it instead has reciprocal-lattice (Bragg) features. The reported diagnostic also gives the Poisson mean nearest-neighbour distance, which is about `0.554*d_cube`, to make this distinction explicit.

The radiation-relevant longitudinal statistic for identical transverse tracks is the bunching factor

\[
b(\omega)=\frac1N\sum_{j=1}^N e^{i\omega t_j},\qquad
\left\langle |b(\omega)|^2\right\rangle=
\frac1N+\left(1-\frac1N\right)|F(\omega)|^2.
\]

The prototype conditions on a fixed physical count `N=|Q|/e`. Its conventional self term is `bunch_incoherent=N*W1`, while the field-fluctuation spectrum about the mean bunch field is

\[
\left\langle |E_N-\langle E_N\rangle|^2\right\rangle
=N\left(1-|F|^2\right)|E_1|^2,
\qquad
W_{\mathrm{noise}}=N\left(1-|F|^2\right)W_1.
\]

The latter is exported as `bunch_fixed_N_shot_noise_fluctuation...`. It approaches `N*W1` only above the smooth bunch form-factor bandwidth. At 0.5 micrometres it does, to machine precision, so the spectrum shown here is the desired independent-particle shot-noise baseline. It is broadband; its shape comes from the single-electron ChDR response, the gap, material dispersion, finite radiator and detector. A hypothetical ensemble with a Poisson-distributed total count has different zero-frequency statistics and is not the fixed-charge beam model used here. The present reduction does not infer a three-dimensional noise spectrum from `d`: it assigns every electron the same `(x,y)` track. Projecting all 6.24 billion electrons onto a 1 mm longitudinal line would give a fictitious mean z separation of only 0.160 pm, which is another reason not to use `d_cube` as a one-dimensional particle spacing.

At 60 MeV, `gamma=118.42`. For a vacuum wavelength `lambda0`, the slowest-decaying planar vacuum mode has coupling length

\[
\ell_\perp=\kappa_0^{-1}=\frac{\gamma\beta\lambda_0}{2\pi}.
\]

At `lambda0=0.5 um`, this is only `9.42 um`. With the illustrative 1 mm gap, even that modal energy is suppressed by `exp(-2a/ell_perp)=6.6e-93`; all other transverse modes are suppressed more strongly. A useful optical study therefore needs a micrometre-scale gap, a physically compatible transverse beam distribution, measured complex `epsilon(omega)`, and eventually a finite-radiator/extraction model. The current constant `epsilon_r=2.13` must not be treated as an optical material model.

## Equations and normalization

Write the time/y Fourier transform with forward factor 1 and inverse factor `(2*pi)^-2`:

\[
E(x,y,z,t)=\frac{1}{(2\pi)^2}\int d\omega\,dk_y\,
\widehat E(x,k_y,\omega)e^{ik_y y+i\omega z/v-i\omega t}.
\]

The invariant point-electron charge is q=-e, so the transformed sources are

\[
\widehat\rho=(q/v)\delta(x-a),\qquad
\widehat J_z=q\delta(x-a).
\]

No extra gamma factor multiplies q. With `k_z=omega/v`,

\[
\kappa=\sqrt{k_y^2+\omega^2/(\gamma^2v^2)},\qquad
q_d=\sqrt{\epsilon_r(\omega/c)^2-k_z^2-k_y^2}.
\]

Choose Re(q_d)>=0, Im(q_d)>=0; the transmitted field varies as `exp(-i*q_d*x)` and propagates or decays into x<0. The vacuum Lorenz potential is

\[
\widehat\phi=\frac{q}{2\epsilon_0v\kappa}e^{-\kappa|x-a|},\qquad
\widehat A_z=(v/c^2)\widehat\phi.
\]

Define the unit tangential wave direction `t=(0,k_y,k_z)/sqrt(k_y^2+k_z^2)` and `s=x_hat cross t`. TE amplitudes are electric components along s; TM amplitudes are magnetic components along s. Matching tangential E and H gives the reflected amplitude factors

\[
r_{\mathrm{TE}}=\frac{i\kappa-q_d}{i\kappa+q_d},\qquad
r_{\mathrm{TM},H}=\frac{i\epsilon_r\kappa-q_d}{i\epsilon_r\kappa+q_d}.
\]

Transmitted TE-electric and TM-magnetic amplitudes are `(1+r)` times the incident amplitudes. Normal components follow from Maxwell's equations. Note the TM coefficient here refers to **magnetic amplitude**, which avoids ambiguity about the sign of an electric-polarization basis under reflection.

Integrating Poynting flux over time and y and using Parseval's identity gives the positive-frequency spectrum

\[
\frac{d^2W_1}{dz\,df}=\frac1\pi\int dk_y\,
-\operatorname{Re}(\widehat{\mathbf E}_d\times\widehat{\mathbf H}_d^*)_x.
\]

An independent check is the induced longitudinal field acting on the electron:

\[
\frac{d^2W_1}{dz\,df}=-\frac q\pi\int dk_y\,
\operatorname{Re}\widehat E_{z,\mathrm{ref}}(x=a,k_y,2\pi f).
\]

For a lossless positive dielectric, only `|k_y| < (omega/c)*sqrt(epsilon_r-beta^-2)` carries normal radiative flux. The substitution `k_y=cutoff*sin(theta)` regularizes the integration endpoints. Below threshold both radiative spectra are zero. **Probe fields include all evanescent modes**, using an adaptive infinite-interval integral; they remain nonzero below threshold and in vacuum. Source-free Maxwell relations and interface conditions are checked independently in the tests.

For N=|Q|/e independent Gaussian arrival times with rms duration sigma_t and identical transverse trajectories,

\[
|F|^2=e^{-(2\pi f\sigma_t)^2},\qquad
\left\langle\frac{d^2W_N}{dz\,df}\right\rangle=
\left[N+N(N-1)|F|^2\right]\frac{d^2W_1}{dz\,df}.
\]

The CSV separates the `N` self term, the `N(N-1)` interference term, and the fixed-`N` fluctuation term above. The mean bunch field is N times the single-electron field times `exp[-(omega*sigma_t)^2/2]`.

For a particular noisy realization, rather than this ensemble mean, one would sample the complex bunching factor for the detector's resolved spatio-temporal mode. In the high-frequency regime `(N-1)*|F|^2 << 1` and at large N, `sqrt(N)*b` is a zero-mean circular complex Gaussian and `N*|b|^2` has an exponential intensity distribution with mean one. A random spectral trace must retain the correlation between nearby frequencies; drawing unrelated random factors at every output point would not represent one physical bunch. This is intentionally left out of the present deterministic reference.

## Validation and comparison with the 3D solver

Run:

```sh
~/.venv-h6/bin/python -m unittest -v test_chdr_1d.py
```

Tests cover interface continuity, homogeneous-region Maxwell relations, outgoing/decaying branches, vacuum and subthreshold limits, modal gap dependence, flux/work equality, frequency-length scaling, quadrature convergence, Gaussian/charge scaling and CLI sweeps. Reconstructed vacuum fields are compared against the independent Bessel-function solution `phi_tilde=q*K0[omega*r/(gamma*v)]/(2*pi*epsilon0*v)`; this also checks absolute Fourier normalization.

Compare complex fields in the 3D calculation before comparing power. Match the laboratory frame, trajectory, material, Fourier convention and sampling times/locations. Account for finite domain and startup transients. A grid-deposited source has a finite shape: average the analytical reference over that shape or demonstrate mesh convergence away from the source. The integrated radiation test should compare energy per trajectory length, with the same frequency band and sufficient convergence against finite-track effects. Passing these checks does not by itself validate the finite radiator or its optical extraction.

## Literature

The spectral half-space construction is described in Section III of A. V. Tyukhtin, S. N. Galyamin and V. V. Vorobev, *Radiation of Charge Moving along Face of Inverted Prism*, [arXiv:2105.01111](https://arxiv.org/abs/2105.01111). The paper uses Gaussian units and the opposite material-side orientation; this script instead uses the separately derived SI potentials and interface coefficients above. No large-prism aperture approximation is used here.

The moving-source discussion in Oskooi and Johnson, Chapter 4 of the 2013 volume co-edited by Taflove, is useful for the subsequent FDTD comparison: [open chapter](https://arxiv.org/abs/1301.5366). Adaptive integration uses SciPy's [`quad_vec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html).

For statistically correct electron-beam noise initialization, see N. J. M. Penman and B. W. J. McNeil, *The physics of SASE FELs*, [Optics Communications **90** (1992), 82-84](https://doi.org/10.1016/0030-4018(92)90333-M). MITHRA's related slice-based noise discussion is documented in [MITHRA 2.0](https://arxiv.org/abs/2009.13645); its FEL-resonant initialization is not directly a ChDR transverse-noise model.
