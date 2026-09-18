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

Dependencies: Python 3.10+, NumPy 2+, SciPy, Matplotlib. The local `~/.venv-h6` environment already provides them. Tests use the standard-library `unittest` module.

## Outputs

The default directory is `output/` beside the script. Numerical arrays use SI units.

- `spectra.png` / `.svg`: comparisons of the single-electron spectrum, expected bunch spectrum and Gaussian squared form factor. Plot spectra are per GHz; CSV spectra are per Hz.
- Each `case_.../spectrum.csv`: positive-frequency **energy entering the dielectric per metre of electron trajectory and per Hz**, in J/(m Hz); a separately calculated work-on-electron check; quadrature error estimate; incoherent, coherent cross-term and total bunch spectra.
- `case_.../probe_fields.csv`: complex time-Fourier E and H at `(x,y,z)=(-probe_depth,0,0)`, for one electron and the mean Gaussian bunch field. E has units V s/m, H has units A s/m. Mean fields are not the incoherent noise field, and their squared magnitude is not the full ensemble-averaged intensity.
- `case_.../field_profile.csv` and `.png` / `.svg`: complex single-electron fields versus x at the selected profile frequency, at y=z=0. The profile includes both sides of the interface and stops before the electron plane. Shading marks the dielectric.
- `case_.../parameters.json`: all settings, gamma, beta, electron count, equivalent rms spatial length, internal angle, form-factor scale, sampled-band integral and flux/work agreement.

Use `--no-fields` for faster spectrum-only scans or `--no-plots` for numerical output only. Complex fields can also be obtained from the importable `reconstructFields` function at arbitrary broadcastable coordinate arrays. This first implementation excludes probe points on x=a, where a separate treatment of the source-plane integral would be needed.

The default plots do not show a camera signal. For an infinite track the total emitted energy is infinite; the useful finite observable is energy **per path length**. Finite-prism extraction, leading/trailing edges, detector optics, material loss/dispersion, transverse bunch size and trajectory feedback are outside this initial model. Epsilon_r=2.13 is an illustrative constant, not a measured microwave material model.

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

The CSV separates the N term and the N(N-1) interference term. The mean bunch field is N times the single-electron field times `exp[-(omega*sigma_t)^2/2]`.

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
