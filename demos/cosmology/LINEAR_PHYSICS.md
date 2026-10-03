# Linear CDM physics and data contract

The linear cosmology executable evolves one collisionless, cold matter species in a
periodic cube. It follows the conventions in Zarija Lukic's supplied initializer
(`Cosmology.cpp`, `Initializer.cpp`, `README`), whose documentation describes its MC2
heritage. This is an explicit flat, Gaussian, radiation-free Lambda-CDM subset:
`Omega_nu=Omega_r=f_NL=0`, `w_de=-1`, and `Omega_Lambda=1-Omega_m`.
`Omega_bar` weights the input transfer function; there is no hydrodynamic baryon species.
The implementation does not claim validation against HACC's production evolution code.

## Units and evolution

Positions `x` and box size `L` are comoving Mpc/h; scale factor `a=1/(1+z)`.
Time is `tau=H0*t`, `E(a)=H(a)/H0=sqrt(Omega_m/a^3+1-Omega_m)`.
The stored momentum is `p=a^2 dx/dtau`, in Mpc/h. The physical peculiar velocity is
`v_pec=100 p/a` km/s. The factor 100, rather than `100*h`, accounts for the Mpc/h
length convention.

The density contrast is `delta=rho/<rho>-1`. The code solves

```
laplacian(phi0) = (3/2) Omega_m delta
F = -gradient(phi0)
dp/da = F / (a^2 E(a))
dx/da = p / (a^3 E(a))
```

Consequently, a kick multiplies a fixed force by `integral da/(a^2 E)` and a drift
multiplies fixed momentum by `integral da/(a^3 E)`. `phi0` includes neither `1/a`
nor dimensional `H0^2`; mixing this definition with the dimensional peculiar
potential would introduce an erroneous scale-factor factor. The periodic DC mode
has zero potential and zero force. Both cosmological integrals use host adaptive
quadrature; mesh and particle kernels do not evaluate these integrals.

The growing mode is

```
D_raw(a) = (5 Omega_m/2) E(a) integral_0^a da' / (a'^3 E(a')^3)
D(a) = D_raw(a) / D_raw(1)
f(a) = d ln D / d ln a
```

This is the growing solution of the same radiation-free Lambda-CDM growth ODE as
Zarija's implementation, evaluated through its exact integral rather than the
legacy finite-start ODE initialization. For `Omega_m=1`, `D=a` and `f=1`.
For a z=0 displacement `psi0`, 1LPT has `x=q+D psi0` and
`p=a^2 E(a) f(a) D(a) psi0`. The sine input's `amplitude` is the density contrast
amplitude at the **initial** redshift, so its displacement is already scaled by
`D(aInitial)`. It is ignored by Gaussian and uniform initial conditions.

Particles carry unit mass internally, with exactly one particle per mesh cell
before displacement; CIC deposits a mean cell value of one. Thus subtracting one
gives `delta` without another cell-volume factor. A physical particle represents
`Omega_m * rho_crit,0 * L^3 / np^3`; unit masses are not solar masses.
The source initializer's generic ASCII velocity is coordinate `dx/dt` expressed
in km/s, whereas its Nyx output applies an additional factor `a`. Its generic
velocity therefore maps to this canonical momentum as `p=a^2*v_ascii/100`, not
the physical-peculiar-velocity conversion above. Direct legacy-file ingestion is
not implemented here.

## Discrete initialization, force, and integration

The IPPL complex transform uses a forward `1/np^3` normalization and an unscaled
inverse. Gaussian density coefficients have ensemble variance `P(k)/L^3`.
SplitMix64 hashes the configured seed and canonical global conjugate-mode index;
Box-Muller then produces the Gaussian pair. The same global mode receives the
same coefficient on every MPI decomposition and thread schedule. Conjugate pairs
are constructed explicitly; DC and all IC Nyquist planes are zero.

Particles start at cell centers `q=(index+1/2)*L/np`. Fourier coefficients include
the corresponding half-cell phase. The displacement is the inverse transform of
`i*k*delta_hat/k^2`; all three components use the same initial density field.
Global particle IDs and Lagrangian positions travel with particles during migration.
The application wraps coordinates with `x -= L*floor(x/L)` before every migration,
including initialization. This avoids the legacy particle boundary functor's
large-overshoot and exact-endpoint behavior; its particle BC is disabled while
the field layout remains periodic. A forced-migration test checks multi-box
crossings, exact endpoints, ownership, unique IDs, and all migrated attributes.

Gravity uses CIC scatter, a spectral Poisson force, and CIC gather. Its denominator
retains the full `k^2`; only a differentiated Nyquist component is zero. There is
no CIC-window deconvolution, force softening prescription beyond the mesh, or
short-range force. Resolved linear modes therefore approach continuum growth
with spatial refinement, rather than being exact at finite mesh spacing.
Kick-drift-kick steps are uniform in `ln(a)`, with the geometric midpoint separating
the kicks. Timestep convergence is assessed at fixed mesh spacing, independently
of spatial convergence. The method is second order in time; it does not make
ordinary mechanical energy constant in an expanding background.

Distributed fields and particles stay in Kokkos's default execution/memory space
for initialization, force evaluation, and integration. Host work evaluates growth
quadratures and constructs a radial `P(k)` lookup copied once to execution memory.
That lookup replicates O(np^2) storage per rank: acceptable locally, not a demonstrated
exascale design. Snapshots explicitly mirror positions, momenta, and IDs to host.
MPI reductions and CIC atomics can alter floating-point summation order; rank/thread
agreement is tested with stated numerical tolerances, not assumed bitwise identity.
Only the OpenMP CPU backend is qualified by this local test campaign.

## Spectrum and transfer functions

The z=0 spectrum is `P(k)=A k^n_s T(k)^2`, with `k` in h/Mpc and `P` in (Mpc/h)^3.
`TFFlag=4` uses precisely Zarija's BBKS shape `q=k/(Omega_m*h)`, with coefficients
2.34, 3.89, 16.1, 5.46, and 6.71. This analytic option is a reference CDM shape
without baryon acoustic oscillations.

`TFFlag=0` reads the first three columns of a CMBFAST-style table: `k`, `T_cdm`,
`T_baryon`; remaining numeric columns may be present. The total matter transfer is
`[(Omega_m-Omega_bar) T_cdm+Omega_bar T_baryon]/Omega_m`, normalized to one at the
first row, and linearly interpolated in `k` as in Zarija. Every row is normalized,
including the first row (the legacy code leaves that first entry unnormalized).
Below the first row `T=1`; requests beyond the last row fail. The table must cover
the 3D mesh's corner wave number `sqrt(3)*pi*np/L`, avoiding extrapolation.
`transfer_file` is relative to the parameter file's directory; output is relative
to the run directory. The table must correspond to the configured cosmology.

Normalization is fixed by

```
sigma_8^2 = integral_0^kmax dk k^2 P(k) W(8k)^2 / (2 pi^2)
W(x) = 3 [sin(x)-x cos(x)] / x^3
```

For compatibility with Zarija, `kmax=10 h/Mpc` for BBKS and the final table `k` for
CMBFAST. This cutoff is part of the numerical contract, not the formal infinite
integral definition of sigma8. Integration uses a dense log-k Simpson grid, a stable
small-argument top-hat series, and an analytic negligible `k<1e-8` tail. The tests
cross-check sigma8 with an independent linear-k midpoint rule. A small periodic
box's realized variance need not equal the ensemble sigma8: finite volume, mesh
cutoffs, and realization variance change it. At time `a`, `P(k,a)=D(a)^2 P(k)`.

## Input restrictions and local scope

`np` is the number of cells and particles **per dimension**; the global particle
count is exactly `np^3`. It must be even and at least four. `nt` is the positive
number of evolution steps. `seed` accepts the full unsigned 64-bit range, fixing
the legacy parser's 32-bit seed truncation. Unknown, duplicate, malformed, and
unsupported parameters fail before initialization. The supported primordial index
range is `0<n_s<2`; start and final redshifts obey `z_in>z_fi>=0`.

The provided Gaussian and sine cases start at z=49 and finish at z=9. They target
linear-growth validation at modest local memory use. These runs do not establish
nonlinear force accuracy, halo statistics, 2LPT accuracy, or exascale scalability.
The uniform case tests the periodic zero-force state. Tiny diagnostic particle
CSV snapshots can be disabled with `write_particles=false`; they are not a
production checkpoint format.

Host tests cover independent growth-ODE agreement, analytic Einstein-de Sitter
kick/drift factors, reversed and zero intervals, sigma8 integration, transfer-table
weighting/interpolation, 64-bit seed parsing, file path resolution, and unsupported
physics rejection. MPI evolution tests must additionally establish exact global
mass/particle count, uniform zero force, linear growth, and agreement across 1–4
ranks. No tolerance should be inferred for nonlinear evolution from those checks.
