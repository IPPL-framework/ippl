# Mathematical model and discretization {#fel_mathematics}

This page connects the equations in **MITHRA 2.0** to the operations performed by
the current FEL mini-app. Start with the [student guide](StudentGuide.md) for
configuration, ownership and execution. Here the formula for an update is derived
from the implementation; the reference to MITHRA explains its origin. Differences
in time centering and source deposition are stated explicitly.

The primary reference is Arya Fallahi, *MITHRA 2.0: A Full-Wave Simulation Tool
for Free Electron Lasers*, 2020, [arXiv:2009.13645](https://arxiv.org/abs/2009.13645).
Chapter 3 is available as [searchable HTML](https://arxiv.org/html/2009.13645).
The equation numbers below refer to that manual, not to this page.

Read in order: [continuum equations](#fel_math_continuum),
[grid and notation](#fel_math_grid), [nonstandard stencil](#fel_math_stencil),
[dispersion](#fel_math_dispersion), [particle sources](#fel_math_sources),
[field reconstruction](#fel_math_fields), [particle push](#fel_math_push),
[boundaries](#fel_math_boundary), [implementation map](#fel_math_map).

## 1. Continuum equations and normalization {#fel_math_continuum}

In vacuum, introduce the scalar potential @f$\phi@f$ and vector potential
@f$\mathbf A@f$. With the **Lorenz gauge**, the SI equations are

@f[
\frac{1}{c^2}\frac{\partial^2\phi}{\partial t^2}-\nabla^2\phi
  =\frac{\rho}{\epsilon_0},\qquad
\frac{1}{c^2}\frac{\partial^2\mathbf A}{\partial t^2}-\nabla^2\mathbf A
  =\mu_0\mathbf J,
@f]
@f[
\nabla\!\cdot\!\mathbf A+\frac{1}{c^2}\frac{\partial\phi}{\partial t}=0,
\qquad
\mathbf E=-\nabla\phi-\frac{\partial\mathbf A}{\partial t},\qquad
\mathbf B=\nabla\!\times\!\mathbf A.
@f]

These are the potential formulation in MITHRA (3.6)–(3.9), with the wave operator
written as time derivative minus Laplacian. The source sign is therefore positive
on the right. Charge is **signed** throughout this page.

For the base scales @f$L_0,T_0,M_0,Q_0@f$ in units.h, the potential and field
scales are

@f[
\Phi_0=\frac{M_0L_0^2}{T_0^2Q_0},\qquad
A_0=\frac{\Phi_0T_0}{L_0},\qquad
E_0=\frac{\Phi_0}{L_0},\qquad
B_0=\frac{\Phi_0T_0}{L_0^2}.
@f]

Divide SI quantities by their respective scales to obtain code values. The
tabulated decimal constants reproduce the corresponding SI vacuum constants to
their stored precision; the equations are implemented with
@f$c=\epsilon_0=\mu_0=1@f$.
Coordinates and times below are in the moving simulation frame; primes are
suppressed. Each of the four collocated components obeys the same scalar equation:

@f[
\partial_t^2 U-\nabla^2U=S,\qquad
U=(\phi,A_x,A_y,A_z),\quad S=(\rho,J_x,J_y,J_z).
@f]

Thus `J[0]` is charge density, not a scalar potential, and the solver's potential
arrays store `U`. In internal units, a particle's normalized momentum and
continuum motion are

@f[
\mathbf u_p=\frac{\mathbf p_p}{m_pc}=\gamma_p\mathbf\beta_p,
\quad \gamma_p=\sqrt{1+|\mathbf u_p|^2}.
@f]
@f[
\frac{d\mathbf r_p}{dt}=\frac{\mathbf u_p}{\gamma_p},\quad
\frac{d\mathbf u_p}{dt}=\frac{q_p}{m_p}
 \left(\mathbf E_p+\frac{\mathbf u_p}{\gamma_p}\times\mathbf B_p\right).
@f]

The self-fields come from the solved potentials. The prescribed undulator fields
are added when evaluating the particle force; the undulator is not evolved as
part of the grid potentials. This is the coupling described by MITHRA's particle
equation (3.48), specialized to the code's units.

The potential wave equations are equivalent to vacuum Maxwell equations only
with compatible gauge, sources and initial/boundary data. In the continuum,
@f$G=\partial_t\phi+\nabla\!\cdot\!\mathbf A@f$ satisfies

@f[
(\partial_t^2-\nabla^2)G=\partial_t\rho+\nabla\!\cdot\!\mathbf J.
@f]

Source continuity and compatible initial gauge data are therefore essential.
There is no separate gauge projection or Gauss-law correction in this mini-app.
The discrete scheme below does not automatically inherit that continuum identity.
With `space-charge=false`, the code sets the scalar source to zero and the
initially zero scalar potential stays zero. Current and vector-potential forces
remain active; this option is not a proof that every longitudinal force vanishes.

## 2. Grid locations and time labels {#fel_math_grid}

Let @f$\mathbf o@f$ be the mesh origin, @f$h_x,h_y,h_z@f$ the spacings,
@f$V=h_xh_yh_z@f$ the cell volume and @f$\tau=\Delta t@f$ the field step.
Global cell indices @f$\mathbf i=(i,j,k)@f$ locate all potential, source and
field components at

@f[
\mathbf x_{ijk}=\mathbf o+
 \big((i+\frac12)h_x,(j+\frac12)h_y,(k+\frac12)h_z\big).
@f]

There is no spatial Yee staggering. At entry to iteration n, call the two
potential histories @f$U^n,U^{n-1}@f$ and the particle endpoints
@f$\mathbf r_p^n,\mathbf r_p^{n-1}@f$. These are **storage labels** for the
actual loop; they do not assert that every stored quantity forms a centred
leapfrog. Initialization sets both potential histories to zero and both particle
endpoints to the same initial position.

The centred second differences used below are

@f[
\delta_{xx}\psi_{ijk}=
 \frac{\psi_{i+1,j,k}-2\psi_{i,j,k}+\psi_{i-1,j,k}}{h_x^2},
\qquad
\delta_{tt}\psi^n=\frac{\psi^{n+1}-2\psi^n+\psi^{n-1}}{\tau^2},
@f]

with analogous y and z differences. For smooth fields these approximate the
corresponding derivatives to second order. That statement concerns the operators;
it does not establish second-order accuracy of the complete particle/field loop.

## 3. The active nonstandard FDTD stencil {#fel_math_stencil}

The ordinary centred wave update would use
@f$\delta_{tt}\psi=(\delta_{xx}+\delta_{yy}+\delta_{zz})\psi+S@f$.
FEL instead selects `NonStandardFDTDSolver`. Following the construction in
MITHRA (3.21)–(3.27), it averages along z before taking transverse differences:

@f[
\bar\psi_{ijk}^n=
 a\psi_{i,j,k-1}^n+(1-2a)\psi_{i,j,k}^n+a\psi_{i,j,k+1}^n,
@f]
@f[
r=\left(\frac{h_z}{h_x}\right)^2+\left(\frac{h_z}{h_y}\right)^2,
\qquad a=\frac14\left(1+\frac{0.02}{r}\right),\qquad
\tau=h_z,\quad 0<r<1.
@f]

The symbol a here is the code's `calA`, MITHRA's NSFD averaging coefficient;
it is unrelated to the beam–radiator gap used in ChDR. The compact implemented
interior equation, independently for all four components, is

@f[
\frac{U_{ijk}^{n+1}-2U_{ijk}^{n}+U_{ijk}^{n-1}}{\tau^2}
 =\delta_{xx}\bar U_{ijk}^{n}+\delta_{yy}\bar U_{ijk}^{n}
  +\delta_{zz}U_{ijk}^{n}+S_{ijk}^{n}.
@f]

Here @f$S^n@f$ means the source array supplied for this call; its charge and
current sampling times are specified in the next sections. To match the C++
coefficient names, set

@f[
\begin{array}{rl}
\lambda_d&=\tau^2/h_d^2,\quad d=x,y,z,\\[0.5ex]
a_1&=2\big[1-(1-2a)(\lambda_x+\lambda_y)-\lambda_z\big],\\[0.5ex]
a_2&=\lambda_x,\qquad a_4=\lambda_y,\\[0.5ex]
a_6&=\lambda_z-2a(\lambda_x+\lambda_y),\qquad a_8=\tau^2.
\end{array}
@f]

The expanded update is then

@f[
\begin{array}{rl}
U_{ijk}^{n+1}={}&-U_{ijk}^{n-1}+a_1U_{ijk}^{n}\\[0.5ex]
 &+a_2\displaystyle\sum_{\sigma=\pm1}
 \left[aU_{i+\sigma,j,k-1}^{n}+(1-2a)U_{i+\sigma,j,k}^{n}
       +aU_{i+\sigma,j,k+1}^{n}\right]\\[0.5ex]
 &+a_4\displaystyle\sum_{\sigma=\pm1}
 \left[aU_{i,j+\sigma,k-1}^{n}+(1-2a)U_{i,j+\sigma,k}^{n}
       +aU_{i,j+\sigma,k+1}^{n}\right]\\[0.5ex]
 &+a_6\left(U_{i,j,k-1}^{n}+U_{i,j,k+1}^{n}\right)+a_8 S_{ijk}^{n}.
\end{array}
@f]

This uses **15 spatial locations** at level n: the centre, two axial neighbours
and twelve neighbours in the x-z and y-z planes. No x-y diagonal is used. The
previous time level contributes only its centre. The positive source term and
the separate x/y contributions in @f$a_6@f$ above are taken from IPPL. The
manual's expanded expression contains apparent sign/notational and repeated-x
transcription inconsistencies; use its differential equation and averaging
construction to derive the stencil rather than copying that coefficient list.

For a source-free solution independent of x and y, the update reduces exactly to

@f[
U_k^{n+1}=U_{k-1}^{n}+U_{k+1}^{n}-U_k^{n-1}.
@f]

This simple limit explains the special role of propagation along the beam axis.

## 4. Numerical dispersion and the mesh condition {#fel_math_dispersion}

Insert a source-free plane wave
@f$U_{ijk}^{n}=U_0\exp[\mathrm{i}(k_xih_x+k_yjh_y+k_zkh_z-\omega n\tau)]@f$
into the stencil. Its numerical dispersion relation is

@f[
\begin{array}{rl}
\displaystyle\frac{\sin^2(\omega\tau/2)}{\tau^2}
 ={}&\big[1-4a\sin^2(k_zh_z/2)\big]\\[1ex]
 &\displaystyle\quad\cdot
  \left[\frac{\sin^2(k_xh_x/2)}{h_x^2}
       +\frac{\sin^2(k_yh_y/2)}{h_y^2}\right]\\[1ex]
 &\displaystyle+\frac{\sin^2(k_zh_z/2)}{h_z^2}.
\end{array}
@f]

This is MITHRA (3.24) with c=1. For @f$k_x=k_y=0@f$ and @f$\tau=h_z@f$,
the physical branch has @f$\omega=|k_z|@f$ in the resolved axial band. The
source-free interior stencil therefore preserves axial phase speed; **oblique
waves still have numerical dispersion**. This is not a statement about boundary
reflections, particle errors or the accuracy of the complete radiation solution.

One can check the actual coefficient choice algebraically. Define

@f[
\zeta=\sin^2(k_zh_z/2)\in[0,1],\qquad
\eta=h_z^2\left[\frac{\sin^2(k_xh_x/2)}{h_x^2}
                +\frac{\sin^2(k_yh_y/2)}{h_y^2}\right]\in[0,r].
@f]

Then the right side after multiplication by @f$\tau^2=h_z^2@f$ is

@f[
\sin^2(\omega\tau/2)
 =\zeta+\left[1-\left(1+\frac{0.02}{r}\right)\zeta\right]\eta.
@f]

This bilinear expression has corner values @f$0,r,1,0.98@f$ on the rectangle
@f$[0,1]\times[0,r]@f$. The enforced @f$r<1@f$ keeps it in [0,1], so the
interior Fourier frequencies are real. This is a dispersion check for the
homogeneous stencil, not a complete stability proof for the driven, bounded PIC
system; limiting frequencies can sit at the endpoints. The grid spacings in
this check are the **boosted-frame** spacings. `timestep-ratio` does not alter
the implemented @f$\tau=h_z@f$.

## 5. Charge, current and field interpolation {#fel_math_sources}

Define the cell-centred cloud-in-cell (CIC) shape

@f[
W_{\mathbf i}(\mathbf r)=
 \prod_{d=x,y,z}\max\left(0,1-\left|\frac{r_d-x_{\mathbf i,d}}{h_d}\right|\right).
@f]

Only the eight surrounding centres contribute. In code, each direction uses
@f$g_d=(r_d-o_d)/h_d-1/2@f$, @f$b_d=\lfloor g_d\rfloor@f$ and
@f$\xi_d=g_d-b_d@f$; the two weights are @f$1-\xi_d@f$ and @f$\xi_d@f$.
Field gather uses the same trilinear weights. With `space-charge=true`,

@f[
\rho_{\mathbf i}^{n}=\frac1V\sum_pq_pW_{\mathbf i}(\mathbf r_p^n),
\qquad F_p(\mathbf r)=\sum_{\mathbf i}W_{\mathbf i}(\mathbf r)F_{\mathbf i}
@f]

for each E/B component F. With the flag false, the scalar source is zero.

The current deposition splits the **straight chord** from
@f$\mathbf r_p^{n-1}@f$ to @f$\mathbf r_p^n@f$ into segments with endpoints
@f$\mathbf a_{p\ell},\mathbf b_{p\ell}@f$. Using their midpoints
@f$\mathbf m_{p\ell}=(\mathbf a_{p\ell}+\mathbf b_{p\ell})/2@f$, the exact
implemented current formula is

@f[
\overline{\mathbf J}_{\mathbf i}^{\,n-1/2}
 =\frac{1}{V\tau}\sum_{p,\ell}q_p
  (\mathbf b_{p\ell}-\mathbf a_{p\ell})W_{\mathbf i}(\mathbf m_{p\ell}),
\qquad S_{\mathbf i}^{n}=
 (\rho_{\mathbf i}^{n},\overline{\mathbf J}_{\mathbf i}^{\,n-1/2}).
@f]

The segmenter cuts at mesh faces and supports at most one crossing per coordinate
per full step. With the active mesh condition and speeds below c=1, a particle's
full-step displacement is less than a cell width in each direction. The segmenter
is nevertheless not a general traversal for arbitrarily long trajectories.
The current formula uses midpoint quadrature; it does not integrate the CIC
shape exactly or retain the three Boris substep paths.

After deposition, halo accumulation sums shared contributions onto their owners.
With valid interpolation support, summing the CIC weights gives the following
checks for charge deposition **when enabled**, and for the current of the
particles included in the deposit:

@f[
V\sum_{\mathbf i}\rho_{\mathbf i}^{n}=\sum_pq_p,\qquad
V\sum_{\mathbf i}\overline{\mathbf J}_{\mathbf i}^{\,n-1/2}
 =\sum_pq_p\frac{\mathbf r_p^n-\mathbf r_p^{n-1}}{\tau}.
@f]

These global identities do not establish the local continuity equation
@f$(\rho^n-\rho^{n-1})/\tau+\nabla_h\!\cdot\!\overline{\mathbf J}^{n-1/2}=0@f$.
MITHRA's section 3.2.3 describes a zigzag deposition; IPPL's implementation here
must be assessed on its own operators. Escaped particles are deleted before the
next deposit, with no additional escaping-current source. Test continuity and
boundary losses explicitly before making conservation claims.

## 6. Reconstructing E and B: the actual time levels {#fel_math_fields}

The solver performs `step()`, `timeShift()`, then `evaluate_EB()`. After shifting,
`A_n` contains @f$U^{n+1}@f$ and `A_nm1` contains @f$U^n@f$ in our loop labels.
Define the centred first difference

@f[
D_x^0 f_{ijk}=\frac{f_{i+1,j,k}-f_{i-1,j,k}}{2h_x},
\qquad \nabla_h^0=(D_x^0,D_y^0,D_z^0).
@f]

The fields handed to the pusher are exactly

@f[
\widetilde{\mathbf E}=
 -\frac{\mathbf A^{n+1}-\mathbf A^n}{\tau}
 -\nabla_h^0\phi^{n+1},\qquad
\widetilde{\mathbf B}=\nabla_h^0\times\mathbf A^{n+1}.
@f]

For example,
@f$\widetilde B_x=D_y^0 A_z^{n+1}-D_z^0 A_y^{n+1}@f$.
The vector-potential difference spans two times, while the spatial derivatives
use the newest potential. The tildes deliberately avoid claiming a common
centred physical time for all terms.

MITHRA (3.50)–(3.55) uses a time average of the two vector-potential levels for
its magnetic reconstruction and discusses a staggered scalar-potential time.
IPPL currently uses the formulas above. Consequently, the manual's field
centering and accuracy arguments cannot be transferred unchanged to this code.
Likewise, centred reconstruction derivatives do not automatically reproduce the
nonstandard Laplacian used in the potential update.

## 7. Three relativistic Boris substeps {#fel_math_push}

Set @f$\delta t=\tau/3@f$ and start with
@f$(\mathbf r_{p,0},\mathbf u_{p,0})=(\mathbf r_p^n,\mathbf u_p^n)@f$.
For substep s=0,1,2, the code samples

@f[
\mathbf E_{p,s}=\sum_{\mathbf i}W_{\mathbf i}(\mathbf r_{p,s})
 \widetilde{\mathbf E}_{\mathbf i}
 +\mathbf E_{\rm ext}(\mathbf r_{p,s},t_n+s\delta t),
@f]

and the analogous B expression. The grid fields are frozen in time but regathered
at each new position. The external-field closure transforms the current position
to laboratory coordinates, evaluates Undulator, and transforms E/B back to the
simulation frame. LorentzTransform.h and Undulator.h document those formulas.

One substep is the following electric kick, magnetic rotation, electric kick and
position drift (MITHRA (3.49) gives the related Boris construction):

@f[
\begin{array}{rl}
\mathbf u^-&=\mathbf u_{p,s}+\displaystyle\frac{q_p\delta t}{2m_p}\mathbf E_{p,s},\\[1ex]
\gamma^-&=\sqrt{1+|\mathbf u^-|^2},\qquad
\mathbf t_B=\displaystyle\frac{q_p\delta t}{2m_p\gamma^-}\mathbf B_{p,s},\\[1ex]
\mathbf s_B&=\displaystyle\frac{2\mathbf t_B}{1+|\mathbf t_B|^2},\\[1ex]
\mathbf u^\star&=\mathbf u^-+\mathbf u^-\times\mathbf t_B,\\[1ex]
\mathbf u^+&=\mathbf u^-+\mathbf u^\star\times\mathbf s_B,\\[1ex]
\mathbf u_{p,s+1}&=\mathbf u^++\displaystyle\frac{q_p\delta t}{2m_p}\mathbf E_{p,s},\\[1ex]
\mathbf r_{p,s+1}&=\mathbf r_{p,s}
  +\delta t\displaystyle\frac{\mathbf u_{p,s+1}}{\sqrt{1+|\mathbf u_{p,s+1}|^2}}.
\end{array}
@f]

Here `t1`, `t2`, `t3` in the C++ correspond to
@f$\mathbf u^-,\mathbf u^\star,\mathbf u^+@f$. The drift uses the **new**
momentum. No separate half-step momentum initialization or midpoint external
sampling is performed. After s=2, the result is stored as the next particle
state; lost particles are removed and survivors migrate between MPI ranks.

The full algorithm is therefore: deposit the previous displacement, advance
potentials, reconstruct fields, push particles, increment the clock. It should
not be described as a conventional fully time-centred PIC leapfrog solely because
its rotation has the Boris form. Initially @f$\mathbf r^{-1}=\mathbf r^0@f$,
so the first current is zero; charge may still be deposited when enabled.

## 8. Initial data and Mur absorbing boundaries {#fel_math_boundary}

All three potential histories start at zero. There is no initial equilibrium
field solve. The spatial boundary values are advanced by the second-order Mur
implementation in `src/MaxwellSolvers/AbsorbingBC.h`, component by component.
MITHRA section 3.1.5 provides the outgoing-wave approximation motivating it.

For a face, let b be its boundary cell, i the adjacent **inward** cell, h the
normal spacing and @f$L_\parallel@f$ the sum of centred second differences in
the two tangential directions. For any potential component @f$\psi@f$, the
actual code coefficients give

@f[
\begin{array}{rl}
\psi_b^{n+1}={}&-\psi_i^{n-1}
 +R\big(\psi_b^{n-1}+\psi_i^{n+1}\big)
 +\displaystyle\frac{2h}{\tau+h}\big(\psi_b^n+\psi_i^n\big)\\[1ex]
 &+\displaystyle\frac{\tau^2h}{2(\tau+h)}
 L_\parallel\big(\psi_b^n+\psi_i^n\big),
 \qquad R=\displaystyle\frac{\tau-h}{\tau+h}.
\end{array}
@f]

R is a coefficient of this update, not a measured reflection coefficient.
The tangential term is what distinguishes the second-order face approximation
from a simple one-dimensional outgoing-wave condition. Edges and corners have
their own updates; the code does not average independently computed face values.

For completeness, all three implemented boundary cases can be expressed by one
compact relation. Let m=1,2,3 denote a face, edge or corner, respectively. Index
the @f$2^m@f$ points of its inward block by
@f$\mathbf s\in\{0,1\}^m@f$, with @f$\mathbf s=0@f$ the boundary point to
be updated and @f$h_d@f$ the m inward-normal spacings. Define

@f[
w_{\mathbf s}=\sum_{d=1}^{m}\frac{2s_d-1}{h_d},\qquad
D_t^0\psi^n=\frac{\psi^{n+1}-\psi^{n-1}}{2\tau}.
@f]

With @f$L_\parallel@f$ now covering the remaining 3-m tangential directions
(zero for a corner), the code's coefficient tables are equivalent to

@f[
\sum_{\mathbf s\in\{0,1\}^m}
 \left[4w_{\mathbf s}D_t^0\psi_{\mathbf s}^n
       -(m+1)\delta_{tt}\psi_{\mathbf s}^n
       +L_\parallel\psi_{\mathbf s}^n\right]=0.
@f]

Solve this equation for @f$\psi_{\mathbf0}^{n+1}@f$, using already updated
interior/face/edge neighbours. This compact form is an algebraic regrouping of
IPPL's boundary coefficients; it does not add a new boundary method.

The kernel sequence is interior update, potential halo fill, faces, edges,
corners, history copies and field reconstruction. Each boundary pass fences.
There is no additional potential halo fill after these physical-boundary writes
before reconstruction. Boundary/partition intersections therefore deserve a
dedicated MPI check. Mur conditions approximate an open boundary; they do not
guarantee zero reflection for arbitrary incidence or evanescent fields.

## 9. Equation-to-code and literature map {#fel_math_map}

| Mathematical operation | Implementation | MITHRA background |
|---|---|---|
| Potential wave update, averaging and coefficients | `src/MaxwellSolvers/NonStandardFDTDSolver.hpp`: `step()` | (3.6)–(3.7), (3.21)–(3.27) |
| Timestep and zero potential histories | Same file: `initialize()` | Section 3.1.3 |
| History shift and the actual E/B reconstruction | `src/MaxwellSolvers/FDTDSolverBase.hpp`: `solve()`, `evaluate_EB()` | Compare with (3.50)–(3.55); centering differs |
| Collocated storage and active solver choice | datatypes.h | Spatial grid convention belongs to this implementation |
| CIC charge and particle push | FreeElectronLaserManager::depositChargeDensity(), FreeElectronLaserManager::push() | (3.48)–(3.49), (3.56)–(3.57) |
| Segmented midpoint current | `src/Interpolation/CurrentDeposition.hpp`, `src/FEM/GridPathSegmenter.hpp` | Compare with section 3.2.3; deposition algorithm differs |
| Face, edge and corner Mur updates | `src/MaxwellSolvers/AbsorbingBC.h` | Section 3.1.5, especially (3.36)–(3.47) |
| Units and external field transformation | units.h, LorentzTransform.h, Undulator.h | Sections 3.3.1–3.3.2 |

The MITHRA manual and its [author-maintained source repository](https://github.com/aryafallahi/mithra)
provide the derivation and implementation history. The formulas on this page
describe the inspected IPPL code. Neither the source-free dispersion calculation
nor agreement of algebraic coefficient forms establishes full PIC convergence,
discrete gauge conservation or quantitative radiation accuracy.
