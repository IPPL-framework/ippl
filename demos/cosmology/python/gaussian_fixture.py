#!/usr/bin/env python3
## @file gaussian_fixture.py
# @brief Common-particle, band-limited Gaussian 1LPT fixtures for local PM studies.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
r"""Common-particle, band-limited Gaussian 1LPT fixtures for local PM studies.

This module generates initial data only: it does not run a simulation. The
same seed specifies the same continuous Fourier field, independently of
particle sampling, PM mesh, starting redshift, or the other modes retained.
It is deliberately separate from the production IC RNG. This is a controlled
BBKS/flat-LCDM numerical comparison, NOT cosmological-statistics qualification.

The Fourier convention is delta(q,z=0) = sum_m delta_m exp(+i k_m.q), with
E[|delta_m|^2] = P(k_m)/L^3 and delta_-m = conjugate(delta_m). The prescribed
sigma8 normalizes the continuous spectrum (with the validated k<=10 h/Mpc
normalization integral); neither a finite-band expectation nor a particular
realization is forced to that variance. Pure BBKS uses q=k/(Omega_m*h) and
does not model baryonic transfer features even though Omega_bar is recorded.

Example from another Python script::

    frame, metadata = make_gaussian_fixture(32, 49, seed=20261003)
    # The caller writes/hashes the shared CSV and passes exactly that file
    # to both unchanged evolution adapters. No data are written here.

Positions are in Mpc/h, k in h/Mpc, P in (Mpc/h)^3. Canonical momentum
p=a^2 dx/d(H0*t) is in Mpc/h. Cell-centred lattice IDs are x-fast. Momentum
is rounded once to float32, then returned as exact float64 values for CSV.
Dependencies are NumPy and pandas, already used by the validation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

import validate_linear
from validate_linear import bbks_spectrum, growth_reference


## @var DefaultCosmology
# @brief Named DefaultCosmology protocol/schema value; the source initializer records its exact contents.
DefaultCosmology = MappingProxyType({
    "Omega_m": .31, "Omega_bar": .0487, "hubble": .675,
    "n_s": .965, "Sigma_8": .82, "box_size": 168.75,
})
## @var FixtureColumns
# @brief Named FixtureColumns protocol/schema value; the source initializer records its exact contents.
FixtureColumns = ("id", "x", "y", "z", "px", "py", "pz", "mass")
## @var RngDomain
# @brief Named RngDomain protocol/schema value; the source initializer records its exact contents.
RngDomain = b"IPPL-band-limited-Gaussian-v1\0"
## @var MaximumCutoff
# @brief Named MaximumCutoff protocol/schema value; the source initializer records its exact contents.
MaximumCutoff = 12


## @brief Validate an integer without accepting booleans or values outside the caller's range.
# @see cosmology_tools
#
# @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
# @param name Stable artifact/run/check identifier as defined by the caller.
# @param minimum Predeclared lower bound on the scalar or integer accepted by this check.
# @param maximum Predeclared upper range bound, not a fitted diagnostic normalization.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return value


## @brief Validate the supported Gaussian fixture cosmology and reject unknown parameter keys.
# @see cosmology_tools
#
# @param cosmology Optional supported Gaussian cosmology mapping; unspecified keys use explicit defaults.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def _parameters(cosmology: Mapping | None) -> dict:
    parameters = dict(DefaultCosmology)
    if cosmology is not None:
        unknown = set(cosmology) - set(parameters)
        if unknown:
            raise ValueError(f"Unsupported cosmology parameters: {sorted(unknown)}")
        parameters.update(cosmology)
    parameters = {key: float(value) for key, value in parameters.items()}
    if not all(math.isfinite(value) for value in parameters.values()):
        raise ValueError("Cosmology parameters must be finite")
    if not (0 < parameters["Omega_m"] <= 1
            and 0 <= parameters["Omega_bar"] <= parameters["Omega_m"]
            and parameters["hubble"] > 0 and parameters["Sigma_8"] > 0
            and parameters["box_size"] > 0 and -3 < parameters["n_s"] < 4):
        raise ValueError("Invalid flat, radiation-free LCDM/BBKS parameters")
    return parameters


## @brief Unit-variance complex Gaussian keyed by a physical integer mode.
# @see cosmology_tools
#
# @param seed Unsigned 64-bit realization seed; the RNG domain/key contract is module-specific.
# @param mode One signed integer Fourier mode; conjugation uses its canonical member.
# @return Unit-variance complex Gaussian; real/imaginary variances are each one half.
def mode_gaussian(seed: int, mode) -> complex:
    """Unit-variance complex Gaussian keyed by a physical integer mode.

    The canonical member has its first nonzero Cartesian component positive.
    SHA256(domain || little-endian uint64 seed || three int32 components)
    provides two disjoint 52-bit open-interval uniform variates, followed by
    Box--Muller. The conjugate member is returned without another RNG draw.
    This is a deterministic hash-counter RNG contract, not a cryptographic
    or formal statistical certification. Re/Im variances are each 1/2.
    """
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    try:
        mode = tuple(mode)
    except TypeError as error:
        raise ValueError("mode must contain three integer components") from error
    if len(mode) != 3:
        raise ValueError("mode must contain three integer components")
    mode = tuple(_integer(value, "mode component", -(2**31)+1, 2**31-1)
                 for value in mode)
    first = next((value for value in mode if value), 0)
    if first == 0:
        raise ValueError("The DC mode has no Gaussian draw")
    canonical = mode if first > 0 else tuple(-value for value in mode)
    digest = hashlib.sha256(RngDomain + struct.pack("<Qiii", seed, *canonical)).digest()
    uniforms = [((int.from_bytes(digest[start:start+8], "little") >> 12) + .5)
                / 2**52 for start in (0, 8)]
    radius = math.sqrt(-math.log(uniforms[0]))
    angle = 2 * math.pi * uniforms[1]
    value = radius * complex(math.cos(angle), math.sin(angle))
    return value if first > 0 else value.conjugate()


## @brief Spherical top-hat Fourier window, including its exact x=0 limit.
# @see cosmology_tools
#
# @param x Dimensionless top-hat argument for window routines; otherwise the coordinate/data defined by the caller.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def top_hat_window(x) -> np.ndarray:
    """Spherical top-hat Fourier window, including its exact x=0 limit."""
    values = np.asarray(x, dtype=np.float64)
    small = np.abs(values) < .01
    result = np.empty_like(values)
    result[small] = (1 - values[small]**2/10 + values[small]**4/280
                     - values[small]**6/15120)
    result[~small] = (3 * (np.sin(values[~small])
                          - values[~small]*np.cos(values[~small])) / values[~small]**3)
    return result


## @brief Continuous z=0 field. Arrays are read-only, ordered lexicographically xyz.
# @see cosmology_tools
# @note A frozen dataclass prevents field rebinding; make_realization also makes NumPy arrays read-only. A direct constructor must honor that array-immutability contract itself.
@dataclass(frozen=True)
class GaussianRealization:
    """Continuous z=0 field. Arrays are read-only, ordered lexicographically xyz."""

    ## @var seed
    # @brief Unsigned 64-bit realization seed; the RNG domain/key contract is module-specific.
    seed: int
    ## @var cutoff
    # @brief Positive spherical integer-mode ceiling, strictly below the sampling Nyquist where required.
    cutoff: int
    ## @var parameters
    # @brief Named protocol or cosmology parameters; unsupported keys are rejected by the calling validator.
    parameters: Mapping
    ## @var modes
    # @brief Signed integer Fourier-mode array with three Cartesian components per mode.
    modes: np.ndarray
    ## @var coefficients
    # @brief Complex dimensionless density Fourier coefficients with the module's declared normalization.
    coefficients: np.ndarray
    ## @var power
    # @brief Density power array under the explicit Fourier/volume normalization.
    power: np.ndarray
    ## @var coefficient_sha256
    # @brief Named coefficient sha256 protocol/schema value; the source initializer records its exact contents.
    coefficient_sha256: str

    ## @brief Evaluate the metadata helper in the documented module workflow.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def metadata(self) -> dict:
        boxSize = self.parameters["box_size"]
        waveNumbers = 2 * math.pi / boxSize * np.linalg.norm(self.modes, axis=1)
        windowSquared = top_hat_window(8 * waveNumbers)**2
        expectedPower = self.power / boxSize**3
        realizedPower = np.abs(self.coefficients)**2
        return {
            "schema": "ippl-common-gaussian-realization-v1",
            "seed": self.seed, "cosmology": dict(self.parameters),
            "coefficient_sha256": self.coefficient_sha256,
            "coefficient_hash_encoding": "domain, xyz int32 little-endian modes, complex128 little-endian delta_z0",
            "rng": "SHA256 keyed physical integer mode + two 52-bit uniforms + Box-Muller; conjugate pairs",
            "fourier_convention": "delta(q,z0)=sum_m delta_m exp(+i k_m.q); E|delta_m|^2=P(k_m)/L^3",
            "coefficient_epoch": "z=0; D(1)=1",
            "cutoff_fundamental": self.cutoff,
            "mode_band": "spherical 0<|m|<=cutoff; DC absent; sampling Nyquist absent",
            "mode_count_including_conjugates": len(self.modes),
            "independent_complex_modes": len(self.modes)//2,
            "k_min_h_per_mpc": float(waveNumbers.min()),
            "k_max_h_per_mpc": float(waveNumbers.max()),
            "spectrum": "pure BBKS T(q), q=k/(Omega_m*h); no baryonic transfer features",
            "omega_bar_role": "recorded cosmology only; pure BBKS shape does not depend on Omega_bar",
            "sigma8_convention": "prescribed continuous untruncated BBKS spectrum normalization, not finite box/band or measured realization",
            "spectrum_normalization_integral_k_h_per_mpc": [1e-8, 10.],
            "finite_band_density_variance_expected_z0": float(expectedPower.sum()),
            "finite_band_density_variance_realized_z0": float(realizedPower.sum()),
            "finite_band_sigma8_squared_expected_z0": float(expectedPower @ windowSquared),
            "finite_band_sigma8_squared_realized_z0": float(realizedPower @ windowSquared),
            "realized_variance_fitted": False,
            "background": "flat LCDM; Omega_Lambda=1-Omega_m; no radiation/neutrinos; w=-1",
            "scope": "band-limited Gaussian 1LPT local PM comparison; not full cosmological-statistics qualification",
            "units": {"position": "Mpc/h", "momentum": "Mpc/h; p=a^2 dx/d(H0*t)",
                      "k": "h/Mpc", "power": "(Mpc/h)^3"},
        }


## @brief Construct fixed continuous modes, independent of NP, NM, and redshift.
# @see cosmology_tools
#
# @param seed Unsigned 64-bit realization seed; the RNG domain/key contract is module-specific.
# @param cutoff Positive spherical integer-mode ceiling, strictly below the sampling Nyquist where required.
# @param cosmology Optional supported Gaussian cosmology mapping; unspecified keys use explicit defaults.
# @return Immutable common-phase GaussianRealization with a coefficient hash.
def make_realization(seed: int, *, cutoff: int = 12,
                     cosmology: Mapping | None = None) -> GaussianRealization:
    """Construct fixed continuous modes, independent of NP, NM, and redshift."""
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    cutoff = _integer(cutoff, "cutoff", 1, MaximumCutoff)
    parameters = _parameters(cosmology)
    modes = np.asarray([mode for mode in itertools.product(range(-cutoff, cutoff+1), repeat=3)
                        if 0 < sum(component**2 for component in mode) <= cutoff**2],
                       dtype=np.int32)
    waveNumbers = 2 * math.pi / parameters["box_size"] * np.linalg.norm(modes, axis=1)
    power = bbks_spectrum(waveNumbers, parameters)
    coefficients = (np.asarray([mode_gaussian(seed, mode) for mode in modes])
                    * np.sqrt(power / parameters["box_size"]**3))
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(power)):
        raise ValueError("Spectrum/realization overflow")
    digest = hashlib.sha256(RngDomain + modes.astype("<i4").tobytes()
                            + coefficients.astype("<c16").tobytes()).hexdigest()
    for array in (modes, coefficients, power):
        array.setflags(write=False)
    return GaussianRealization(seed, cutoff, MappingProxyType(parameters),
                               modes, coefficients, power, digest)


## @brief Require an even particle lattice and a Fourier cutoff strictly below its Nyquist.
# @see cosmology_tools
#
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @param cutoff Positive spherical integer-mode ceiling, strictly below the sampling Nyquist where required.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def _sampling_grid(particle_grid: int, cutoff: int) -> int:
    particleGrid = _integer(particle_grid, "particle_grid", 4, 1024)
    if particleGrid % 2 or 2*cutoff >= particleGrid:
        raise ValueError("Use an even particle_grid with cutoff strictly below its Nyquist")
    return particleGrid


## @brief Return the cell-centred x-fast lattice, independent of the PM mesh.
# @see cosmology_tools
#
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @param box_size Positive periodic comoving box side in Mpc/h.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def lattice_positions(particle_grid: int, box_size: float) -> np.ndarray:
    """Return the cell-centred x-fast lattice, independent of the PM mesh."""
    n = _integer(particle_grid, "particle_grid", 1, 1024)
    if not math.isfinite(box_size) or box_size <= 0:
        raise ValueError("box_size must be finite and positive")
    ids = np.arange(n**3, dtype=np.int64)
    return (np.column_stack((ids % n, ids//n % n, ids//n**2)) + .5) * (box_size/n)


## @brief Evaluate psi0 at cell centres with psi0_hat=+i k delta_hat/k^2.
# @see cosmology_tools
#
# @param realization Common-phase GaussianRealization with fixed coefficients and cosmology.
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @return Float64 (NP^3,3) z=0 displacement in Mpc/h, ordered by x-fast ID.
def sample_displacement(realization: GaussianRealization, particle_grid: int) -> np.ndarray:
    r"""Evaluate psi0 at cell centres with psi0_hat=+i k delta_hat/k^2.

    Storage is z,y,x, so flattening yields x-fast particle IDs. The half-cell
    phase belongs only to evaluation, never to the continuous coefficients.
    NumPy norm='forward' makes the inverse transform the unnormalized sum.
    """
    n = _sampling_grid(particle_grid, realization.cutoff)
    waves = 2 * math.pi / realization.parameters["box_size"] * realization.modes
    waveSquared = np.einsum("ij,ij->i", waves, waves)
    phase = np.exp(1j * math.pi * realization.modes.sum(axis=1) / n)
    indices = tuple((realization.modes[:, component] % n) for component in (2, 1, 0))
    displacement = np.empty((n**3, 3), dtype=np.float64)
    for component in range(3):
        mesh = np.zeros((n, n, n), dtype=np.complex128)
        mesh[indices] = 1j * waves[:, component]/waveSquared * realization.coefficients * phase
        values = np.fft.ifftn(mesh, norm="forward")
        scale = float(np.max(np.abs(values.real)))
        if float(np.max(np.abs(values.imag))) > 64*np.finfo(float).eps*scale:
            raise ValueError("Non-real inverse transform: broken Hermitian/axis convention")
        displacement[:, component] = values.real.ravel()
    return displacement


## @brief Sample the symmetric psi0 derivative -k_i k_j delta_hat/k^2.
# @see cosmology_tools
#
# @param realization Common-phase GaussianRealization with fixed coefficients and cosmology.
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @return Float64 (NP^3,3,3) displacement derivative tensor, dimensionless.
def sample_deformation(realization: GaussianRealization, particle_grid: int) -> np.ndarray:
    r"""Sample the symmetric psi0 derivative -k_i k_j delta_hat/k^2.

    This is an initial-map diagnostic only, not an evolution or force kernel.
    The full 3x3 tensor costs 18 MiB for NP=64; no particle-mode dense matrix
    is formed. Its trace is minus the prescribed linear delta field.
    """
    n = _sampling_grid(particle_grid, realization.cutoff)
    waves = 2 * math.pi / realization.parameters["box_size"] * realization.modes
    waveSquared = np.einsum("ij,ij->i", waves, waves)
    phase = np.exp(1j * math.pi * realization.modes.sum(axis=1) / n)
    indices = tuple((realization.modes[:, component] % n) for component in (2, 1, 0))
    derivative = np.empty((n**3, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(row, 3):
            mesh = np.zeros((n, n, n), dtype=np.complex128)
            mesh[indices] = (-waves[:, row]*waves[:, column]/waveSquared
                             * realization.coefficients * phase)
            values = np.fft.ifftn(mesh, norm="forward")
            scale = float(np.max(np.abs(values.real)))
            if float(np.max(np.abs(values.imag))) > 64*np.finfo(float).eps*scale:
                raise ValueError("Non-real deformation inverse transform")
            derivative[:, row, column] = values.real.ravel()
            derivative[:, column, row] = values.real.ravel()
    return derivative


## @brief Hash numeric data, not a CSV serialization; caller must hash CSV bytes.
# @see cosmology_tools
#
# @param frame Particle DataFrame with the exact columns/ID/finite-state contract checked by this routine.
# @return Hexadecimal canonical numerical digest; not the CSV-byte hash.
def phase_space_sha256(frame: pd.DataFrame) -> str:
    """Hash numeric data, not a CSV serialization; caller must hash CSV bytes."""
    if tuple(frame.columns) != FixtureColumns:
        raise ValueError("Unexpected phase-space columns")
    digest = hashlib.sha256(b"IPPL-common-Gaussian-phase-space-v1\0")
    digest.update(frame["id"].to_numpy(dtype="<u8").tobytes())
    digest.update(frame.loc[:, FixtureColumns[1:]].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


## @brief Return exactly shared unit-mass CSV columns and a JSON-safe audit record.
# @see cosmology_tools
#
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @param redshift Initialization redshift; a=1/(1+redshift).
# @param seed Unsigned 64-bit realization seed; the RNG domain/key contract is module-specific.
# @param cutoff Positive spherical integer-mode ceiling, strictly below the sampling Nyquist where required.
# @param cosmology Optional supported Gaussian cosmology mapping; unspecified keys use explicit defaults.
# @return DataFrame with id,x,y,z,px,py,pz,mass and a JSON-safe audit record.
def make_gaussian_fixture(particle_grid: int, redshift: float, seed: int, *,
                          cutoff: int = 12, cosmology: Mapping | None = None
                          ) -> tuple[pd.DataFrame, dict]:
    """Return exactly shared unit-mass CSV columns and a JSON-safe audit record."""
    realization = make_realization(seed, cutoff=cutoff, cosmology=cosmology)
    n = _sampling_grid(particle_grid, realization.cutoff)
    redshift = float(redshift)
    if not math.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and nonnegative")
    a = 1 / (1 + redshift)
    omegaMatter, boxSize = (realization.parameters[key] for key in ("Omega_m", "box_size"))
    growth, rate = growth_reference(a, omegaMatter)
    expansion = math.sqrt(omegaMatter/a**3 + 1-omegaMatter)
    displacement = sample_displacement(realization, n)
    initialEigenvalues = 1 + growth*np.linalg.eigvalsh(sample_deformation(realization, n))
    minimumEigenvalue = float(initialEigenvalues.min())
    minimumJacobian = float(np.prod(initialEigenvalues, axis=1).min())
    if minimumEigenvalue <= 0:
        raise ValueError("Initial 1LPT map has a nonpositive sampled eigenvalue; no resampling/rescaling is permitted")
    positions = np.remainder(lattice_positions(n, boxSize) + growth*displacement, boxSize)
    rawMomentum = (a*a*expansion*rate*growth) * displacement
    momentum = rawMomentum.astype(np.float32).astype(np.float64)
    if not np.all(np.isfinite(momentum)):
        raise ValueError("Momentum exceeds the common float32 storage range")
    frame = pd.DataFrame({"id": np.arange(n**3, dtype=np.int64)})
    frame.loc[:, ["x", "y", "z"]] = positions
    frame.loc[:, ["px", "py", "pz"]] = momentum
    frame["mass"] = 1.
    rawNorm = float(np.linalg.norm(rawMomentum))
    metadata = realization.metadata()
    metadata.update({
        "fixture": "band_limited_gaussian_lcdm", "particle_grid": n,
        "particle_count": n**3, "particle_mass": 1., "redshift_initial": redshift,
        "a_initial": a, "D_initial": growth, "f_initial": rate, "E_initial": expansion,
        "sampling": "cell-centred lattice q=(j+1/2)*L/NP; x-fast IDs; independent of PM mesh",
        "initialization": "1LPT: x=wrap(q+D*psi0); p=a^2*E*f*D*psi0",
        "minimum_sampled_initial_map_eigenvalue": minimumEigenvalue,
        "minimum_sampled_initial_map_jacobian": minimumJacobian,
        "initial_map_diagnostic": "Fourier derivative at particle lattice sites; positive samples do not prove global continuous injectivity",
        "momentum_contract": "rounded once to float32 then represented as exact float64, shared by both codes",
        "momentum_rounding_relative_l2": float(np.linalg.norm(momentum-rawMomentum)/rawNorm),
        "initial_displacement_per_component_rms_mpc_over_h": float(growth*np.sqrt(np.mean(displacement**2))),
        "initial_displacement_vector_rms_mpc_over_h": float(growth*np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
        "initial_max_displacement_over_particle_spacing": float(growth*np.max(np.linalg.norm(displacement, axis=1))/(boxSize/n)),
        "initial_linear_density_rms": float(growth*np.linalg.norm(realization.coefficients)),
        "mean_displacement_z0_mpc_over_h": displacement.mean(axis=0).tolist(),
        "mean_quantized_momentum_mpc_over_h": momentum.mean(axis=0).tolist(),
        "phase_space_sha256": phase_space_sha256(frame),
        "phase_space_hash_encoding": "domain, uint64 little-endian IDs, then row-major float64 little-endian x,y,z,px,py,pz,mass; not CSV bytes",
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(validate_linear.__file__))},
    })
    return frame, metadata
