#!/usr/bin/env python3
"""Independent equal-mass particle-density measurements for local PM studies.

The simulation mesh is not used. Density coefficients use the convention
delta_m = mean(exp(-2 pi i m.x/L)) for nonzero integer m. Interlaced PCS (cubic
B-spline assignment) on a fixed analysis grid removes odd alias images;
dividing by its known sinc^4 window does not remove every alias. Direct low-band
coefficients remain the qualified measurement. Higher-band FFT coefficients
are explicitly characterization. Legacy CIC routines remain available to
inspect the failed initial analysis measurement; simulation CIC is unchanged.

This module never changes particle data or a simulation/acceptance tolerance.
"""
from __future__ import annotations

import itertools
import numpy as np


ShellEdges = (.5, 1.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5)
AnalysisGrid = 128
DirectRelativeLimit = 1e-3
DirectAbsoluteFloor = 1e-12
MassRelativeLimit = 1e-12


def _geometry(positions, box, grid=None):
    positions = np.asarray(positions, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
            or not np.isfinite(positions).all() or not np.isfinite(box) or box <= 0):
        raise ValueError("Require finite nonempty Nx3 positions and a positive finite box")
    if grid is not None and (isinstance(grid, (bool, np.bool_))
                            or not isinstance(grid, (int, np.integer))
                            or grid < 4 or grid % 2):
        raise ValueError("Analysis grid must be an even integer >= 4")
    return positions


def _modes(modes, grid=None):
    modes = np.asarray(modes)
    if (modes.ndim != 2 or modes.shape[1] != 3 or not len(modes)
            or not np.isfinite(modes).all() or not np.equal(modes, np.rint(modes)).all()
            or np.any(np.all(modes == 0, axis=1))):
        raise ValueError("Require finite nonempty Mx3 nonzero integer modes")
    if np.max(np.abs(modes)) > np.iinfo(np.int32).max:
        raise ValueError("Mode exceeds supported integer range")
    modes = modes.astype(np.int64)
    if grid is not None and np.any(2 * np.abs(modes) >= grid):
        raise ValueError("Modes must lie strictly inside each analysis Nyquist plane")
    return modes


def unique_modes(max_mode=12):
    """One representative of each +/- pair in the spherical integer band."""
    if (isinstance(max_mode, (bool, np.bool_))
            or not isinstance(max_mode, (int, np.integer)) or max_mode < 1):
        raise ValueError("Maximum spherical mode must be a positive integer")
    return np.asarray([
        mode for mode in itertools.product(range(-max_mode, max_mode + 1), repeat=3)
        if 0 < sum(v*v for v in mode) <= max_mode**2
        and next(v for v in mode if v) > 0
    ], dtype=np.int64)


def deposit_cic(positions, box, grid=AnalysisGrid, shift_cells=(0., 0., 0.)):
    """Return equal-particle counts on nodes (j + shift_cells)*L/grid.

    Array axes are physical (x,y,z), independent of particle ID order. Each
    particle contributes total mass one. Positions may wrap any box boundary.
    """
    positions = _geometry(positions, box, grid)
    shift = np.asarray(shift_cells, dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("Grid origin shift must be a finite three-vector")
    coordinate = (np.remainder(positions, box) * (grid / box) - shift) % grid
    left = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - left
    counts = np.zeros(grid**3, dtype=np.float64)
    for corner in itertools.product((0, 1), repeat=3):
        index = (left + corner) % grid
        weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
        flat = (index[:, 0] * grid + index[:, 1]) * grid + index[:, 2]
        counts += np.bincount(flat, weights=weight, minlength=grid**3)
    return counts.reshape(grid, grid, grid)


def cic_window(modes, grid=AnalysisGrid):
    """Fourier transform of the normalized separable linear-assignment kernel."""
    _geometry(np.zeros((1, 3)), 1., grid)
    modes = _modes(modes, grid)
    return np.prod(np.sinc(modes / grid)**2, axis=1)


def cic_alias_bound(modes, grid=AnalysisGrid, interlaced=True):
    """Conservative absolute error bound after CIC-window deconvolution.

    For normalized nonnegative particle masses, |delta_alias| <= 1. With
    t=m/grid, sum_l (t/(t+l))^2 = (pi*t/sin(pi*t))^2 = S, and the alternating
    sum is S*cos(pi*t). Averaging the unshifted and half-cell shifted grids
    retains even sum(l), so the sum of alias magnitudes is
    [product(S) + product(S*cos(pi*t))]/2 - 1. This is an absolute coefficient
    bound, NOT a relative error guarantee for a weak density mode.
    """
    _geometry(np.zeros((1, 3)), 1., grid)
    modes = _modes(modes, grid)
    t = modes / grid
    log_sum = -2 * np.log(np.sinc(t)).sum(axis=1)
    if not interlaced:
        return np.expm1(log_sum)
    log_alternating = log_sum + np.log(np.cos(np.pi * t)).sum(axis=1)
    return np.maximum(0., .5 * (np.expm1(log_sum) + np.expm1(log_alternating)))


def _cic_extract(positions, box, modes, grid, interlaced, deconvolve):
    positions = _geometry(positions, box, grid)
    modes = _modes(modes, grid)
    shifts = ((0., 0., 0.), (.5, .5, .5)) if interlaced else ((0., 0., 0.),)
    result = np.zeros(len(modes), dtype=np.complex128)
    mass_errors = []
    indices = tuple((modes % grid).T)
    for shift in shifts:
        counts = deposit_cic(positions, box, grid, shift)
        mass_errors.append(float(abs(counts.sum(dtype=np.float64) / len(positions) - 1)))
        # FFT convention is exp(-2*pi*i*m*j/grid); physical nodes include the
        # origin shift, hence the NEGATIVE phase below. Normalization is total
        # particle mass, not the number of analysis cells.
        transformed = np.fft.fftn(counts)
        phase = np.exp(-2j * np.pi * (modes @ np.asarray(shift)) / grid)
        result += transformed[indices] * phase / len(positions)
        del transformed, counts
    result /= len(shifts)
    if deconvolve:
        result /= cic_window(modes, grid)
    return result, mass_errors


def cic_coefficients(positions, box, modes, grid=AnalysisGrid,
                     interlaced=True, deconvolve=True):
    """Physical-origin non-DC Fourier coefficients; defaults use interlaced CIC."""
    return _cic_extract(positions, box, modes, grid, interlaced, deconvolve)[0]


def deposit_pcs(positions, box, grid=AnalysisGrid, shift_cells=(0., 0., 0.)):
    """Equal-mass cubic B-spline deposition, four nodes per Cartesian axis.

    For f in [0,1), the nodes floor(x/h)-1 through floor(x/h)+2 receive
    ((1-f)^3, 4-6f^2+3f^3, 1+3f+3f^2-3f^3, f^3)/6. Array axes and shifted
    physical node origins have exactly the same convention as deposit_cic.
    This is an analysis estimator only, never the simulation force assignment.
    """
    positions = _geometry(positions, box, grid)
    shift = np.asarray(shift_cells, dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("Grid origin shift must be a finite three-vector")
    coordinate = (np.remainder(positions, box) * (grid / box) - shift) % grid
    left = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - left
    indices, weights = [], []
    for axis in range(3):
        f = fraction[:, axis]
        indices.append([(left[:, axis] + offset) % grid for offset in (-1, 0, 1, 2)])
        weights.append(((1-f)**3 / 6, (4-6*f*f+3*f**3) / 6,
                        (1+3*f+3*f*f-3*f**3) / 6, f**3 / 6))
    counts = np.zeros(grid**3, dtype=np.float64)
    for i, j, k in itertools.product(range(4), repeat=3):
        flat = (indices[0][i] * grid + indices[1][j]) * grid + indices[2][k]
        weight = weights[0][i] * weights[1][j] * weights[2][k]
        counts += np.bincount(flat, weights=weight, minlength=grid**3)
    return counts.reshape(grid, grid, grid)


def pcs_window(modes, grid=AnalysisGrid):
    """Normalized cubic B-spline Fourier window, product sinc(m/grid)^4."""
    _geometry(np.zeros((1, 3)), 1., grid)
    modes = _modes(modes, grid)
    return np.prod(np.sinc(modes / grid)**4, axis=1)


def pcs_alias_bound(modes, grid=AnalysisGrid, interlaced=True):
    """Absolute alias bound for PCS-window-deconvolved particle coefficients.

    Let t=m/grid. Three derivatives of pi*cot(pi*t) and pi*csc(pi*t) give
    A=sum_l(t/(t+l))^4=(pi*t/sin(pi*t))^4*(2+cos(2*pi*t))/3 and
    B=sum_l(-1)^l*(t/(t+l))^4=(pi*t/sin(pi*t))^4*cos(pi*t)*(5+cos(pi*t)^2)/6.
    A=B=1 at t=0. With nonnegative masses normalized to one, every alias
    coefficient has magnitude <=1. The ordinary bound is product(A)-1;
    half-cell interlacing leaves only even total image parity and gives
    (product(A)+product(B))/2-1. It is not a relative weak-mode bound.
    """
    _geometry(np.zeros((1, 3)), 1., grid)
    modes = _modes(modes, grid)
    t = modes / grid
    cosine = np.cos(np.pi * t)
    log_base = -4 * np.log(np.sinc(t))
    log_ordinary = np.sum(log_base + np.log((2 + np.cos(2*np.pi*t)) / 3), axis=1)
    if not interlaced:
        return np.maximum(0., np.expm1(log_ordinary))
    log_alternating = np.sum(log_base + np.log(cosine * (5 + cosine*cosine) / 6), axis=1)
    return np.maximum(0., .5 * (np.expm1(log_ordinary) + np.expm1(log_alternating)))


def _pcs_extract(positions, box, modes, grid, interlaced, deconvolve):
    positions = _geometry(positions, box, grid)
    modes = _modes(modes, grid)
    shifts = ((0., 0., 0.), (.5, .5, .5)) if interlaced else ((0., 0., 0.),)
    result = np.zeros(len(modes), dtype=np.complex128)
    mass_errors = []
    indices = tuple((modes % grid).T)
    for shift in shifts:
        counts = deposit_pcs(positions, box, grid, shift)
        mass_errors.append(float(abs(counts.sum(dtype=np.float64) / len(positions) - 1)))
        transformed = np.fft.fftn(counts)
        phase = np.exp(-2j * np.pi * (modes @ np.asarray(shift)) / grid)
        result += transformed[indices] * phase / len(positions)
        del transformed, counts
    result /= len(shifts)
    if deconvolve:
        result /= pcs_window(modes, grid)
    return result, mass_errors


def pcs_coefficients(positions, box, modes, grid=AnalysisGrid,
                     interlaced=True, deconvolve=True):
    """Physical-origin coefficients measured using interlaced cubic B-splines."""
    return _pcs_extract(positions, box, modes, grid, interlaced, deconvolve)[0]


def direct_coefficients(positions, box, modes):
    """Mesh-free normalized Fourier sum, processed in bounded mode chunks."""
    positions = _geometry(positions, box)
    modes = _modes(modes)
    # Wrap first to avoid needlessly large exponential arguments for valid
    # particle coordinates outside the primary image.
    phase_position = np.remainder(positions, box) * (2 * np.pi / box)
    result = np.empty(len(modes), dtype=np.complex128)
    for start in range(0, len(modes), 16):
        phase = phase_position @ modes[start:start + 16].T
        result[start:start + 16] = np.exp(-1j * phase).mean(axis=0)
    return result


def dephase(coefficients, modes, box, translation):
    """Undo a rigid particle translation: delta'_m=exp(-ik.Delta)*delta_m."""
    modes = _modes(modes)
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    shift = np.asarray(translation, dtype=float)
    if (coefficients.shape != (len(modes),) or not np.isfinite(coefficients).all()
            or shift.shape != (3,) or not np.isfinite(shift).all()
            or not np.isfinite(box) or box <= 0):
        raise ValueError("Invalid coefficients or physical translation")
    return coefficients * np.exp(2j * np.pi * (modes @ np.remainder(shift / box, 1.)))


def shell_statistics(coefficients, modes, box, edges=ShellEdges):
    """Physical P(k)=L^3 mean(|delta_m|^2); unique +/- pairs count once.

    No shot-noise subtraction is made: the evolved lattice is not a Poisson
    sample. These are deterministic realization statistics, not theory errors.
    Empty shells are omitted, preserving increasing radial shell order.
    """
    modes = _modes(modes)
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    edges = np.asarray(edges, dtype=float)
    if (coefficients.shape != (len(modes),) or not np.isfinite(coefficients).all()
            or not np.isfinite(box) or box <= 0 or edges.ndim != 1 or len(edges) < 2
            or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0)
            or edges[0] < 0):
        raise ValueError("Invalid spectrum or shell geometry")
    representatives = [tuple(mode if next(v for v in mode if v) > 0 else -mode)
                       for mode in modes]
    if len(set(representatives)) != len(representatives):
        raise ValueError("Shell statistics require unique +/- mode pairs")
    radius = np.linalg.norm(modes, axis=1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (radius >= lower) & (radius < upper)
        count = int(selected.sum())
        if count:
            power_sum = float(np.vdot(coefficients[selected], coefficients[selected]).real)
            rows.append({"lower": float(lower), "upper": float(upper), "pairs": count,
                         "k_mean": float(radius[selected].mean() * 2 * np.pi / box),
                         "delta_power_sum": power_sum, "power_mean": box**3 * power_sum / count})
    return rows


def analyze_spectrum(positions, box, grid=AnalysisGrid, max_mode=12,
                     direct_max_mode=4, translation=None):
    """Return FFT coefficients, exact direct low modes, shells and audit metrics.

    The direct extraction gate is fixed before the study: the low-mode residual
    norm must be <= max(1e-12, 1e-3*direct norm). There is no universal relative
    alias guarantee; reported absolute alias bounds explain this limitation.
    The returned direct arrays, not the FFT approximation, should be used for
    qualified low-band scientific comparisons. No high-band pass is implied.
    """
    _geometry(positions, box, grid)
    if max_mode >= grid / 2:
        raise ValueError("Maximum mode must be strictly inside the analysis Nyquist band")
    modes = unique_modes(max_mode)
    low_modes = unique_modes(direct_max_mode)
    if direct_max_mode > max_mode:
        raise ValueError("Direct band must be contained in the measured band")
    coefficients, mass_errors = _pcs_extract(positions, box, modes, grid, True, True)
    low_mask = np.sum(modes*modes, axis=1) <= direct_max_mode**2
    if not np.array_equal(modes[low_mask], low_modes):
        raise ValueError("Internal direct-mode ordering mismatch")
    direct = direct_coefficients(positions, box, low_modes)
    absolute = float(np.linalg.norm(coefficients[low_mask] - direct))
    reference = float(np.linalg.norm(direct))
    allowed = max(DirectAbsoluteFloor, DirectRelativeLimit * reference)
    bound = pcs_alias_bound(modes, grid)
    diagnostics = {
        "analysis_grid": int(grid), "max_mode": int(max_mode),
        "assignment": "PCS", "assignment_order": 4,
        "interlacing": "mean of node grids shifted by (0,0,0) and (1/2,1/2,1/2) cells",
        "window_deconvolution": "product sinc(m/grid)^4",
        "simulation_assignment": "unchanged CIC; PCS is analysis only",
        "direct_max_mode": int(direct_max_mode), "direct_pairs": len(low_modes),
        "direct_absolute_error": absolute, "direct_reference_norm": reference,
        "direct_complex_relative": absolute / reference if reference > DirectAbsoluteFloor else None,
        "direct_relative_limit": DirectRelativeLimit,
        "direct_absolute_floor": DirectAbsoluteFloor, "direct_allowed_error": allowed,
        "direct_gate_passed": bool(absolute <= allowed),
        "dc_mass_relative_errors": mass_errors, "mass_relative_limit": MassRelativeLimit,
        "mass_gate_passed": bool(max(mass_errors) <= MassRelativeLimit),
        "alias_absolute_bound_max": float(bound.max()),
        "alias_absolute_bound_direct_band_max": float(bound[low_mask].max()),
        "alias_bound_meaning": "Absolute per-coefficient bound for interlaced, PCS-deconvolved equal nonnegative masses; not a relative guarantee",
        "qualified_measurement": "direct low-band coefficients only",
        "higher_band_status": "FFT characterization; low-band test does not qualify higher bands",
        "power_convention": "P(k)=L^3 mean(abs(delta_hat)^2), each +/- pair once; no shot-noise subtraction",
        "translation_removed": None,
    }
    if translation is not None:
        coefficients = dephase(coefficients, modes, box, translation)
        direct = dephase(direct, low_modes, box, translation)
        diagnostics["translation_removed"] = np.asarray(translation, dtype=float).tolist()
    return {"modes": modes, "coefficients": coefficients,
            "direct_low_modes": low_modes, "direct_low_coefficients": direct,
            "diagnostics": diagnostics,
            "shells": shell_statistics(coefficients, modes, box),
            "direct_low_shells": shell_statistics(direct, low_modes, box)}
