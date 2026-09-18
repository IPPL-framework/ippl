#!/usr/bin/env python3
"""SI spectral half-space ChDR benchmark; see README.md for normalization.

The electron remains in vacuum at x=a and moves along +z. Each (omega, ky)
problem is one-dimensional in x. Integration over ky recovers a point source.
This is an analytical reference with numerical quadrature, not a PIC solver.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.constants import c as LIGHT_SPEED, epsilon_0 as EPSILON_0
from scipy.constants import mu_0 as MU_0, elementary_charge as ELEMENTARY_CHARGE
from scipy.constants import physical_constants
from scipy.integrate import quad_vec

REST_ENERGY_MEV = physical_constants["electron mass energy equivalent in MeV"][0]
VACUUM_IMPEDANCE = math.sqrt(MU_0 / EPSILON_0)
PI = math.pi


@dataclass(frozen=True)
class BeamCase:
    gapMm: float = 1.0
    energyMeV: float = 60.0
    pulsePs: float = 10.0
    chargeNc: float = 5.0
    epsilonR: float = 2.13

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.electronCount < 1:
            raise ValueError("Bunch charge must correspond to at least one electron")

    @property
    def gamma(self):
        return 1.0 + self.energyMeV / REST_ENERGY_MEV

    @property
    def beta(self):
        g = self.gamma
        return math.sqrt(g - 1.0) * math.sqrt(g + 1.0) / g

    @property
    def velocity(self):
        return self.beta * LIGHT_SPEED

    @property
    def gap(self):
        return self.gapMm * 1e-3

    @property
    def sigmaTime(self):
        return self.pulsePs * 1e-12

    @property
    def electronCount(self):
        return self.chargeNc * 1e-9 / ELEMENTARY_CHARGE

    @property
    def propagates(self):
        return self.epsilonR * self.beta**2 > 1.0

    @property
    def angleDeg(self):
        if not self.propagates:
            return None
        return math.degrees(math.acos(1.0 / (math.sqrt(self.epsilonR) * self.beta)))


@dataclass
class ModeFields:
    incidentE: np.ndarray
    incidentH: np.ndarray
    reflectedE: np.ndarray
    reflectedH: np.ndarray
    transmittedE: np.ndarray
    transmittedH: np.ndarray
    kappa: float
    normalWave: complex
    kz: float


def solveMode(case, frequencyHz, ky, sourceCharge=-ELEMENTARY_CHARGE):
    """Double-Fourier field amplitudes at x=0, after removing exp(i*kz*z).

    E_hat has units V s; H_hat has units A s. Normal dielectric wave number
    is -normalWave. The principal root gives propagation/decay into x<0.
    """
    if not math.isfinite(frequencyHz) or frequencyHz <= 0 or not math.isfinite(ky):
        raise ValueError("Frequency must be positive and ky finite")
    omega = 2 * PI * frequencyHz
    k0 = omega / LIGHT_SPEED
    kz = omega / case.velocity
    kappa = math.hypot(ky, omega / (case.gamma * case.velocity))
    kt = math.hypot(ky, kz)
    normalWave = complex(np.sqrt(complex((case.epsilonR - 1) * k0**2 - kappa**2)))
    normal = np.array([1.0, 0.0, 0.0])
    tangent = np.array([0.0, ky / kt, kz / kt])
    transverse = np.array([0.0, -kz / kt, ky / kt])

    # Lorenz-gauge vacuum potentials: A_z = v*phi/c^2. The transformed
    # charge/current sources are q/v*delta(x-a) and q*delta(x-a), respectively.
    phi = sourceCharge * math.exp(-kappa * case.gap) / (2 * EPSILON_0 * case.velocity * kappa)
    incidentE = phi * np.array([-kappa, -1j * ky, -1j * omega / (case.gamma**2 * case.velocity)])
    incidentH = phi * EPSILON_0 * case.velocity * np.array([1j * ky, -kappa, 0.0])
    electricTE = np.dot(incidentE, transverse)
    magneticTM = np.dot(incidentH, transverse)
    reflectionTE = (1j * kappa - normalWave) / (1j * kappa + normalWave)
    reflectionTM = (1j * case.epsilonR * kappa - normalWave) / (1j * case.epsilonR * kappa + normalWave)

    def planeFields(kx, epsilonR, electricAmplitude, magneticAmplitude):
        electric = electricAmplitude * transverse + magneticAmplitude / (omega * EPSILON_0 * epsilonR) * (kx * tangent - kt * normal)
        magnetic = magneticAmplitude * transverse + electricAmplitude / (omega * MU_0) * (kt * normal - kx * tangent)
        return electric, magnetic

    reflectedE, reflectedH = planeFields(1j * kappa, 1.0, reflectionTE * electricTE, reflectionTM * magneticTM)
    transmittedE, transmittedH = planeFields(-normalWave, case.epsilonR, (1 + reflectionTE) * electricTE, (1 + reflectionTM) * magneticTM)
    return ModeFields(incidentE, incidentH, reflectedE, reflectedH, transmittedE, transmittedH, kappa, normalWave, kz)


def spectralRadiation(case, frequencyHz, relativeTolerance=1e-8):
    """Return single-electron (flux, work, error) in J / (m Hz).

    Flux into the dielectric and work against the reflected longitudinal
    field are independent observables. Integration is over propagating ky.
    """
    if not math.isfinite(frequencyHz) or frequencyHz <= 0:
        raise ValueError("Frequency must be finite and positive")
    if not case.propagates:
        return np.zeros(3)
    omega = 2 * PI * frequencyHz
    k0 = omega / LIGHT_SPEED
    cutoff = k0 * math.sqrt(case.epsilonR - case.beta**-2)
    kappa0 = omega / (case.gamma * case.velocity)

    def integrand(theta):
        # ky=cutoff*sin(theta) regularizes the outgoing-wave branch point.
        ky = cutoff * math.sin(theta)
        mode = solveMode(case, frequencyHz, ky, sourceCharge=-1.0)
        flux = -np.cross(mode.transmittedE, np.conjugate(mode.transmittedH))[0].real
        # -q*Re(E_ref,z) at x=a, evaluated for q=-1 C before rescaling.
        work = mode.reflectedE[2].real * math.exp(-mode.kappa * case.gap)
        return 2 * cutoff * math.cos(theta) * np.array([flux, work])

    split = min(PI / 4, kappa0 / cutoff)
    values, error, info = quad_vec(integrand, 0, PI / 2, epsabs=1e-200,
                                  epsrel=relativeTolerance, points=[split], full_output=True)
    if not info.success:
        raise RuntimeError(f"Radiation quadrature failed at {frequencyHz:g} Hz: {info.message}")
    scale = ELEMENTARY_CHARGE**2 / PI
    return np.array([values[0] * scale, values[1] * scale, error * scale])


def reconstructFields(case, frequencyHz, x, y=0.0, z=0.0, relativeTolerance=1e-8):
    """Time-Fourier point-electron fields at points (x,y,z), in SI.

    Includes propagating AND evanescent ky. E_tilde: V s/m, H_tilde: A s/m.
    Scalar or broadcastable coordinate arrays accepted. x=0 uses vacuum
    trace; x=a is excluded to avoid conditionally convergent source-plane
    integrals. Probes away from that plane suffice for this first benchmark.
    """
    if not math.isfinite(frequencyHz) or frequencyHz <= 0:
        raise ValueError("Frequency must be finite and positive")
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    if not all(np.all(np.isfinite(v)) for v in (x, y, z)):
        raise ValueError("Probe coordinates must be finite")
    distance = np.where(x < 0, case.gap - x, np.abs(x - case.gap))
    if np.any(distance == 0):
        raise ValueError("Use probe positions away from the electron plane x=a")
    omega = 2 * PI * frequencyHz
    k0 = omega / LIGHT_SPEED
    kz = omega / case.velocity
    kScale = 1.0 / float(np.min(distance))
    kappa0 = omega / (case.gamma * case.velocity)
    vacuum = x >= 0

    def signedIntegrand(ky):
        mode = solveMode(case, frequencyHz, ky, sourceCharge=-1.0)
        electric = np.zeros(x.shape + (3,), dtype=complex)
        magnetic = np.zeros_like(electric)
        # Use masking rather than np.where to avoid evaluating growing
        # exponentials on the wrong side of the interface.
        if np.any(vacuum):
            xv = x[vacuum]
            phi = -np.exp(-mode.kappa * np.abs(xv - case.gap)) / (2 * EPSILON_0 * case.velocity * mode.kappa)
            sign = np.sign(xv - case.gap)
            sourceE = np.stack([sign * mode.kappa * phi, -1j * ky * phi,
                               -1j * omega / (case.gamma**2 * case.velocity) * phi], axis=-1)
            sourceH = EPSILON_0 * case.velocity * np.stack([1j * ky * phi, sign * mode.kappa * phi, np.zeros_like(phi)], axis=-1)
            factor = np.exp(-mode.kappa * xv)[..., None]
            electric[vacuum] = sourceE + factor * mode.reflectedE
            magnetic[vacuum] = sourceH + factor * mode.reflectedH
        if np.any(~vacuum):
            factor = np.exp(-1j * mode.normalWave * x[~vacuum])[..., None]
            electric[~vacuum] = factor * mode.transmittedE
            magnetic[~vacuum] = factor * mode.transmittedH
        phase = np.exp(1j * (ky * y + kz * z))[..., None]
        # Impedance scaling prevents E dominating H in the quadrature norm.
        return np.concatenate([electric * phase, VACUUM_IMPEDANCE * magnetic * phase], axis=-1)

    def integrand(u):
        ky = kScale * u
        return kScale * (signedIntegrand(ky) + signedIntegrand(-ky))

    points = [kappa0 / kScale]
    if case.propagates:
        points.append(k0 * math.sqrt(case.epsilonR - case.beta**-2) / kScale)
    values, error, info = quad_vec(integrand, 0, np.inf, epsabs=1e-200,
                                  epsrel=relativeTolerance, points=sorted(set(points)), full_output=True)
    if not info.success:
        raise RuntimeError(f"Field quadrature failed at {frequencyHz:g} Hz: {info.message}")
    values *= ELEMENTARY_CHARGE / (2 * PI)
    return values[..., :3], values[..., 3:] / VACUUM_IMPEDANCE


def bunchSpectra(case, frequenciesHz, singleSpectrum):
    """Expected spectrum for independent Gaussian arrival times, same track."""
    frequenciesHz = np.asarray(frequenciesHz)
    formFactorSquared = np.exp(-(2 * PI * frequenciesHz * case.sigmaTime)**2)
    n = case.electronCount
    incoherent = n * np.asarray(singleSpectrum)
    coherent = n * (n - 1) * formFactorSquared * np.asarray(singleSpectrum)
    return formFactorSquared, incoherent, coherent, incoherent + coherent


def writeCsv(path, columns):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def fieldColumns(electric, magnetic, suffix=""):
    result = {}
    for prefix, fields, unit in [("E", electric, "V_s_per_m"), ("H", magnetic, "A_s_per_m")]:
        for i, axis in enumerate("xyz"):
            for part in ["real", "imag"]:
                result[f"{prefix}{axis}_{part}{suffix}_{unit}"] = getattr(fields[..., i], part)
    return result


def makePlots(output, frequencies, results, profileFrequencyHz):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 220, "axes.grid": True, "grid.alpha": 0.2})
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
    anyRadiation = False
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, entry in enumerate(results):
        case, data, caseDir, profile = entry
        color = colors[index % len(colors)]
        label = f"a={case.gapMm:g} mm, K={case.energyMeV:g} MeV, rms={case.pulsePs:g} ps"
        positive = data["single"] > 0
        anyRadiation |= bool(np.any(positive))
        if np.any(positive):
            axes[0].loglog(frequencies[positive] / 1e9, data["single"][positive] * 1e9, label=label, color=color)
            axes[1].loglog(frequencies[positive] / 1e9, data["total"][positive] * 1e9, label=label, color=color)
        axes[2].semilogx(frequencies / 1e9, data["formFactor"], label=label, color=color)
        if profile is not None:
            x, electric, magnetic = profile
            profileFig, profileAxes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
            for ax, component, title in zip(profileAxes, [0, 2], ["x", "z"]):
                ax.plot(x * 1e3, electric[:, component].real, label="Real part", lw=1.8)
                ax.plot(x * 1e3, electric[:, component].imag, label="Imaginary part", ls="--", lw=1.8)
                ax.axvspan(x.min() * 1e3, 0, color="#176b87", alpha=0.09)
                ax.axvline(0, color="0.4", lw=0.8)
                ax.set_ylabel(rf"$\widetilde{{E}}_{title}$ [V s/m]")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
                ax.legend(loc="best")
            profileAxes[-1].set_xlabel("Interface-normal coordinate x [mm]; dielectric x < 0")
            profileFig.suptitle(f"Single-electron complex field, {profileFrequencyHz / 1e9:g} GHz\na={case.gapMm:g} mm, K={case.energyMeV:g} MeV, epsilon_r={case.epsilonR:g}; y=z=0")
            for extension in ["png", "svg"]:
                profileFig.savefig(caseDir / f"field_profile.{extension}")
            plt.close(profileFig)
    for ax in axes[:2]:
        ax.set_ylabel(r"$d^2W/(dz\,df)$ [J m$^{-1}$ GHz$^{-1}$]")
    if not anyRadiation:
        for ax in axes[:2]:
            ax.text(0.5, 0.5, "No propagating Cherenkov modes", transform=ax.transAxes, ha="center")
    axes[0].set_title("Single electron: energy entering the dielectric per path length")
    axes[1].set_title("Gaussian bunch: expected coherent + incoherent spectrum")
    axes[2].set_ylabel(r"$|F(f)|^2$")
    axes[2].set_xlabel("Frequency [GHz]")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(loc="upper right", fontsize=8)
    firstCase = results[0][0]
    fig.suptitle("Planar ChDR | infinite, lossless dielectric half-space\n"
                 f"epsilon_r={firstCase.epsilonR:g}; bunch charge magnitude={firstCase.chargeNc:g} nC")
    for extension in ["png", "svg"]:
        fig.savefig(output / f"spectra.{extension}")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-mm", nargs="+", type=float, default=[1.0], help="Vacuum gap a [mm]; one or more values (default: illustrative 1 mm)")
    parser.add_argument("--energy-mev", nargs="+", type=float, default=[60.0], help="Electron KINETIC energy [MeV]; one or more values")
    parser.add_argument("--pulse-ps", nargs="+", type=float, default=[10.0], help="Gaussian RMS arrival-time duration [ps]; one or more values; FWHM=2.35482*rms")
    parser.add_argument("--charge-nc", type=float, default=5.0, help="Magnitude of bunch charge [nC]")
    parser.add_argument("--epsilon-r", type=float, default=2.13, help="Positive, real, constant relative permittivity")
    parser.add_argument("--fmin-ghz", type=float, default=0.2)
    parser.add_argument("--fmax-ghz", type=float, default=200.0)
    parser.add_argument("--points", type=int, default=120, help="Log-spaced frequency samples")
    parser.add_argument("--probe-depth-mm", type=float, default=1.0, help="Field probe at x=-depth, y=z=0")
    parser.add_argument("--profile-frequency-ghz", type=float, default=16.0)
    parser.add_argument("--profile-depth-mm", type=float, default=20.0)
    parser.add_argument("--rtol", type=float, default=1e-8, help="Relative quadrature tolerance")
    parser.add_argument("--no-fields", action="store_true", help="Only calculate radiated spectra; faster parameter scans")
    parser.add_argument("--no-plots", action="store_true", help="Only export numerical results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args(argv)
    for value in [args.fmin_ghz, args.fmax_ghz, args.probe_depth_mm, args.profile_frequency_ghz, args.profile_depth_mm, args.rtol]:
        if not math.isfinite(value) or value <= 0:
            parser.error("Frequencies, depths and rtol must be finite and positive")
    if args.fmax_ghz <= args.fmin_ghz or args.points < 2 or args.rtol >= 1:
        parser.error("Require fmax>fmin, points>=2 and 0<rtol<1")
    try:
        cases = [BeamCase(a, energy, pulse, args.charge_nc, args.epsilon_r)
                 for a, energy, pulse in product(args.gap_mm, args.energy_mev, args.pulse_ps)]
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frequencies = np.geomspace(args.fmin_ghz * 1e9, args.fmax_ghz * 1e9, args.points)
    results = []
    # Pulse-duration scans reuse the same single-electron calculation.
    cache = {}
    for index, case in enumerate(cases, 1):
        name = f"case_{index:03d}_a{case.gapMm:g}mm_K{case.energyMeV:g}MeV_rms{case.pulsePs:g}ps"
        caseDir = output / name
        caseDir.mkdir(exist_ok=True)
        print(f"[{index}/{len(cases)}] {name}", flush=True)
        key = (case.gapMm, case.energyMeV)
        if key not in cache:
            radiation = np.array([spectralRadiation(case, f, args.rtol) for f in frequencies])
            probe = profile = None
            if not args.no_fields:
                fields = [reconstructFields(case, f, -args.probe_depth_mm * 1e-3, relativeTolerance=args.rtol) for f in frequencies]
                probe = (np.array([item[0] for item in fields]), np.array([item[1] for item in fields]))
                x = np.concatenate([np.linspace(-args.profile_depth_mm * 1e-3, -min(case.gap, args.profile_depth_mm * 1e-3) * 1e-9, 72), np.linspace(0, 0.8 * case.gap, 36)])
                electric, magnetic = reconstructFields(case, args.profile_frequency_ghz * 1e9, x, relativeTolerance=args.rtol)
                profile = (x, electric, magnetic)
            cache[key] = radiation, probe, profile
        radiation, probe, profile = cache[key]
        single, work, errors = radiation.T
        formFactor, incoherent, coherent, total = bunchSpectra(case, frequencies, single)
        columns = {"frequency_Hz": frequencies, "frequency_GHz": frequencies / 1e9,
                   "single_electron_J_per_m_per_Hz": single, "work_check_J_per_m_per_Hz": work,
                   "quadrature_error_estimate_J_per_m_per_Hz": errors,
                   "form_factor_squared": formFactor, "bunch_incoherent_J_per_m_per_Hz": incoherent,
                   "bunch_coherent_cross_term_J_per_m_per_Hz": coherent, "bunch_total_J_per_m_per_Hz": total}
        writeCsv(caseDir / "spectrum.csv", columns)
        if probe is not None:
            electric, magnetic = probe
            meanFactor = case.electronCount * np.sqrt(formFactor)[:, None]
            writeCsv(caseDir / "probe_fields.csv", {"frequency_Hz": frequencies,
                     **fieldColumns(electric, magnetic, "_single"),
                     **fieldColumns(meanFactor * electric, meanFactor * magnetic, "_mean_bunch")})
        if profile is not None:
            x, electric, magnetic = profile
            writeCsv(caseDir / "field_profile.csv", {"x_m": x, **fieldColumns(electric, magnetic)})
        positive = single > 0
        balance = float(np.max(np.abs(single[positive] - work[positive]) / single[positive])) if np.any(positive) else 0.0
        metadata = {"parameters": asdict(case), "gamma": case.gamma, "beta": case.beta,
                    "electron_count": case.electronCount, "sigma_z_mm": case.velocity * case.sigmaTime * 1e3,
                    "form_factor_e_minus_one_GHz": 1 / (2 * PI * case.sigmaTime) / 1e9,
                    "internal_cherenkov_angle_deg": case.angleDeg,
                    "flux_work_max_relative_difference": balance,
                    "sampled_band_bunch_energy_J_per_m": float(np.trapezoid(total, frequencies)),
                    "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "model": "Infinite, planar, lossless, nondispersive half-space; prescribed identical transverse tracks; independent Gaussian arrival times",
                    "normalization": "Forward Fourier transform integral dt dy; inverse /(2pi)^2. Spectrum uses positive f in Hz and is per metre of trajectory.",
                    "field_probe_m": [-args.probe_depth_mm * 1e-3, 0, 0],
                    "reference": "https://arxiv.org/abs/2105.01111, Section III; independent SI derivation in README.md"}
        (caseDir / "parameters.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        results.append((case, {"single": single, "total": total, "formFactor": formFactor}, caseDir, profile))
        print(f"  angle={case.angleDeg}, sigma_z={metadata['sigma_z_mm']:.6g} mm, flux/work relative difference={balance:.3g}", flush=True)
    if not args.no_plots:
        makePlots(output, frequencies, results, args.profile_frequency_ghz * 1e9)
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
