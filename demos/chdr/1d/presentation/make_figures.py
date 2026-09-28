#!/usr/bin/env python3
"""Generate the reproducible figures used by the 1-D ChDR presentation.

The figures use the analytical spectral half-space reference in the parent
directory.  Spectral energy is reported inside the dielectric per unit path
length.  A common 60 MeV, 1 nC, 1 mm rms bunch supports the full-bunch-length
case and the optical microbunching case.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIRECTORY = HERE.parent
FIGURE_DIRECTORY = HERE / "figures"
sys.path.insert(0, str(MODEL_DIRECTORY))

from chdr_1d import (  # noqa: E402
    BeamCase,
    LIGHT_SPEED,
    PI,
    UniformDensityEstimate,
    bunchSpectra,
    fixedCountShotNoiseFluctuation,
    leastEvanescentLength,
    spectralRadiation,
)


def configureMatplotlib():
    """Use deterministic, readable non-interactive plotting defaults."""
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "legend.fontsize": 8,
        "savefig.dpi": 220,
    })
    return plt


def calculateSpectrum(case, wavelengthsUm):
    """Return the single-electron and fixed-N statistical spectra."""
    frequenciesHz = np.sort(LIGHT_SPEED / (np.asarray(wavelengthsUm) * 1e-6))
    radiation = np.array([spectralRadiation(case, frequency, 1e-9) for frequency in frequenciesHz])
    single = radiation[:, 0]
    formFactor, selfTerm, coherent, total = bunchSpectra(case, frequenciesHz, single)
    fluctuation = fixedCountShotNoiseFluctuation(case, frequenciesHz, single)
    return frequenciesHz, single, formFactor, selfTerm, coherent, total, fluctuation


def makeOpticalSpectrum(plt, case, density):
    """Plot the smooth optical half-space spectrum and the density diagnostic."""
    wavelengthsUm = np.geomspace(0.25, 2.0, 100)
    frequency, single, formFactor, selfTerm, coherent, total, fluctuation = calculateSpectrum(case, wavelengthsUm)
    densityFrequency = LIGHT_SPEED / density.cubicSpacing
    frequencyThz = frequency / 1e12

    # The wide aspect ratio keeps both panels legible when placed on a Beamer slide.
    figure, axes = plt.subplots(2, 1, figsize=(8.6, 4.0), sharex=True, layout="constrained")
    axes[0].loglog(frequencyThz, single * 1e12, color="#176b87", lw=2.0)
    axes[0].set_ylabel(r"$\mathcal{W}_1$ [J m$^{-1}$ THz$^{-1}$]")
    axes[0].text(0.02, 0.08, "one electron", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=10, color="0.25")

    axes[1].loglog(frequencyThz, fluctuation * 1e12, color="#b35c1e", lw=2.0,
                   label=r"$\mathcal{W}_{\rm fluct}=N(1-|F|^2)\mathcal{W}_1$")
    axes[1].loglog(frequencyThz, total * 1e12, color="#176b87", lw=1.25, ls="--",
                   label=r"$\langle\mathcal{W}_N\rangle=[N+N(N-1)|F|^2]\mathcal{W}_1$")
    axes[1].set_ylabel(r"Spectral energy [J m$^{-1}$ THz$^{-1}$]")
    axes[1].set_xlabel("Frequency [THz]")
    axes[1].text(0.02, 0.08, r"1 nC bunch, $\sigma_z=1$ mm", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontsize=10, color="0.25")

    for axis in axes:
        axis.axvline(densityFrequency / 1e12, color="0.25", ls=":", lw=1.2)
    axes[1].plot([], [], color="0.25", ls=":", lw=1.2,
                 label=r"$c/n^{-1/3}$ diagnostic, not a radiation line")
    axes[1].legend(loc="best")

    for suffix in ["png", "pdf"]:
        figure.savefig(FIGURE_DIRECTORY / f"optical_shot_noise_spectrum.{suffix}")
    plt.close(figure)

    # Evaluate the caption value exactly, independently of the plotted sampling.
    exact = calculateSpectrum(case, [0.5])
    return {
        "frequency_0p5_thz": exact[0][0] / 1e12,
        "single_0p5_j_per_m_hz": exact[1][0],
        "noise_0p5_j_per_m_hz": exact[6][0],
        "form_factor_0p5": exact[2][0],
        "coherent_0p5_j_per_m_hz": exact[4][0],
        "total_0p5_j_per_m_hz": exact[5][0],
    }


def makeGapDependence(plt, case):
    """Plot the optical coupling scale versus beam--surface gap."""
    wavelengthUm = 0.5
    frequency = LIGHT_SPEED / (wavelengthUm * 1e-6)
    gapsUm = np.geomspace(1.0, 1000.0, 55)
    single = np.array([
        spectralRadiation(replace(case, gapMm=gap * 1e-3), frequency, 1e-9)[0]
        for gap in gapsUm
    ])
    # This is numerically equal to N W1 in the optical band of this case, but
    # retain the exact fixed-N expression if the pulse length changes later.
    noise = fixedCountShotNoiseFluctuation(
        case, np.full_like(gapsUm, frequency, dtype=float), single
    )
    couplingLengthUm = float(leastEvanescentLength(case, frequency) * 1e6)

    figure, axis = plt.subplots(figsize=(7.6, 3.7), layout="constrained")
    axis.loglog(gapsUm, noise * 1e12, color="#b35c1e", lw=2.0,
                label=r"fixed-$N$ fluctuation at $\lambda_0=0.5\,\mu$m")
    axis.axvline(couplingLengthUm, color="#176b87", ls="--", lw=1.4,
                label=rf"$\ell_\perp={couplingLengthUm:.2f}\,\mu$m")
    axis.set_xlabel("Vacuum gap $a$ [$\\mu$m]")
    axis.set_ylabel(r"$\mathcal{W}_{\rm fluct}$ [J m$^{-1}$ THz$^{-1}$]")
    axis.set_title("Planar vacuum-field coupling at 60 MeV")
    axis.legend(loc="best")
    for suffix in ["png", "pdf"]:
        figure.savefig(FIGURE_DIRECTORY / f"gap_dependence_0p5um.{suffix}")
    plt.close(figure)

    singleAtMillimetre = spectralRadiation(replace(case, gapMm=1.0), frequency, 1e-9)[0]
    return {
        "coupling_length_um": couplingLengthUm,
        "single_at_1mm_j_per_m_hz": singleAtMillimetre,
        "minimum_power_factor_1mm": np.exp(-2.0 * 1000.0 / couplingLengthUm),
    }


def makeCoherenceCrossover(plt, case):
    """Plot the coherent-to-self-term ratio for the Gaussian bunch envelope."""
    frequency = np.geomspace(0.1e9, 3.0e12, 500)
    formFactor = np.exp(-(2 * PI * frequency * case.sigmaTime) ** 2)
    ratio = (case.electronCount - 1.0) * formFactor
    displayedRatio = np.where(ratio >= 1e-6, ratio, np.nan)
    crossing = np.sqrt(np.log(case.electronCount - 1.0)) / (2 * PI * case.sigmaTime)

    figure, axis = plt.subplots(figsize=(7.6, 3.7), layout="constrained")
    axis.loglog(frequency / 1e9, displayedRatio, color="#176b87", lw=2.0,
                label=r"coherent cross term / $N\mathcal{W}_1$")
    axis.axhline(1.0, color="0.25", ls=":", lw=1.2, label="equal coherent and self terms")
    axis.axvline(crossing / 1e9, color="#b35c1e", ls="--", lw=1.4,
                label=rf"crossover {crossing / 1e9:.1f} GHz")
    axis.set_xlabel("Frequency [GHz]")
    axis.set_ylabel(r"$(N-1)|F|^2$")
    axis.set_ylim(1e-6, 1e11)
    axis.set_title("Gaussian bunch statistics for 1 nC and $\\sigma_z=1$ mm")
    axis.text(0.02, 0.05, r"curve omitted below $10^{-6}$", transform=axis.transAxes,
              ha="left", va="bottom", fontsize=8, color="0.35")
    axis.legend(loc="best")
    for suffix in ["png", "pdf"]:
        figure.savefig(FIGURE_DIRECTORY / f"coherence_crossover.{suffix}")
    plt.close(figure)
    return {"coherence_crossover_ghz": crossing / 1e9}


def makeBunchLengthFormFactor(plt, case):
    """Show the Gaussian longitudinal shape factor used for length diagnostics.

    This dimensionless factor multiplies the one-electron spectral response;
    it is not itself an emitted photon or energy spectrum.
    """
    frequency = np.linspace(0.0, 160e9, 500)
    scaleHz = 1.0 / (2 * PI * case.sigmaTime)
    formFactor = np.exp(-(frequency / scaleHz) ** 2)

    figure, axis = plt.subplots(figsize=(8.6, 3.3), layout="constrained")
    axis.plot(frequency / 1e9, formFactor, color="#176b87", lw=2.3,
              label=r"Gaussian $|F(f)|^2=\exp[-(2\pi f\sigma_t)^2]$")
    axis.axvline(scaleHz / 1e9, color="#b35c1e", ls="--", lw=1.5,
                 label=rf"$f_\ast=1/(2\pi\sigma_t)={scaleHz / 1e9:.2f}$ GHz")
    axis.axhline(np.exp(-1), color="0.4", ls=":", lw=1.2,
                 label=r"$|F|^2=1/e$")
    axis.plot(scaleHz / 1e9, np.exp(-1), "o", color="#b35c1e", ms=5)
    axis.set(xlim=(0, 160), ylim=(0, 1.05), xlabel="Frequency [GHz]",
             ylabel=r"Longitudinal form factor $|F|^2$",
             title=r"Case 1: full-bunch-length response, $\sigma_z=1$ mm")
    axis.legend(loc="upper right")
    axis.text(0.98, 0.46, "Dimensionless shape factor;\ncombine with the radiation response",
              transform=axis.transAxes, ha="right", va="top", fontsize=9, color="0.35")
    for suffix in ["png", "pdf"]:
        figure.savefig(FIGURE_DIRECTORY / f"bunch_length_form_factor.{suffix}")
    plt.close(figure)
    return {
        "form_factor_scale_ghz": scaleHz / 1e9,
        "form_factor_wavelength_mm": LIGHT_SPEED / scaleHz * 1e3,
    }


def texScientific(value, digits=4):
    """Format a scalar as a TeX mantissa times an unambiguous power of ten."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def writeNumbers(case, density, optical, gap, coherence, bunchLength):
    """Write TeX macros so the slides and numerical figures stay consistent."""
    text = f"""% Generated by make_figures.py; do not edit by hand.
\\newcommand{{\\BeamEnergyMeV}}{{{case.energyMeV:g}}}
\\newcommand{{\\BeamGamma}}{{{case.gamma:.3f}}}
\\newcommand{{\\BeamBeta}}{{{case.beta:.8f}}}
\\newcommand{{\\BunchChargeNc}}{{{case.chargeNc:g}}}
\\newcommand{{\\ElectronCount}}{{{texScientific(case.electronCount)}}}
\\newcommand{{\\SigmaZmm}}{{{case.velocity * case.sigmaTime * 1e3:.4f}}}
\\newcommand{{\\SigmaTimePs}}{{{case.pulsePs:.5f}}}
\\newcommand{{\\DensityPerMThree}}{{{texScientific(density.numberDensity)}}}
\\newcommand{{\\DensitySpacingUm}}{{{density.cubicSpacing * 1e6:.4f}}}
\\newcommand{{\\DensityFrequencyTHz}}{{{LIGHT_SPEED / density.cubicSpacing / 1e12:.2f}}}
\\newcommand{{\\OpticalFrequencyTHz}}{{{optical['frequency_0p5_thz']:.3f}}}
\\newcommand{{\\OpticalNoise}}{{{texScientific(optical['noise_0p5_j_per_m_hz'], 3)}}}
\\newcommand{{\\OpticalSingle}}{{{texScientific(optical['single_0p5_j_per_m_hz'], 3)}}}
\\newcommand{{\\CouplingLengthUm}}{{{gap['coupling_length_um']:.3f}}}
\\newcommand{{\\GapSuppression}}{{{texScientific(gap['minimum_power_factor_1mm'], 3)}}}
\\newcommand{{\\CoherenceCrossoverGHz}}{{{coherence['coherence_crossover_ghz']:.2f}}}
\\newcommand{{\\FormFactorScaleGHz}}{{{bunchLength['form_factor_scale_ghz']:.2f}}}
\\newcommand{{\\FormFactorWavelengthMm}}{{{bunchLength['form_factor_wavelength_mm']:.3f}}}
\\newcommand{{\\ShotNoiseBunchingRms}}{{{texScientific(case.electronCount**-0.5, 3)}}}
\\newcommand{{\\OpticalPeriodFs}}{{{1e3 / optical['frequency_0p5_thz']:.4f}}}
"""
    (FIGURE_DIRECTORY / "numbers.tex").write_text(text)


def main():
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt = configureMatplotlib()
    case = BeamCase(gapMm=0.01, energyMeV=60.0, chargeNc=1.0, epsilonR=2.13)
    # The specified rms spatial length is exactly 1 mm: sigma_t = sigma_z / v.
    case = replace(case, pulsePs=1e-3 / case.velocity * 1e12)
    density = UniformDensityEstimate(1.0, 1.0, 1.0, case.electronCount)
    optical = makeOpticalSpectrum(plt, case, density)
    gap = makeGapDependence(plt, case)
    coherence = makeCoherenceCrossover(plt, case)
    bunchLength = makeBunchLengthFormFactor(plt, case)
    writeNumbers(case, density, optical, gap, coherence, bunchLength)
    print(f"Figures: {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
