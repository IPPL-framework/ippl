#!/usr/bin/env python3
"""Compare actual Zarija and IPPL Gaussian 1LPT initial conditions.

This is a statistical comparison, not a phase-matched RNG comparison.  The
original source is never modified.  Inputs, logs, executable/source hashes and
predeclared checks are retained in a fresh work directory.  Only initial IPPL
snapshots are used, although its executable currently also takes one PM step.

Dependencies: numpy and pandas (as for validate_linear.py).  The companion
public-API scalar probe supplies the tighter T/P/D/gdot comparison; this runner
uses independently integrated spectra, never fitted or probe-derived power.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from validate_linear import bbks_spectrum, growth_reference
from runtime_metadata import validate_runtime_metadata


Seeds = (1234567, 7654321, 104729, 130363, 32452843, 49979687, 67867967, 86028121)
ShellEdges = np.asarray((0.5, 2.5, 4.5, 8.5, 12.5, 16.5, 27.0))
Parameters = {"np": 32, "box_size": 168.75, "hubble": 0.675,
              "Omega_m": 0.31, "Omega_bar": 0.0487, "Sigma_8": 0.82, "n_s": 0.965}
Tolerances = {"gaussian_standard_errors": 6.0, "reference_power_floor": 5.0e-4,
              "reference_momentum_relative": 2.0e-5, "ippl_momentum_relative": 1.0e-9,
              "longitudinal_relative": 1.0e-9, "displacement_dc_relative": 1.0e-9,
              "ippl_rank_relative": 1.0e-10, "position_roundoff_eps_box": 128.0,
              "reference_redshift_scaling_relative": 5.0e-5,
              "reference_transfer_scaling_relative": 5.0e-4,
              "ippl_redshift_scaling_relative": 1.0e-9,
              "ippl_transfer_scaling_relative": 1.0e-8, "ippl_background_relative": 1.0e-8,
              "reference_printed_background_relative": 5.0e-6}
BinaryDtype = np.dtype([(name, "<f8") for name in ("x", "vx", "y", "vy", "z", "vz")]
                       + [("id", "<i4")], align=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_spectrum(waveNumber: np.ndarray, parameters: dict, path: Path) -> np.ndarray:
    """Independent corrected CMBFAST normalization, split at every table knot.

    Each interval is at most 0.25 h/Mpc wide and integrated by Gauss16, resolving
    the R=8 top-hat oscillations.  This differs from production log-Simpson and
    the legacy adaptive uniform midpoint rule.  All rows are normalized; T=1
    below the first row, intentionally fixing the documented legacy row-0 bug.
    """
    table = np.loadtxt(path)
    if table.ndim != 2 or table.shape[1] < 3 or len(table) < 2:
        raise ValueError("Transfer table requires at least two k,CDM,baryon rows")
    knots = table[:, 0]
    weighted = (parameters["Omega_bar"] * table[:, 2]
                + (parameters["Omega_m"]-parameters["Omega_bar"]) * table[:, 1])
    if (not np.isfinite(table[:, :3]).all() or not (knots > 0).all()
            or not (np.diff(knots) > 0).all() or not (weighted > 0).all()):
        raise ValueError("Transfer table requires finite increasing k and positive weighted T")
    transfer = weighted / weighted[0]
    waveNumber = np.asarray(waveNumber, dtype=float)
    if not np.isfinite(waveNumber).all() or (waveNumber < 0).any() or (waveNumber > knots[-1]).any():
        raise ValueError("Requested wave numbers are outside the transfer table")
    edges = [0.0]
    for lower, upper in zip(np.r_[0.0, knots[:-1]], knots):
        pieces = max(1, math.ceil((upper-lower)/0.25))
        edges.extend(np.linspace(lower, upper, pieces+1)[1:].tolist())
    edges = np.asarray(edges)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    widths = np.diff(edges)/2
    k = (edges[:-1]+edges[1:])[:, None]/2 + widths[:, None]*nodes
    t = np.interp(k, knots, transfer, left=1.0)
    x = 8*k
    window = np.empty_like(x)
    small = x < 0.01
    window[small] = 1-x[small]**2/10+x[small]**4/280-x[small]**6/15120
    window[~small] = 3*(np.sin(x[~small])-x[~small]*np.cos(x[~small]))/x[~small]**3
    variance = np.sum(k**(2+parameters["n_s"])*t*t*window*window*widths[:, None]*weights)
    variance /= 2*math.pi**2
    return (parameters["Sigma_8"]**2/variance * waveNumber**parameters["n_s"]
            * np.interp(waveNumber, knots, transfer, left=1.0)**2)


@dataclass
class Snapshot:
    ids: np.ndarray
    positions: np.ndarray
    momentum: np.ndarray


@dataclass
class Geometry:
    n: int
    box: float

    def __post_init__(self):
        indices = np.fft.fftfreq(self.n)*self.n
        iz, iy, ix = np.meshgrid(indices, indices, indices, indexing="ij")
        self.integerWave = np.stack((ix, iy, iz), axis=-1)
        self.wave = self.integerWave*(2*math.pi/self.box)
        self.k2 = np.sum(self.wave**2, axis=-1)
        self.radius = np.sqrt(np.sum(self.integerWave**2, axis=-1))
        self.common = (self.k2 > 0) & np.all(np.abs(self.integerWave) < self.n/2, axis=-1)
        self.unique = self.common & ((ix > 0) | ((ix == 0) & (iy > 0))
                                     | ((ix == 0) & (iy == 0) & (iz > 0)))
        self.originPhase = np.exp(-0.5j*self.box/self.n*np.sum(self.wave, axis=-1))


def lattice(ids: np.ndarray, geometry: Geometry, code: str) -> np.ndarray:
    n = geometry.n
    if code == "zarija":
        cells = np.column_stack((ids//(n*n), (ids//n) % n, ids % n))
        offset = 0.0
    elif code == "ippl":
        cells = np.column_stack((ids % n, (ids//n) % n, ids//(n*n)))
        offset = 0.5
    else:
        raise ValueError("Unknown generator")
    return (cells+offset)*(geometry.box/n)


def validate_snapshot(snapshot: Snapshot, geometry: Geometry) -> None:
    if not np.array_equal(snapshot.ids, np.arange(geometry.n**3)):
        raise ValueError("Missing, duplicate, unordered or invalid particle IDs")
    if snapshot.positions.shape != (geometry.n**3, 3) or snapshot.momentum.shape != snapshot.positions.shape:
        raise ValueError("Invalid phase-space array shape")
    if not np.isfinite(snapshot.positions).all() or not np.isfinite(snapshot.momentum).all():
        raise ValueError("Nonfinite particle phase space")
    if (snapshot.positions < 0).any() or (snapshot.positions >= geometry.box).any():
        raise ValueError("Particle position outside [0,L)")


def read_zarija(prefix: Path, ranks: int, geometry: Geometry, a: float) -> Snapshot:
    """Read unpadded PrintFormat=2 records; reject incompatible precision/ID ABI."""
    if BinaryDtype.itemsize != 52 or geometry.n**3 % ranks:
        raise ValueError("Unsupported reference binary ABI or unequal slab particle count")
    blocks = []
    localCount = geometry.n**3//ranks
    expected = {Path(str(prefix)+f".bin.{rank}") for rank in range(ranks)}
    if set(prefix.parent.glob(prefix.name+".bin.*")) != expected:
        raise ValueError("Missing or unexpected reference rank files")
    for rank in range(ranks):
        path = Path(str(prefix)+f".bin.{rank}")
        if path.stat().st_size != localCount*52:
            raise ValueError(f"{path}: incompatible ABI/file size (expected 52 bytes per particle)")
        block = np.fromfile(path, dtype=BinaryDtype)
        if not np.array_equal(block["id"], np.arange(rank*localCount, (rank+1)*localCount)):
            raise ValueError(f"{path}: unexpected IDs/order; check integer ABI and PrintFormat=2")
        blocks.append(block)
    data = np.concatenate(blocks)
    snapshot = Snapshot(data["id"].astype(np.int64),
                        np.column_stack([data[name] for name in ("x", "y", "z")]),
                        a*a/100*np.column_stack([data[name] for name in ("vx", "vy", "vz")]))
    validate_snapshot(snapshot, geometry)
    return snapshot


def read_ippl(directory: Path, ranks: int, geometry: Geometry) -> Snapshot:
    expected = {directory/f"particles_initial_rank{rank}.csv" for rank in range(ranks)}
    if set(directory.glob("particles_initial_rank*.csv")) != expected:
        raise ValueError("Missing or unexpected IPPL initial rank files")
    data = pd.concat([pd.read_csv(path, dtype={"id": np.int64}) for path in sorted(expected)])
    data = data.sort_values("id")
    snapshot = Snapshot(data["id"].to_numpy(), data[["x", "y", "z"]].to_numpy(),
                        data[["px", "py", "pz"]].to_numpy())
    validate_snapshot(snapshot, geometry)
    return snapshot


def relative_norm(error: np.ndarray, reference: np.ndarray) -> float:
    norm = float(np.linalg.norm(reference.ravel()))
    return float(np.linalg.norm(error.ravel())/norm) if norm else math.inf


def recover(snapshot: Snapshot, geometry: Geometry, code: str, a: float,
            omegaMatter: float) -> tuple[np.ndarray, dict]:
    """Recover physical-origin delta_0 Fourier coefficients from Lagrangian displacements."""
    validate_snapshot(snapshot, geometry)
    displacement = snapshot.positions-lattice(snapshot.ids, geometry, code)
    displacement -= geometry.box*np.rint(displacement/geometry.box)
    growth, rate = growth_reference(a, omegaMatter)
    expansion = math.sqrt(omegaMatter/a**3+1-omegaMatter)
    factor = a*a*expansion*rate
    predicted = factor*displacement
    momentumError = relative_norm(snapshot.momentum-predicted, predicted)
    # A velocity-derived displacement also catches ambiguous minimum-image unwraps.
    maximumMove = float(np.max(np.abs(snapshot.momentum/factor)))
    psi = (displacement/growth).reshape((geometry.n,)*3+(3,))
    if code == "zarija":
        psi = psi.transpose(2, 1, 0, 3)  # x,y,z source storage -> z,y,x analysis storage
    psiHat = np.fft.fftn(psi, axes=(0, 1, 2), norm="forward")
    dot = np.sum(geometry.wave*psiHat, axis=-1)
    projected = np.zeros_like(psiHat)
    nonzero = geometry.k2 > 0
    projected[nonzero] = geometry.wave[nonzero]*(dot[nonzero]/geometry.k2[nonzero])[:, None]
    transverseError = relative_norm((psiHat-projected)[geometry.common], psiHat[geometry.common])
    dcError = float(np.linalg.norm(psiHat[0, 0, 0])
                    / math.sqrt(np.mean(np.sum(psi*psi, axis=-1))))
    delta = -1j*dot
    if code == "ippl":
        delta *= geometry.originPhase
    return delta, {"momentum_relative_error": momentumError,
                   "longitudinal_relative_error": transverseError,
                   "displacement_dc_relative_error": dcError,
                   "maximum_velocity_inferred_displacement": maximumMove,
                   "displacement_rms": float(np.sqrt(np.mean(displacement**2))),
                   "momentum_rms": float(np.sqrt(np.mean(snapshot.momentum**2))),
                   "theory_D": growth, "theory_f": rate, "theory_gdot": expansion*rate*growth}


def gaussian_metrics(coefficients: np.ndarray, deterministicFloor: float = 0.0) -> list[dict]:
    """Independent-pair z=sqrt(V/P)*delta: E|z|²=1; E|z|⁴=2."""
    z = np.asarray(coefficients).ravel()
    count = len(z)
    if count == 0 or not np.isfinite(z).all():
        return [{"statistic": "finite nonempty sample", "passed": False, "pairs": count}]
    power = np.abs(z)**2
    specifications = (
        ("mean_power", float(power.mean()), 1.0, 6/math.sqrt(count)+deterministicFloor),
        ("second_power_moment", float(np.mean(power**2)), 2.0,
         6*math.sqrt(20/count)+4*deterministicFloor),
        ("mean_real", float(z.real.mean()), 0.0, 6*math.sqrt(0.5/count)),
        ("mean_imaginary", float(z.imag.mean()), 0.0, 6*math.sqrt(0.5/count)),
    )
    return [{"statistic": name, "passed": abs(value-target) <= tolerance,
             "value": value, "target": target, "tolerance": tolerance, "pairs": count}
            for name, value, target, tolerance in specifications]


class Validation:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.directory = (args.work_dir.resolve() if args.work_dir else
                          Path(tempfile.mkdtemp(prefix="zarija-validation-", dir=Path.cwd())))
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError("Work directory must be empty; previous evidence is never overwritten")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.geometry = Geometry(Parameters["np"], Parameters["box_size"])
        self.seeds = Seeds[:2] if args.quick else Seeds
        self.results = {"started_utc": datetime.now(timezone.utc).isoformat(),
                        "command": sys.argv, "profile": "quick-2-seed" if args.quick else "full-8-seed",
                        "parameters": Parameters, "seeds": self.seeds,
                        "redshifts": [49.0, 200.0], "transfer_flags": [4, 0],
                        "shell_edges_integer_k": ShellEdges.tolist(), "tolerances": Tolerances,
                        "tolerance_rationale": {
                            "ippl_transfer_scaling": "Pre-campaign independent comparison: production log-Simpson65536 versus per-knot Gauss16 gives TF0 P difference -6.33736e-9, amplitude -3.16868e-9. Transfer-scaling tolerance is 1e-8; same-TF redshift tolerance remains 1e-9. No normalization is fitted to outputs.",
                            "reference_power_floor": "5e-4 covers source midpoint quadrature tolerance1e-4 and observed scalar-probe normalization discrepancy3.3e-5; statistical standard errors are counted separately."},
                        "scope": "Matched Gaussian flat radiation-free LCDM 1LPT IC only; not nonlinear evolution",
                        "conventions": {
                            "reference_output": "PrintFormat=2; packed little-endian 6 float64 + int32; x,vx,y,vy,z,vz,id",
                            "reference_velocity": "raw v=100 dx/d(H0t); canonical p=a^2*v/100",
                            "lattice": "Zarija node-centered z-fastest; IPPL half-cell-centered x-fastest",
                            "Fourier": "forward 1/N^3; remove IPPL half-cell phase; delta0=-i k.dot(psi0)",
                            "mask": "DC and all Nyquist planes excluded; one member per conjugate pair",
                            "randomness": "Distinct RNGs; no pointwise inter-code equality expected; same seed across z/TF is not independent"},
                        "reference_caveats": [
                            "Legacy table first row is not normalized; corrected independent spectrum is authoritative at low k.",
                            "Legacy midpoint sigma8 integral skips the malformed tiny-k interval; expected P error about 3.3e-5 for supplied table.",
                            "Legacy growth starts at a=1/100001 with zero derivative; finite-start decaying mode plus ODE tolerance remains.",
                            "Legacy Nyquist gradients discard imaginary components; comparing common interior only.",
                            "Legacy PrintFormat=1 has a long/int MPI-ID mismatch; not used.",
                            "Original seed parser is signed int32; fixed seeds are in range."],
                        "provenance": {}, "runs": [], "checks": []}
        self.capture_provenance()
        self.power = {}
        for flag in (4, 0):
            magnitudes = np.sqrt(self.geometry.k2[self.geometry.unique])
            self.power[flag] = (bbks_spectrum(magnitudes, Parameters) if flag == 4 else
                                table_spectrum(magnitudes, Parameters, args.transfer_file))
        self.save()
        print(f"Validation results: {self.directory/'results.json'}", flush=True)

    def capture_provenance(self):
        target = self.directory/"provenance"
        target.mkdir()
        files = sorted(path for path in self.args.zarija_source.rglob("*") if path.is_file()
                       and ".git" not in path.parts and
                       (path.suffix in (".cpp", ".h", ".c", ".hpp") or path.name in ("Makefile", "README", "input.par")))
        provenance = self.results["provenance"]
        provenance["reference_source"] = str(self.args.zarija_source)
        provenance["reference_source_sha256"] = {str(path.relative_to(self.args.zarija_source)): sha256(path) for path in files}
        provenance["ippl_cosmology_source_sha256"] = {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("Cosmology*"))
            if path.is_file() and path.suffix in (".h", ".cpp")}
        for name, path in (("ippl_executable", self.args.ippl_exe), ("zarija_executable", self.args.zarija_exe),
                           ("transfer_table", self.args.transfer_file), ("analysis_script", Path(__file__)),
                           ("independent_reference_script", Path(__file__).with_name("validate_linear.py")),
                           ("execution_metadata_header", Path(__file__).with_name("ExecutionMetadata.h")),
                           ("runtime_metadata_script", Path(__file__).with_name("runtime_metadata.py"))):
            provenance[name] = {"path": str(path), "sha256": sha256(path)}
        shutil.copy2(self.args.transfer_file, target/"transfer.tf")
        # All runs read the captured transfer artifact, not a mutable source path.
        self.args.transfer_file = target/"transfer.tf"
        provenance["captured_transfer_table"] = {"path": str(self.args.transfer_file),
                                                 "sha256": sha256(self.args.transfer_file)}
        manifest = self.args.reference_build_manifest or self.args.zarija_exe.parent/"build-manifest.txt"
        if manifest.exists():
            text = manifest.read_text()
            provenance["reference_build_manifest"] = {"path": str(manifest), "sha256": sha256(manifest), "text": text}
            shutil.copy2(manifest, target/"reference-build-manifest.txt")
            # Source-build helper records these measured ABI fields. Reject contradictions.
            for key, value in (("sizeof_real", "8"), ("sizeof_integer", "4"),
                               ("endian", "little"), ("binary_record_bytes", "52")):
                found = re.search(r"\b"+key+r"\s*=\s*(\S+)", text)
                if found and found.group(1) != value:
                    raise ValueError(f"Reference build manifest ABI mismatch: {key}={found.group(1)}")
            for name in ("source.sha256", "copied-source.sha256"):
                if (manifest.parent/name).exists():
                    shutil.copy2(manifest.parent/name, target/name)
        else:
            provenance["reference_build_manifest"] = "not supplied; ABI enforced by exact sizes/IDs/finite values"

    def save(self):
        def encode(value):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            return value
        (self.directory/"results.json").write_text(json.dumps(encode(self.results), indent=2, allow_nan=False)+"\n")

    def check(self, name: str, passed: bool, **details):
        self.results["checks"].append({"name": name, "passed": bool(passed), **details})
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
        self.save()

    def run(self, code: str, flag: int, redshift: float, seed: int, ranks: int = 1):
        name = f"{code}_tf{flag}_z{redshift:g}_s{seed}_r{ranks}"
        directory = self.directory/name
        directory.mkdir()
        parameters = dict(Parameters, TFFlag=flag, z_in=redshift, seed=seed)
        if code == "ippl":
            parameters.update(nt=1, z_fi=redshift-1, ic_mode="gaussian", write_particles=1,
                              diagnostics_every=1, transfer_file=str(self.args.transfer_file),
                              output=str(directory/"output"))
            executable = self.args.ippl_exe
        else:
            parameters.update(Omega_nu=0.0, Omega_r=0.0, w_de=-1.0, N_nu=3,
                              nu_pairs=4, f_NL=0.0, PrintFormat=2)
            executable = self.args.zarija_exe
        inputPath = directory/"input.par"
        inputPath.write_text("".join(f"{key}={value}\n" for key, value in parameters.items()))
        command = (shlex.split(self.args.mpiexec)+self.args.mpi_arg
                   +[self.args.numproc_flag, str(ranks), str(executable), str(inputPath)])
        if code == "zarija":
            command += [str(self.args.transfer_file), str(directory/"particles")]
        environment = os.environ.copy()
        overrides = {"OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false",
                     "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        environment.update(overrides)
        record = {"name": name, "code": code, "ranks": ranks, "parameters": parameters,
                  "command": command, "environment_overrides": overrides,
                  "input_sha256": sha256(inputPath), "directory": str(directory)}
        self.results["runs"].append(record)
        self.save()
        print(f"RUN {name}", flush=True)
        started = time.monotonic()
        try:
            with (directory/"run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=directory, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returnCode = process.wait(timeout=self.args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError(f"Run exceeded {self.args.timeout:g} seconds")
            record["return_code"] = returnCode
            record["elapsed_seconds"] = time.monotonic()-started
            if returnCode:
                raise RuntimeError(f"Executable exited {returnCode}; see run.log")
            a = 1/(1+redshift)
            snapshot = (read_ippl(directory/"output", ranks, self.geometry) if code == "ippl" else
                        read_zarija(directory/"particles", ranks, self.geometry, a))
            delta, metrics = recover(snapshot, self.geometry, code, a, Parameters["Omega_m"])
            record["metrics"] = metrics
            self.check(name+": execution, ABI and particle integrity", True, particles=len(snapshot.ids))
            self.check(name+": canonical momentum", metrics["momentum_relative_error"] <=
                       Tolerances[f"{'reference' if code == 'zarija' else 'ippl'}_momentum_relative"],
                       relative_error=metrics["momentum_relative_error"])
            self.check(name+": longitudinal displacement", metrics["longitudinal_relative_error"] <=
                       Tolerances["longitudinal_relative"], relative_error=metrics["longitudinal_relative_error"])
            self.check(name+": zero displacement DC", metrics["displacement_dc_relative_error"] <=
                       Tolerances["displacement_dc_relative"], relative_error=metrics["displacement_dc_relative_error"])
            self.check(name+": unambiguous periodic displacement", metrics["maximum_velocity_inferred_displacement"] <
                       self.geometry.box/4, maximum=metrics["maximum_velocity_inferred_displacement"], limit=self.geometry.box/4)
            self.background_check(name, directory, code, metrics, ranks)
            normalized = delta[self.geometry.unique]*np.sqrt(self.geometry.box**3/self.power[flag])
            return {"snapshot": snapshot, "normalized": normalized, "metrics": metrics, "name": name}
        except (OSError, ValueError, RuntimeError, KeyError, pd.errors.ParserError) as error:
            record["elapsed_seconds"] = time.monotonic()-started
            record["error"] = str(error)
            self.check(name+": execution/data", False, error=str(error))
            return None

    def background_check(self, name, directory, code, metrics, ranks):
        if code == "ippl":
            meta = dict(line.split("=", 1) for line in (directory/"output"/"metadata.txt").read_text().splitlines() if "=" in line)
            execution = validate_runtime_metadata(meta, ranks)
            self.check(name+": actual MPI/backend/host-thread configuration", True, execution=execution)
            diagnostics = pd.read_csv(directory/"output"/"diagnostics.csv")
            first = diagnostics.iloc[0]
            error = max(abs(first["D"]/metrics["theory_D"]-1), abs(first["f"]/metrics["theory_f"]-1))
            self.check(name+": independent background", first["step"] == 0 and error <=
                       Tolerances["ippl_background_relative"], maximum_relative_error=float(error))
        else:
            text = (directory/"run.log").read_text()
            match = re.search(r"growth factor=\s*([\d.eE+\-]+).*?derivative=\s*([\d.eE+\-]+)", text)
            if not match:
                raise ValueError("Missing reference printed growth factor/derivative")
            values = [float(match[1]), float(match[2])]
            error = max(abs(values[0]/metrics["theory_D"]-1), abs(values[1]/metrics["theory_gdot"]-1))
            self.check(name+": printed background versus independent theory", error <=
                       Tolerances["reference_printed_background_relative"], maximum_relative_error=error,
                       printed_D=values[0], printed_gdot=values[1], stdout_significant_digits=6)

    def statistics(self, name, arrays, code):
        values = np.asarray(arrays)
        floor = Tolerances["reference_power_floor"] if code == "zarija" else 0.0
        for metric in gaussian_metrics(values, floor):
            passed = metric.pop("passed")
            label = metric.pop("statistic")
            self.check(name+": "+label, passed, **metric)
        shells = self.geometry.radius[self.geometry.unique]
        means = []
        for lower, upper in zip(ShellEdges[:-1], ShellEdges[1:]):
            selected = values[:, (shells >= lower) & (shells < upper)].ravel()
            count = len(selected)
            if not count:
                raise ValueError("Empty predeclared power shell")
            mean = float(np.mean(np.abs(selected)**2))
            limit = 6/math.sqrt(count)+floor
            self.check(f"{name}: power shell [{lower:g},{upper:g})", abs(mean-1) <= limit,
                       mean_power=mean, tolerance=limit, independent_pairs=count)
            means.append((mean, count))
        for left, right in itertools.combinations(range(len(values)), 2):
            correlation = complex(np.mean(values[left]*np.conj(values[right])))
            limit = 6/math.sqrt(2*values.shape[1])
            self.check(f"{name}: independent seeds {self.seeds[left]}/{self.seeds[right]}",
                       max(abs(correlation.real), abs(correlation.imag)) <= limit,
                       real=correlation.real, imaginary=correlation.imag, tolerance=limit)
        return means

    def rank_check(self, reference, other):
        left, right = reference["snapshot"], other["snapshot"]
        dr = right.positions-left.positions
        dr -= self.geometry.box*np.rint(dr/self.geometry.box)
        dp = right.momentum-left.momentum
        positionError, momentumError = float(np.sqrt(np.mean(dr*dr))), float(np.sqrt(np.mean(dp*dp)))
        positionLimit = max(reference["metrics"]["displacement_rms"]*Tolerances["ippl_rank_relative"],
                            Tolerances["position_roundoff_eps_box"]*np.finfo(float).eps*self.geometry.box)
        momentumLimit = reference["metrics"]["momentum_rms"]*Tolerances["ippl_rank_relative"]
        self.check(other["name"]+": identical initial phase space across IPPL ranks",
                   positionError <= positionLimit and momentumError <= momentumLimit,
                   position_rms=positionError, momentum_rms=momentumError,
                   position_limit=positionLimit, momentum_limit=momentumLimit)

    def run_suite(self):
        redshiftBaseline = {}
        transferBaseline = {}
        for flag, redshift in itertools.product((4, 0), (49.0, 200.0)):
            ensemble = {}
            for code in ("zarija", "ippl"):
                arrays = []
                first = None
                for seed in self.seeds:
                    result = self.run(code, flag, redshift, seed)
                    if result is None:
                        continue
                    arrays.append(result["normalized"])
                    if seed == self.seeds[0]:
                        first = result
                    redshiftKey = (code, flag, seed)
                    if redshift == 49:
                        redshiftBaseline[redshiftKey] = result["normalized"]
                    elif redshiftKey in redshiftBaseline:
                        base = redshiftBaseline[redshiftKey]
                        error = relative_norm(result["normalized"]-base, base)
                        tolerance = Tolerances[f"{'ippl' if code == 'ippl' else 'reference'}_redshift_scaling_relative"]
                        self.check(result["name"]+": same-seed same-TF redshift scaling", error <= tolerance,
                                   relative_error=error, tolerance=tolerance, baseline=f"same code/seed, TF{flag},z49")
                    transferKey = (code, redshift, seed)
                    if flag == 4:
                        transferBaseline[transferKey] = result["normalized"]
                    elif transferKey in transferBaseline:
                        base = transferBaseline[transferKey]
                        error = relative_norm(result["normalized"]-base, base)
                        tolerance = Tolerances[f"{'ippl' if code == 'ippl' else 'reference'}_transfer_scaling_relative"]
                        self.check(result["name"]+": same-seed same-redshift transfer scaling", error <= tolerance,
                                   relative_error=error, tolerance=tolerance, baseline=f"same code/seed, TF4,z{redshift:g}")
                self.check(f"{code}_tf{flag}_z{redshift:g}: complete independent-seed ensemble",
                           len(arrays) == len(self.seeds), completed=len(arrays), required=len(self.seeds))
                if len(arrays) == len(self.seeds):
                    ensemble[code] = self.statistics(f"{code}_tf{flag}_z{redshift:g}", arrays, code)
                # Reference rank decomposition changes its RNG; test statistics, not phases.
                for ranks in ((2, 4) if code == "zarija" else (2, 3, 4)):
                    result = self.run(code, flag, redshift, self.seeds[0], ranks)
                    if result is not None:
                        if code == "ippl" and first is not None:
                            self.rank_check(first, result)
                        elif code == "zarija":
                            self.statistics(result["name"]+"_smoke", [result["normalized"]], code)
            if len(ensemble) == 2:
                for index, ((left, countLeft), (right, countRight)) in enumerate(zip(ensemble["ippl"], ensemble["zarija"])):
                    tolerance = 6*math.sqrt(1/countLeft+1/countRight)+Tolerances["reference_power_floor"]
                    self.check(f"paired TF{flag} z{redshift:g} shell{index}: equal ensemble power",
                               abs(left-right) <= tolerance, ippl=left, zarija=right, tolerance=tolerance)
        self.verify_provenance_unchanged()
        self.results["completed_utc"] = datetime.now(timezone.utc).isoformat()
        passed = all(check["passed"] for check in self.results["checks"])
        self.results["passed"] = passed
        self.results["summary"] = {"runs": len(self.results["runs"]), "checks": len(self.results["checks"]),
                                   "failures": sum(not check["passed"] for check in self.results["checks"])}
        self.save()
        return passed

    def verify_provenance_unchanged(self):
        provenance = self.results["provenance"]
        expected = provenance["reference_source_sha256"]
        actual = {}
        for name in expected:
            path = self.args.zarija_source/name
            actual[name] = sha256(path) if path.is_file() else None
        unchanged = actual == expected
        self.results["reference_source_unchanged"] = unchanged
        self.results["reference_source_sha256_after"] = actual
        self.check("reference source unchanged throughout campaign", unchanged,
                   changed_files=[name for name in expected if expected[name] != actual[name]])
        self.results["artifact_sha256_after"] = {}
        for name in ("ippl_executable", "zarija_executable", "transfer_table",
                     "captured_transfer_table", "analysis_script", "independent_reference_script",
                     "execution_metadata_header", "runtime_metadata_script"):
            artifact = provenance[name]
            path = Path(artifact["path"])
            current = sha256(path) if path.is_file() else None
            self.results["artifact_sha256_after"][name] = current
            self.check(name+" unchanged throughout campaign", current == artifact["sha256"],
                       expected_sha256=artifact["sha256"], actual_sha256=current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ippl-exe", type=Path, required=True)
    parser.add_argument("--zarija-exe", type=Path, required=True)
    parser.add_argument("--zarija-source", type=Path, required=True)
    parser.add_argument("--transfer-file", type=Path)
    parser.add_argument("--reference-build-manifest", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-run wall timeout in seconds")
    parser.add_argument("--quick", action="store_true", help="Two seeds instead of eight; all MPI/TF/z cases retained")
    args = parser.parse_args()
    for name in ("ippl_exe", "zarija_exe", "zarija_source"):
        setattr(args, name, getattr(args, name).resolve())
    args.transfer_file = (args.transfer_file or args.zarija_source/"cmb.tf").resolve()
    if args.reference_build_manifest:
        args.reference_build_manifest = args.reference_build_manifest.resolve()
    if not args.zarija_source.is_dir() or args.timeout <= 0:
        parser.error("Require existing source directory and positive per-run timeout")
    validation = None
    try:
        validation = Validation(args)
        return 0 if validation.run_suite() else 1
    except (OSError, ValueError, RuntimeError) as error:
        if validation is not None:
            validation.check("validation harness fatal error", False, error=str(error))
            validation.results["passed"] = False
            validation.save()
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
