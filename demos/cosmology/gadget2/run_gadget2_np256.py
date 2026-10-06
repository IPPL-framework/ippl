#!/usr/bin/env python3
"""Run an audited GADGET-2 TreePM 256^3-particle/mesh comparison on Merlin."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import time


SharedRoot = Path("/data/user/adelmann/cosmology-zeldovich-np256-20261006")
GadgetRoot = Path("/data/user/adelmann/gadget2-zeldovich-np256-20261006")
ICCsv = SharedRoot / "ics/shared-z99.csv"
ICManifest = SharedRoot / "ics/ic-manifest.json"
ReferencePlotManifest = SharedRoot / "plots/z99-np256-ippl-fastpm-v1/manifest.json"
ReferencePlotValues = SharedRoot / "plots/z99-np256-ippl-fastpm-v1/plotted_values.json"
SpectrumSource = SharedRoot / "source/demos/cosmology/analyze_zeldovich_benchmark.py"
IPPLRun = SharedRoot / "runs/z99-ippl-a100-p256-m256/run.json"
FastPMRun = SharedRoot / "runs/z99-fastpm-cpu-p256-m256/run.json"
ExpectedICSha256 = "3b4a3e1864ad535369444b98779c117750e1e980cbe7d01afedc20f8329cda11"
Executable = GadgetRoot / "prefix/Gadget2-TreePM-256-double"
ExecutableSha256 = "3b1668629958329d3e5a9913717548e3d690dea3201f45816bddfafb6d90a0ac"
GadgetSourceArchive = Path("/data/user/adelmann/gadget2-zeldovich-20261004/downloads/gadget-2.0.7.tar.gz")
FFTWSourceArchive = Path("/data/user/adelmann/gadget2-zeldovich-20261004/downloads/fftw-2.1.5.tar.gz")
BuildMakefile = GadgetRoot / "build/gadget-work-256/Makefile"
BuildLog = GadgetRoot / "build/gadget2-pmgrid256-build.log"
ConverterSource = GadgetRoot / "source/convert_shared_ic.py"
PMGridPatch = GadgetRoot / "source/pmgrid256.patch"
BatchSource = GadgetRoot / "source/run_gadget2_np256.sbatch"
LoginSource = GadgetRoot / "source/run_gadget2_np256.login.sh"
ConvertedIC = GadgetRoot / "ics/shared-z99-gadget2.bin"
ConvertedICSidecar = ConvertedIC.with_suffix(ConvertedIC.suffix + ".json")
ParticleGrid = 256
PMGrid = 256
Ranks = 8
BoxMpcH = 168.75
OmegaMatter = .31
OmegaLambda = .69
HubbleParam = .675
InitialScaleFactor = .01
FinalScaleFactor = 1.0
SofteningKpcH = BoxMpcH * 1000.0 / ParticleGrid * .02
TimeoutSeconds = 6 * 24 * 60 * 60
GadgetCpuLimitSeconds = 7 * 24 * 60 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def session_members(session_id: int) -> list[dict]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tail = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if tail[0] in ("Z", "X") or int(tail[3]) != session_id:
                continue
            if entry.stat().st_uid != os.getuid():
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace").strip()
            result.append({"pid": int(entry.name), "start_ticks": int(tail[19]),
                           "command": command})
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    return result


def terminate_verified_session(session_id: int, grace_seconds: float = 8.0) -> None:
    def signal_members(sig: int) -> None:
        for item in session_members(session_id):
            process = Path("/proc") / str(item["pid"])
            try:
                tail = (process / "stat").read_text().rsplit(")", 1)[1].split()
                identity = (int(tail[3]), int(tail[19]), process.stat().st_uid)
                if identity == (session_id, item["start_ticks"], os.getuid()):
                    os.kill(item["pid"], sig)
            except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
                continue

    signal_members(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and session_members(session_id):
        time.sleep(.1)
    signal_members(signal.SIGKILL)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and session_members(session_id):
        time.sleep(.1)
    survivors = session_members(session_id)
    if survivors:
        raise RuntimeError(f"task-owned MPI session {session_id} survived cleanup: {survivors}")


def validate_reference() -> dict:
    if sha256(ICCsv) != ExpectedICSha256:
        raise ValueError("shared 256^3 CSV SHA256 differs from the approved input")
    ic = json.loads(ICManifest.read_text())
    fixture = next((row for row in ic["fixtures"] if row["redshift_initial"] == 99.0), None)
    if (fixture is None or ic.get("particle_grid") != ParticleGrid
            or ic.get("particle_count") != ParticleGrid**3
            or ic.get("cutoff_fundamental") != 96
            or fixture.get("csv_sha256") != ExpectedICSha256):
        raise ValueError("shared IC manifest does not describe the expected 256^3 z=99 realization")
    if sha256(ICManifest) != json.loads(ReferencePlotManifest.read_text())["input_hashes"][str(ICManifest)]:
        raise ValueError("IC manifest changed since the IPPL/FastPM spectrum audit")

    runs = {}
    for label, path, success_field in (("ippl", IPPLRun, "solver_exit_status"),
                                       ("fastpm", FastPMRun, "slurm_exit_status")):
        record = json.loads(path.read_text())
        if (record.get("state") != "complete" or record.get(success_field) != 0
                or record.get("particle_grid") != ParticleGrid
                or record.get("force_mesh_grid") != PMGrid
                or record.get("particles") != ParticleGrid**3
                or record.get("steps") != 2400 or record.get("checkpoints") != 24
                or record.get("input_csv_sha256") != ExpectedICSha256):
            raise ValueError(f"the audited {label} 256^3 run is not a valid matched reference")
        runs[label] = record

    plot_manifest = json.loads(ReferencePlotManifest.read_text())
    if (plot_manifest.get("campaign_complete") is not True
            or plot_manifest.get("initial_conditions", {}).get("csv_sha256") != ExpectedICSha256):
        raise ValueError("the existing IPPL/FastPM spectrum report is incomplete or mismatched")
    plot_values = json.loads(ReferencePlotValues.read_text())
    if (plot_values.get("particle_grid") != ParticleGrid
            or plot_values.get("force_mesh_grid") != PMGrid
            or plot_values.get("shared_input_sha256") != ExpectedICSha256
            or len(plot_values.get("shells", [])) != 96):
        raise ValueError("the existing IPPL/FastPM plotted values do not cover the expected 96 shells")
    for name, expected in sorted(plot_manifest["input_hashes"].items()):
        print(f"Verifying existing plot input: {name}", flush=True)
        if sha256(Path(name)) != expected:
            raise ValueError(f"a reference input changed after the existing spectrum plot: {name}")
    fastpm_snapshots = []
    snapshot_hashes = runs["fastpm"].get("snapshot_sha256", {})
    for name, expected in sorted(snapshot_hashes.items()):
        path = Path(name)
        if path.name.startswith("particles_checkpoint0024_rank"):
            print(f"Verifying final FastPM rank shard: {path}", flush=True)
            if sha256(path) != expected:
                raise ValueError(f"FastPM final checkpoint hash mismatch: {path}")
            fastpm_snapshots.append({"path": str(path), "sha256": expected})
    if len(fastpm_snapshots) != Ranks:
        raise ValueError("the final FastPM checkpoint does not have eight rank shards")

    return {"ic_manifest_sha256": sha256(ICManifest),
            "reference_plot_manifest_sha256": sha256(ReferencePlotManifest),
            "reference_plot_values_sha256": sha256(ReferencePlotValues),
            "spectrum_source_sha256": sha256(SpectrumSource),
            "ippl_run_json_sha256": sha256(IPPLRun),
            "fastpm_run_json_sha256": sha256(FastPMRun),
            "fastpm_final_shards": fastpm_snapshots,
            "particle_count": ParticleGrid**3, "cutoff": 96, "runs": runs}


def parameter_text(run_dir: Path, output_dir: Path) -> tuple[str, str]:
    output_list = run_dir / "outputs-z99.txt"
    output_list.write_text("1.0\n")
    parameters = f"""InitCondFile {ConvertedIC}
OutputDir {output_dir}/
EnergyFile energy.txt
InfoFile info.txt
TimingsFile timings.txt
CpuFile cpu.txt
RestartFile restart
SnapshotFileBase snapshot
OutputListFilename {output_list}
TimeLimitCPU {GadgetCpuLimitSeconds}
ResubmitOn 0
ResubmitCommand none
ICFormat 1
SnapFormat 1
ComovingIntegrationOn 1
TypeOfTimestepCriterion 0
OutputListOn 0
PeriodicBoundariesOn 1
TimeBegin {InitialScaleFactor:.17g}
TimeMax {FinalScaleFactor:.17g}
Omega0 {OmegaMatter:.17g}
OmegaLambda {OmegaLambda:.17g}
OmegaBaryon 0.0487
HubbleParam {HubbleParam:.17g}
BoxSize {BoxMpcH * 1000.0:.17g}
TimeBetSnapshot 2.0
TimeOfFirstSnapshot 2.0
CpuTimeBetRestartFile 36000.0
TimeBetStatistics 0.05
CourantFac 0.15
NumFilesPerSnapshot 1
NumFilesWrittenInParallel 1
ErrTolIntAccuracy 0.025
MaxRMSDisplacementFac 0.2
MaxSizeTimestep 0.025
MinSizeTimestep 0.0
ErrTolTheta 0.5
TypeOfOpeningCriterion 1
ErrTolForceAcc 0.005
TreeDomainUpdateFrequency 0.1
DesNumNgb 33
MaxNumNgbDeviation 2
ArtBulkViscConst 0.8
InitGasTemp 0
MinGasTemp 0
PartAllocFactor 1.6
TreeAllocFactor 0.8
BufferSize 64
UnitLength_in_cm 3.085678e21
UnitMass_in_g 1.989e43
UnitVelocity_in_cm_per_s 1e5
GravityConstantInternal 0
MinGasHsmlFractional 0.25
SofteningGas 0
SofteningHalo {SofteningKpcH:.17g}
SofteningDisk 0
SofteningBulge 0
SofteningStars 0
SofteningBndry 0
SofteningGasMaxPhys 0
SofteningHaloMaxPhys {SofteningKpcH:.17g}
SofteningDiskMaxPhys 0
SofteningBulgeMaxPhys 0
SofteningStarsMaxPhys 0
SofteningBndryMaxPhys 0
"""
    parameter_file = run_dir / "gadget_z99_np256.param"
    parameter_file.write_text(parameters)
    return str(parameter_file), str(output_list)


def validate_snapshot(path: Path) -> dict:
    with path.open("rb") as stream:
        marker = stream.read(4)
        if len(marker) != 4 or struct.unpack("<i", marker)[0] != 256:
            raise ValueError("GADGET output is not a format-1 256-byte header")
        header = stream.read(256)
        footer = stream.read(4)
        if len(header) != 256 or struct.unpack("<i", footer)[0] != 256:
            raise ValueError("GADGET format-1 header record is truncated")
    counts = struct.unpack_from("<6i", header, 0)
    total_counts = struct.unpack_from("<6I", header, 96)
    a, redshift = struct.unpack_from("<2d", header, 72)
    box, omega_m, omega_lambda, hubble = struct.unpack_from("<4d", header, 128)
    if (counts != (0, ParticleGrid**3, 0, 0, 0, 0)
            or total_counts != counts or not math.isclose(a, 1.0, abs_tol=1e-12)
            or not math.isclose(redshift, 0.0, abs_tol=1e-12)
            or not math.isclose(box, BoxMpcH * 1000.0, abs_tol=1e-8)
            or not math.isclose(omega_m, OmegaMatter, abs_tol=1e-12)
            or not math.isclose(omega_lambda, OmegaLambda, abs_tol=1e-12)
            or not math.isclose(hubble, HubbleParam, abs_tol=1e-12)):
        raise ValueError("GADGET snapshot header disagrees with the requested z=0 256^3 cosmology")
    return {"counts_by_type": counts, "total_counts_by_type": total_counts,
            "a": a, "redshift": redshift, "box_size_kpc_h": box,
            "omega_m": omega_m, "omega_lambda": omega_lambda, "hubble_param": hubble}


def run(login_mode: bool = False) -> dict:
    job_id = os.environ.get("SLURM_JOB_ID")
    host = socket.gethostname()
    if login_mode:
        if not host.startswith("merlin-l-") or job_id:
            raise RuntimeError("--login is allowed only on merlin-l-* outside a Slurm allocation")
        execution_mode = "authorized_login"
    else:
        if (not job_id or os.environ.get("SLURM_JOB_PARTITION") != "gwendolen"
                or int(os.environ.get("SLURM_NTASKS", "0")) != Ranks
                or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1):
            raise RuntimeError("run only inside the authorized eight-rank, one-thread Gwendolen Slurm allocation")
        execution_mode = "slurm_gwendolen"
    if ExecutableSha256 == "TO_BE_SET_AFTER_BUILD" or sha256(Executable) != ExecutableSha256:
        raise RuntimeError("GADGET-2 PMGRID=256 executable checksum is not frozen or does not match")
    if sha256(GadgetSourceArchive) != "8e321110b9fb2d05819f9cfcbffda19f56bb77c7cdd1ca21139b81b1fca4e3ed":
        raise RuntimeError("official GADGET-2 source archive checksum mismatch")
    if sha256(FFTWSourceArchive) != "f8057fae1c7df8b99116783ef3e94a6a44518d49c72e2e630c24b689c6022630":
        raise RuntimeError("official FFTW 2.1.5 source archive checksum mismatch")
    if not BuildMakefile.is_file() or not BuildLog.is_file():
        raise RuntimeError("PMGRID=256 build provenance is missing")
    build_text = BuildMakefile.read_text()
    if "-DPMGRID=256" not in build_text or "-DPMGRID=128" in build_text:
        raise RuntimeError("GADGET build Makefile is not configured exclusively for PMGRID=256")
    references = validate_reference()
    if not ConvertedIC.is_file() or not ConvertedICSidecar.is_file():
        raise RuntimeError("the shared CSV must first be converted to audited GADGET format-1")
    conversion = json.loads(ConvertedICSidecar.read_text())
    if (conversion["input"]["sha256"] != ExpectedICSha256
            or conversion["initial_conditions"]["particle_count"] != ParticleGrid**3
            or conversion["initial_conditions"]["redshift"] != 99.0
            or sha256(ConvertedIC) != conversion["output"]["sha256"]):
        raise RuntimeError("converted GADGET IC does not preserve the approved shared z=99 file")

    campaign_path = GadgetRoot / "campaign.json"
    if campaign_path.exists():
        raise FileExistsError("isolated 256^3 Gadget campaign already exists; refusing to overwrite")
    run_dir = GadgetRoot / "runs/z99_gadget2_np256"
    output_dir = GadgetRoot / "outputs/z99_gadget2_np256"
    if run_dir.exists() or output_dir.exists():
        raise FileExistsError("Gadget run/output path exists; refusing to overwrite")
    run_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    parameter_file, output_list = parameter_text(run_dir, output_dir)
    command = [shutil.which("mpiexec") or "mpiexec", "--bind-to", "none", "--map-by", "slot",
               "-n", str(Ranks), str(Executable), parameter_file]
    parameter_hash = sha256(Path(parameter_file))
    record = {"schema": "ippl-gadget2-np256-shared-ic-run-v1", "state": "running",
        "case": "z99_gadget2_np256", "job_id": job_id,
        "host": socket.gethostname(), "ranks": Ranks, "threads_per_rank": 1,
        "particle_grid": ParticleGrid, "pm_grid": PMGrid, "particle_count": ParticleGrid**3,
        "input_csv": str(ICCsv), "input_csv_sha256": ExpectedICSha256,
        "converted_ic": str(ConvertedIC), "converted_ic_sha256": sha256(ConvertedIC),
        "converter_sidecar": str(ConvertedICSidecar), "converter_sidecar_sha256": sha256(ConvertedICSidecar),
        "executable": str(Executable), "executable_sha256": ExecutableSha256,
        "gadget2_source_archive": str(GadgetSourceArchive), "gadget2_source_archive_sha256": sha256(GadgetSourceArchive),
        "fftw2_source_archive": str(FFTWSourceArchive), "fftw2_source_archive_sha256": sha256(FFTWSourceArchive),
        "build_makefile": str(BuildMakefile), "build_makefile_sha256": sha256(BuildMakefile),
        "build_log": str(BuildLog), "build_log_sha256": sha256(BuildLog),
        "controller_source": str(Path(__file__).resolve()), "controller_source_sha256": sha256(Path(__file__)),
        "converter_source": str(ConverterSource), "converter_source_sha256": sha256(ConverterSource),
        "batch_source": str(BatchSource), "batch_source_sha256": sha256(BatchSource),
        "login_source": str(LoginSource), "login_source_sha256": sha256(LoginSource),
        "pmgrid_patch": str(PMGridPatch), "pmgrid_patch_sha256": sha256(PMGridPatch),
        "input_manifest_sha256": references["ic_manifest_sha256"],
        "spectrum_source": str(SpectrumSource),
        "spectrum_source_sha256": references["spectrum_source_sha256"],
        "reference_plot_manifest_sha256": references["reference_plot_manifest_sha256"],
        "reference_plot_values_sha256": references["reference_plot_values_sha256"],
        "ippl_run_json_sha256": references["ippl_run_json_sha256"],
        "fastpm_run_json_sha256": references["fastpm_run_json_sha256"],
        "execution_mode": execution_mode, "execution_host": host,
        "initial_redshift": 99.0, "initial_scale_factor": InitialScaleFactor,
        "final_redshift": 0.0, "final_scale_factor": FinalScaleFactor,
        "box_mpc_over_h": BoxMpcH, "omega_m": OmegaMatter,
        "omega_lambda": OmegaLambda, "hubble_param": HubbleParam,
        "steps": "GADGET-2 adaptive synchronized integration",
        "time_integration": {"ErrTolIntAccuracy": .025, "MaxRMSDisplacementFac": .2,
            "MaxSizeTimestep": .025, "ErrTolTheta": .5, "ErrTolForceAcc": .005,
            "SYNCHRONIZATION": True},
        "softening_kpc_over_h": SofteningKpcH,
        "parameter_file": parameter_file, "parameter_sha256": parameter_hash,
        "output_list": output_list, "output_list_sha256": sha256(Path(output_list)),
        "command": command, "output_directory": str(output_dir),
        "timeout_seconds": TimeoutSeconds, "started_utc": utc_now()}
    report = {"schema": "ippl-gadget2-np256-shared-ic-campaign-v1", "root": str(GadgetRoot),
              "status": "running", "complete": False, "run": record,
              "reference_audit": {k: v for k, v in references.items() if k != "runs"}}
    atomic_json(campaign_path, report)
    (GadgetRoot / "logs/exit-status.txt").write_text("phase=running\n")
    log_path = run_dir / "launch.log"
    environment = {"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    began = time.monotonic()
    process = None
    timed_out = False
    code = None
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, **environment}, start_new_session=True)
            record["launcher_pid"] = process.pid
            record["launcher_session_id"] = process.pid
            atomic_json(campaign_path, report)
            try:
                code = process.wait(timeout=TimeoutSeconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=8)
                code = process.returncode
    except BaseException:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
        if process is not None:
            terminate_verified_session(process.pid)
        raise
    time.sleep(.25)
    survivors = session_members(process.pid)
    if survivors:
        record_survivors = survivors
        terminate_verified_session(process.pid)
        if session_members(process.pid):
            raise RuntimeError(f"MPI ranks survived task-owned cleanup: {session_members(process.pid)}")
        record["terminated_task_owned_mpi_survivors"] = record_survivors
        code = code or 1
        survivors = record_survivors
    wall_seconds = time.monotonic() - began
    snapshots = sorted(output_dir.glob("snapshot*"))
    record.update({"return_code": code, "timed_out": timed_out,
                   "solver_wall_seconds": wall_seconds, "finished_utc": utc_now(),
                   "snapshots": [{"path": str(path), "bytes": path.stat().st_size,
                                  "sha256": sha256(path)} for path in snapshots]})
    if timed_out or code != 0 or not snapshots or survivors:
        record["state"] = "failed"
        report.update({"status": "execution_failed", "failure": (
            "launch timed out" if timed_out else "task-owned MPI ranks survived launcher exit" if survivors
            else f"GADGET exit {code}" if code else "no snapshot"),
            "complete": False})
        atomic_json(campaign_path, report)
        (GadgetRoot / "logs/exit-status.txt").write_text(
            f"phase=execution_failed exit={code} timed_out={int(timed_out)} wall_seconds={wall_seconds:.3f}\n")
        raise RuntimeError(report["failure"])
    try:
        header = validate_snapshot(snapshots[-1])
    except Exception as error:
        record["state"] = "failed_validation"
        report.update({"status": "snapshot_validation_failed", "failure": str(error)})
        atomic_json(campaign_path, report)
        (GadgetRoot / "logs/exit-status.txt").write_text(f"phase=snapshot_validation_failed {error}\n")
        raise
    record.update({"state": "complete", "final_snapshot_header": header})
    report.update({"status": "complete", "complete": True, "finished_utc": utc_now()})
    atomic_json(campaign_path, report)
    (GadgetRoot / "logs/exit-status.txt").write_text(
        f"phase=complete exit=0 wall_seconds={wall_seconds:.3f}\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", action="store_true",
                        help="run on the authorized Merlin CPU login host outside Slurm")
    arguments = parser.parse_args()
    print(json.dumps(run(login_mode=arguments.login), indent=2, allow_nan=False))
