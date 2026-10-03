"""Disk-aware execution and lossless *numerical* archives for cosmology studies.

The helper owns an initially empty directory, not previous campaign data. CSV
bytes are hashed but are not reconstructible from an archive; parsed uint64 IDs
and float64 phase space are retained exactly, including signed zero. Scientific
acceptance remains the caller's responsibility: storage completion is not a pass.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess

import numpy as np
import pandas as pd


Columns = ["id", "x", "y", "z", "px", "py", "pz"]
GiB = 1024**3
FixedEnvironment = {"OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false",
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


class DiskSpaceBlocked(RuntimeError):
    """Insufficient free space: resumable block, never a scientific result."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_bytes(particle_grid, checkpoints):
    """Full raw run + conservative archive bound + one-checkpoint workspace."""
    if type(particle_grid) is not int or particle_grid < 1 or type(checkpoints) is not int or checkpoints < 1:
        raise ValueError("Positive integer particle grid and checkpoint count required")
    particles = particle_grid**3
    return particles * ((180 + 64) * (checkpoints + 1) + 2 * 64) + 64 * 1024**2


def array_digest(ids, phase_space):
    digest = hashlib.sha256(b"cosmology-particles-uint64-float64-v1\0")
    digest.update(struct.pack("<Q", len(ids)))
    digest.update(np.ascontiguousarray(ids, dtype="<u8").tobytes())
    digest.update(np.ascontiguousarray(phase_space, dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_arrays(ids, phase_space, particles):
    if (ids.dtype != np.dtype("uint64") or phase_space.dtype != np.dtype("float64")
            or ids.shape != (particles,) or phase_space.shape != (particles, 6)
            or not np.array_equal(ids, np.arange(particles, dtype=np.uint64))
            or not np.isfinite(phase_space).all()):
        raise ValueError("Archive requires sorted complete IDs and finite float64 Nx6 phase space")


def read_archive(path, particles, *, expected_sha=None, expected_digest=None):
    if expected_sha is not None and sha256(path) != expected_sha:
        raise ValueError(f"Archive file hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"ids", "phase_space"}:
            raise ValueError("Unexpected archive array names")
        ids, phase = arrays["ids"], arrays["phase_space"]
    validate_arrays(ids, phase, particles)
    if expected_digest is not None and array_digest(ids, phase) != expected_digest:
        raise ValueError(f"Archive numerical digest mismatch: {path}")
    frame = pd.DataFrame(phase, columns=Columns[1:])
    frame.insert(0, "id", ids)
    return frame


class StudyStorage:
    """One writer; completed runs reuse verified archives without re-execution.

    Call ``output_dir(name)`` when constructing the adapter command, then
    ``execute(name, command, particle_grid=..., ranks=..., checkpoints=...)``.
    On resume, command, environment, input, executable and provenance must match.
    Interrupted executions are retained and rejected, not guessed complete or
    overwritten. Interrupted archival after a successful exit can be resumed.
    """
    def __init__(self, directory, provenance, *, reserve_bytes=GiB, resume=False):
        self.root = Path(directory).resolve()
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a nonnegative integer")
        self.reserve_bytes = reserve_bytes
        self.provenance = {str(Path(path).resolve()): digest for path, digest in provenance.items()}
        if not self.provenance:
            raise ValueError("Nonempty source/build provenance required")
        self._verify_hashes(self.provenance)
        self.manifest_path = self.root / "storage.json"
        if resume:
            self.manifest = json.loads(self.manifest_path.read_text())
            if (self.manifest.get("schema") != "cosmology-study-storage-v1"
                    or self.manifest.get("root") != str(self.root)
                    or self.manifest.get("provenance") != self.provenance):
                raise ValueError("Resume provenance or storage identity mismatch")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            if any(self.root.iterdir()):
                raise ValueError("New storage directory must be empty")
            self.manifest = {"schema": "cosmology-study-storage-v1", "root": str(self.root),
                             "provenance": self.provenance, "runs": {},
                             "archive_contract": "Lossless numerical values; original CSV bytes are not recoverable"}
        self._lock = (self.root / "storage.lock").open("a")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("Another process owns this study storage directory") from None
        if resume:
            # Read again under the lock; another owner might have just completed.
            current = json.loads(self.manifest_path.read_text())
            if current["provenance"] != self.provenance or current["root"] != str(self.root):
                self.close()
                raise ValueError("Storage identity changed during resume")
            self.manifest = current
        self._save()

    def close(self):
        if getattr(self, "_lock", None) is not None and not self._lock.closed:
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        self.close()

    def _save(self):
        temporary = self.root / "storage.json.partial"
        with temporary.open("w") as stream:
            json.dump(self.manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.manifest_path)

    @staticmethod
    def _verify_hashes(hashes):
        for path, expected in hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Provenance/artifact hash mismatch: {path}")

    def output_dir(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
            raise ValueError("Run name must contain only safe filename characters")
        path = self.root / name
        if path.is_symlink():
            raise ValueError("Run output must not be a symlink")
        return path

    def _preflight(self, run, incremental_bytes, phase):
        free = shutil.disk_usage(self.root).free
        run["disk_preflight"] = {"free_bytes": free, "estimated_peak_increment_bytes": incremental_bytes,
                                 "reserve_bytes": self.reserve_bytes, "phase": phase}
        if free < incremental_bytes + self.reserve_bytes:
            run.update(state="blocked_disk", resume_phase=phase)
            self._save()
            raise DiskSpaceBlocked(f"Need {incremental_bytes + self.reserve_bytes} free bytes; have {free}. "
                                   f"No {phase} started. Free disk and resume {self.root}")

    def execute(self, name, command, *, particle_grid, ranks, checkpoints, timeout=300, env=None):
        if type(ranks) is not int or ranks < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive rank count and finite timeout required")
        estimate = peak_bytes(particle_grid, checkpoints)
        command = [str(part) for part in command]
        output = self.output_dir(name)
        # The two adapters share these ten positional arguments; no shell is used.
        if (len(command) < 11 or command[-10] != str(particle_grid)
                or command[-3] != str(checkpoints) or Path(command[-1]).resolve() != output):
            raise ValueError("Command does not match the fixed evolution adapter contract")
        executable, input_path = Path(command[-11]).resolve(), Path(command[-2]).resolve()
        if not executable.is_file() or not input_path.is_file():
            raise ValueError("Executable and shared CSV input must exist")
        environment = {str(key): str(value) for key, value in (env or {}).items()}
        if any(key in environment and environment[key] != value for key, value in FixedEnvironment.items()):
            raise ValueError("Study requires fixed one-thread execution environment")
        environment.update(FixedEnvironment)
        descriptor = {"command": command, "particle_grid": particle_grid, "ranks": ranks,
                      "checkpoints": checkpoints, "environment": environment,
                      "input_sha256": sha256(input_path), "executable_sha256": sha256(executable),
                      "output": str(output)}
        self._verify_hashes(self.provenance)
        run = self.manifest["runs"].get(name)
        if run is not None and run["descriptor"] != descriptor:
            raise ValueError("Resume run descriptor, input or executable differs")
        if run is None:
            if output.exists() or (self.root / (name + ".log")).exists():
                raise ValueError("Unowned output/log already exists; nothing will be overwritten")
            run = {"descriptor": descriptor, "state": "pending", "snapshots": [], "retained_artifacts": {}}
            self.manifest["runs"][name] = run
            self._save()
        if run["state"] in ("complete", "integrity_failed"):
            try:
                self._verify_completed(run)
            except Exception as error:
                run.update(state="integrity_failed", storage_complete=False, integrity_error=str(error))
                self._save()
                raise
            run.update(state="complete", storage_complete=True)
            run.pop("integrity_error", None)
            self._save()
            return deepcopy(run)
        if run["state"] not in ("pending", "blocked_disk", "running", "execution_failed", "archiving", "archive_failed"):
            raise ValueError("Unknown persisted run state; refusing to launch")
        phase = run.get("resume_phase", "launch") if run["state"] == "blocked_disk" else "launch"
        if run["state"] in ("running", "execution_failed"):
            raise RuntimeError(f"Incomplete execution retained at {output}; not safe to infer success or rerun")
        if run["state"] in ("archiving", "archive_failed"):
            phase = "archive"
        if phase == "launch":
            self._preflight(run, estimate, "launch")
            run.update(state="running", log_path=str(self.root / (name + ".log")))
            self._save()
            try:
                self._launch(command, Path(run["log_path"]), timeout, environment)
                run.update(state="archiving", return_code=0)
                self._save()
            except BaseException as error:
                run.update(state="execution_failed", execution_error=str(error))
                self._save()
                raise
        try:
            self._archive_run(run)
            self._verify_hashes(self.provenance)
            self._verify_hashes({str(executable): descriptor["executable_sha256"],
                                 str(input_path): descriptor["input_sha256"]})
            run.update(state="complete", storage_complete=True, scientific_acceptance="not assessed")
            run.pop("archive_error", None)
            run.pop("resume_phase", None)
            self._save()
            return deepcopy(run)
        except DiskSpaceBlocked:
            raise
        except BaseException as error:
            run.update(state="archive_failed", archive_error=str(error))
            self._save()
            raise

    def _launch(self, command, log_path, timeout, environment):
        with log_path.open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, **environment), start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                # Terminate only this helper's process group, including its ranks.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)

    def _archive_run(self, run):
        descriptor = run["descriptor"]
        output = Path(descriptor["output"])
        particles, ranks = descriptor["particle_grid"]**3, descriptor["ranks"]
        # Full successful exit is required; checkpoint rows alone are not used
        # as proof that all MPI writers have closed their files.
        required = [output / "metadata.txt", output / "checkpoints.csv", Path(run["log_path"])]
        if (output / "factors.csv").exists():
            required.append(output / "factors.csv")
        current = {str(path): sha256(path) for path in required}
        if run["retained_artifacts"] and run["retained_artifacts"] != current:
            raise ValueError("Retained run metadata/log/factors changed during archival")
        run["retained_artifacts"] = current
        self._save()
        for checkpoint in range(descriptor["checkpoints"] + 1):
            archived = [entry for entry in run["snapshots"] if entry["checkpoint"] == checkpoint]
            if len(archived) > 1:
                raise ValueError("Duplicate checkpoint archive")
            if archived:
                self._verify_snapshot(archived[0], particles)
                self._remove_recorded_csvs(archived[0], output)
                continue
            self._preflight(run, 2 * 64 * particles + 4 * 1024**2, "archive")
            paths = [output / f"particles_checkpoint{checkpoint:04d}_rank{rank}.csv" for rank in range(ranks)]
            if set(output.glob(f"particles_checkpoint{checkpoint:04d}_rank*.csv")) != set(paths):
                raise ValueError("Incomplete or unexpected checkpoint rank CSV set")
            sources, frames = [], []
            for path in paths:
                if path.is_symlink():
                    raise ValueError("CSV source must not be a symlink")
                before = sha256(path)
                frame = pd.read_csv(path, dtype={"id": np.uint64, **{key: np.float64 for key in Columns[1:]}},
                                    float_precision="round_trip")
                if list(frame.columns) != Columns or sha256(path) != before:
                    raise ValueError("CSV changed while reading or has unexpected columns")
                sources.append({"path": str(path), "sha256": before, "rows": len(frame), "bytes": path.stat().st_size})
                frames.append(frame)
            frame = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)
            ids = frame.id.to_numpy(dtype=np.uint64)
            phase = frame[Columns[1:]].to_numpy(dtype=np.float64)
            validate_arrays(ids, phase, particles)
            target = output / f"particles_checkpoint{checkpoint:04d}.npz"
            temporary = target.with_suffix(".npz.partial")
            pending = {"checkpoint": checkpoint, "path": str(target), "temporary": str(temporary),
                       "csv_sources": sources, "array_sha256": array_digest(ids, phase), "particles": particles}
            existing = run.get("pending_archive")
            if existing is not None and existing != pending:
                raise ValueError("Pending archive descriptor changed")
            if existing is None:
                if temporary.exists() or target.exists():
                    raise ValueError("Unrecorded archive already exists; no overwrite authorized")
                run["pending_archive"] = pending
                self._save()
            if target.exists():
                if target.is_symlink():
                    raise ValueError("Archive must not be a symlink")
                read_archive(target, particles, expected_digest=pending["array_sha256"])
            else:
                # An interrupted temporary write is owned by the persisted
                # transaction; original CSVs still exist and have been rechecked.
                if temporary.exists():
                    if temporary.is_symlink():
                        raise ValueError("Archive temporary file must not be a symlink")
                    temporary.unlink()
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, ids=ids, phase_space=phase)
                    stream.flush()
                    os.fsync(stream.fileno())
                recovered = read_archive(temporary, particles, expected_digest=pending["array_sha256"])
                if (not np.array_equal(recovered.id.to_numpy(), ids)
                        or recovered[Columns[1:]].to_numpy().tobytes() != phase.tobytes()):
                    raise ValueError("Archive differs from exact parsed CSV values")
                os.replace(temporary, target)
            record = {key: value for key, value in pending.items() if key != "temporary"}
            record.update(sha256=sha256(target), bytes=target.stat().st_size)
            run["snapshots"].append(record)
            run.pop("pending_archive", None)
            run["state"] = "archiving"
            self._save()  # Commit verified archive before removing any CSV.
            self._remove_recorded_csvs(record, output)
        self._verify_completed(run)

    @staticmethod
    def _remove_recorded_csvs(record, output):
        for source in record["csv_sources"]:
            path = Path(source["path"])
            if (path.parent != output or not re.fullmatch(r"particles_checkpoint[0-9]+_rank[0-9]+\.csv", path.name)
                    or path.is_symlink()):
                raise ValueError("Refusing to remove an unowned CSV path")
            if path.exists():
                if sha256(path) != source["sha256"]:
                    raise ValueError("CSV changed after archive verification; original retained")
                path.unlink()

    @staticmethod
    def _verify_snapshot(record, particles):
        return read_archive(record["path"], particles, expected_sha=record["sha256"],
                            expected_digest=record["array_sha256"])

    def _verify_completed(self, run):
        expected = set(range(run["descriptor"]["checkpoints"] + 1))
        if (len(run["snapshots"]) != len(expected)
                or {entry["checkpoint"] for entry in run["snapshots"]} != expected):
            raise ValueError("Incomplete archive checkpoint coverage")
        self._verify_hashes(run["retained_artifacts"])
        for record in run["snapshots"]:
            self._verify_snapshot(record, run["descriptor"]["particle_grid"]**3)

    def read_snapshot(self, name, checkpoint):
        self.output_dir(name)
        run = self.manifest["runs"][name]
        if run["state"] != "complete":
            raise ValueError("Run storage is not complete")
        matches = [entry for entry in run["snapshots"] if entry["checkpoint"] == checkpoint]
        if len(matches) != 1:
            raise ValueError("Checkpoint not archived exactly once")
        return self._verify_snapshot(matches[0], run["descriptor"]["particle_grid"]**3)
