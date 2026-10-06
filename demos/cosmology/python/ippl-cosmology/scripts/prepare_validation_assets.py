#!/usr/bin/env python3
## @file prepare_validation_assets.py
# @brief Copy nine released validation PNGs without rerendering or changing evidence.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Copy nine released validation PNGs without rerendering or changing evidence.

Example (standard library only)::

    python3 scripts/prepare_validation_assets.py \
        --validation-root /path/to/ippl-cosmology-linear

Run again with --verify-only to check the entire curated set. Existing files
must be byte-identical; differing files are never overwritten. Source report
snapshots are verified in place, not copied into the paper.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


## @var BUNDLES
# @brief Named BUNDLES protocol/schema value; the source initializer records its exact contents.
BUNDLES = (
    ("initial-conditions", "zarija-plots-final", "plot_manifest.json", (
        ("matched_ic_power.png", "ic-power.png"),
        ("mpi_rank_consistency.png", "ic-mpi.png"),
    )),
    ("particle-mesh", "pm-validation-plots-release", "manifest.json", (
        ("frozen_force_comparison.png", "frozen-force.png"),
        ("pancake_convergence.png", "pancake-convergence.png"),
        ("pancake_local_errors.png", "pancake-local.png"),
    )),
    ("nonlinear-evolution", "matched-evolution-5v7u3y98/figures-release",
     "sha256-manifest.json", (
         ("pancake-phase-space.png", "nonlinear-phase-space.png"),
         ("resolved-density-evolution.png", "nonlinear-modes.png"),
     )),
    ("local-spatial", "resolution-study-cffjva__/figures-spatial-release",
     "manifest.json", (("spatial-controls.png", "spatial-controls.png"),)),
    ("merlin-cpu-gaussian", "merlin-cpu-login-20261004/figures-gaussian-release",
     "manifest.json", (("gaussian-controls.png", "gaussian-controls.png"),)),
)


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param data Finite numerical or serialized record data in the routine's explicit schema.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


## @brief Compute the declared retained-file digest, with decompression only when explicitly requested.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @param compressed Whether the digest/reader applies the explicitly documented compression contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def file_hash(path: Path, *, compressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


## @brief Read the three historical release-manifest layouts, without guessing.
# @see cosmology_tools
#
# @param manifest Retained source/build/output provenance manifest; every referenced artifact must match its digest.
# @param name Stable artifact/run/check identifier as defined by the caller.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def output_record(manifest: dict, name: str) -> dict:
    """Read the three historical release-manifest layouts, without guessing."""
    records = []
    if "outputs" in manifest:
        records.extend(item for item in manifest["outputs"]
                       if Path(item["path"]).name == name)
    for key in ("outputs_sha256", "output_sha256"):
        records.extend({"path": path, "sha256": digest}
                       for path, digest in manifest.get(key, {}).items()
                       if Path(path).name == name)
    if len(records) != 1:
        raise ValueError(f"Expected exactly one release hash for {name}")
    return records[0]


## @brief Check bytes.
# @see cosmology_tools
#
# @param data Finite numerical or serialized record data in the routine's explicit schema.
# @param expected Independent or frozen expected value checked under the declared tolerance.
# @param label Stable human-readable curve or check label retained in the report.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def check_bytes(data: bytes, expected: dict, label: str) -> None:
    if sha256(data) != expected["sha256"]:
        raise ValueError(f"SHA256 mismatch: {label}")
    if "bytes" in expected and len(data) != expected["bytes"]:
        raise ValueError(f"Byte-count mismatch: {label}")


## @brief Verify exact report inputs; prefer the release snapshot over live state.
# @see cosmology_tools
#
# @param manifest Retained source/build/output provenance manifest; every referenced artifact must match its digest.
# @param bundle Released paper-figure bundle and its retained source/output manifest.
# @param validation_root Root of retained validation evidence from which released paper figures are copied or verified.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def report_records(manifest: dict, bundle: Path, validation_root: Path) -> list:
    """Verify exact report inputs; prefer the release snapshot over live state."""
    if "input_report" in manifest:
        source = manifest["input_report"]
        snapshot = bundle / Path(source["snapshot"]).name
        expected = output_record(manifest, snapshot.name)
        snapshot_hash = file_hash(snapshot)
        if snapshot_hash != expected["sha256"]:
            raise ValueError(f"Snapshot SHA256 mismatch: {snapshot}")
        if snapshot.stat().st_size != expected["bytes"]:
            raise ValueError(f"Snapshot byte-count mismatch: {snapshot}")
        if file_hash(snapshot, compressed=True) != source["sha256"]:
            raise ValueError(f"Decompressed report SHA256 mismatch: {snapshot}")
        return [{
            "original_path": source["path"], "sha256": source["sha256"],
            "verification": "exact decompressed release snapshot",
            "verified_path": str(snapshot),
            "snapshot_original_path": source["snapshot"],
            "snapshot_sha256": snapshot_hash,
            "snapshot_bytes": snapshot.stat().st_size,
            "copied": False,
        }]

    sources = manifest.get("source_sha256", manifest.get("inputs_sha256", {}))
    reports = []
    for original, digest in sources.items():
        if Path(original).suffix != ".json":
            continue
        # Historical local release paths can be relocated with the source tree.
        marker = "/build_openmp/demos/cosmology/"
        if marker not in original:
            raise ValueError(f"Unrecognized report location: {original}")
        path = validation_root / "build_openmp/demos/cosmology" / original.split(marker, 1)[1]
        if file_hash(path) != digest:
            raise ValueError(f"Source report SHA256 mismatch: {path}")
        reports.append({
            "original_path": original, "sha256": digest,
            "verification": "exact report bytes", "verified_path": str(path),
            "bytes": path.stat().st_size, "copied": False,
        })
    if not reports:
        raise ValueError(f"No source reports declared in {bundle}")
    return reports


## @brief Prepare the documented module workflow.
# @see cosmology_tools
#
# @param validation_root Root of retained validation evidence from which released paper figures are copied or verified.
# @param output_dir Output directory; use a fresh location when required by the workflow.
# @param verify_only Verify released assets in place without copying or changing any output bytes.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def prepare(validation_root: Path, output_dir: Path, *, verify_only: bool = False) -> dict:
    validation_root = validation_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    base = validation_root / "build_openmp/demos/cosmology"
    files: dict[Path, bytes] = {}
    source_checks: dict[Path, str] = {}
    assets, provenance = [], []

    # Validate every source before creating any destination file.
    for name, relative, manifest_name, selected in BUNDLES:
        bundle = base / relative
        manifest_path = bundle / manifest_name
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_hash = sha256(manifest_bytes)
        destination = Path("provenance") / f"{name}-plot-manifest.json"
        files[destination] = manifest_bytes
        source_checks[manifest_path] = manifest_hash
        provenance.append({
            "bundle": name, "source_manifest": str(manifest_path),
            "source_manifest_sha256": manifest_hash,
            "source_manifest_bytes": len(manifest_bytes),
            "preserved_manifest": str(destination),
            "source_reports": report_records(manifest, bundle, validation_root),
        })
        for source_name, paper_name in selected:
            source_path = bundle / source_name
            data = source_path.read_bytes()
            expected = output_record(manifest, source_name)
            check_bytes(data, expected, str(source_path))
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Not a PNG: {source_path}")
            files[Path(paper_name)] = data
            source_checks[source_path] = sha256(data)
            assets.append({
                "asset": paper_name, "bundle": name,
                "source_path": str(source_path),
                "original_manifest_output_path": expected["path"],
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": manifest_hash,
                "sha256": sha256(data), "bytes": len(data),
                "copy_contract": "byte-identical released PNG; no plot edits",
            })

    script = Path(__file__).resolve()
    paper_manifest = {
        "schema": "ippl-cosmology-paper-validation-assets-v1",
        "validation_root": str(validation_root), "output_dir": str(output_dir),
        "script": {"path": str(script), "sha256": file_hash(script)},
        "asset_count": len(assets), "total_png_bytes": sum(a["bytes"] for a in assets),
        "simulations_run": False, "plots_rerendered": False,
        "source_reports_copied": False,
        "scope": "Figure-byte provenance only; retained scientific failures are unchanged.",
        "assets": assets, "provenance": provenance,
    }
    files[Path("asset-manifest.json")] = (
        json.dumps(paper_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    # Preflight the entire destination, so a known conflict cannot leave a
    # partially copied set. Exclusive creation also rejects concurrent writers.
    for relative, data in files.items():
        path = output_dir / relative
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {path}")
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"Refusing to overwrite differing file: {path}")
        elif verify_only:
            raise ValueError(f"Missing curated file: {path}")
    if not verify_only:
        for relative, data in files.items():
            path = output_dir / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(data)

    for relative, data in files.items():
        if (output_dir / relative).read_bytes() != data:
            raise ValueError(f"Post-copy verification failed: {relative}")
    for source, digest in source_checks.items():
        if file_hash(source) != digest:
            raise ValueError(f"Source changed during verification: {source}")
    return {
        "mode": "verify-only" if verify_only else "prepare",
        "asset_count": len(assets), "total_png_bytes": paper_manifest["total_png_bytes"],
        "manifest": str(output_dir / "asset-manifest.json"),
        "manifest_sha256": sha256(files[Path("asset-manifest.json")]),
        "verified_files": len(files),
    }


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True,
                        help="Root of the preserved ippl-cosmology-linear worktree")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "ippl-cosmology/figures/validation")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify sources and all curated files without writing")
    args = parser.parse_args()
    try:
        result = prepare(args.validation_root, args.output_dir, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Validation asset preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
