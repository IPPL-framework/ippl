#!/usr/bin/env python3
## @file audit_a100_study.py
# @brief Verify complete A100/CPU study scope and saved hashes, not physical acceptance.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Verify complete A100/CPU study scope and saved hashes, not physical acceptance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_metadata import validate_runtime_metadata
from validate_resolution_study import Budgets, Codes, Parameters, canonical, epochs, study_cases
import gpu_mpiexec as launcher


## @brief Compute the declared retained-file digest, with decompression only when explicitly requested.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


## @brief Validate scope.
# @see cosmology_tools
#
# @param report Structured campaign/audit report; recorded failures are not retroactively changed.
# @param stage Declared campaign stage (for example spatial or Gaussian), not a inferred favorable subset.
# @param execution_space Expected recorded execution-space name (for example CUDA or OpenMP).
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def validate_scope(report, stage, execution_space):
    cases = study_cases(stage, False)
    if not (report.get('state') == 'complete' and report.get('complete') is True
            and report.get('integrity_passed') is True):
        raise ValueError('Study is incomplete or its integrity gates failed')
    if report['configuration']['stage'] != stage or report['configuration']['smoke']:
        raise ValueError('Wrong study stage or a reduced smoke matrix')
    expected_stages = ['spatial', 'gaussian'] if stage == 'all' else [stage]
    if report['completed_stages'] != expected_stages:
        raise ValueError('Missing completed stage')
    if report['planned_cases'] != [asdict(case) for case in cases]:
        raise ValueError('Planned case matrix differs from the full declared protocol')
    if report['parameters'] != canonical(Parameters) or report['budgets'] != canonical(Budgets):
        raise ValueError('Parameters or numerical budgets differ from the declared protocol')
    expected = {case.name + '_' + code: (case, code) for case in cases for code in Codes}
    runs = report['runs']
    if (report['expected_runs'] != len(expected) or len(runs) != len(expected)
            or {run['name'] for run in runs} != set(expected)):
        raise ValueError('Missing, duplicate, or unexpected executed run')
    for run in runs:
        case, code = expected[run['name']]
        if run['code'] != code or any(run[key] != value for key, value in asdict(case).items()):
            raise ValueError('Executed run descriptor differs from the planned case')
        ai, af = epochs(case, False)
        contract = dict(n_particles_grid=case.particles, n_grid=case.mesh, n_steps=case.steps,
                        n_checkpoints=8, box_size=Parameters['box_size'], omega_m=Parameters['omega_m'],
                        a_initial=ai, a_final=af)
        if any(float(run['metadata'][key]) != value for key, value in contract.items()):
            raise ValueError('Executed mesh/particle/time/cosmology metadata differs from protocol')
        runtime = validate_runtime_metadata(run['metadata'], case.ranks, 1, code=code,
            expected_execution_space=execution_space if code == 'ippl' else None)
        if code == 'ippl' and execution_space == 'Cuda' and runtime['memory_space'] != 'Cuda':
            raise ValueError('A100 protocol requires actual CUDA device memory, not managed/host memory')
    checks = report['checks']
    if not checks or len({check['name'] for check in checks}) != len(checks):
        raise ValueError('Missing or duplicate scientific checks')
    if any(type(check.get('passed')) is not bool for check in checks):
        raise ValueError('Check results must be explicit booleans')
    integrity = [check for check in checks if check.get('category') == 'integrity']
    if not integrity or not all(check['passed'] for check in integrity):
        raise ValueError('Recorded integrity checks are missing or failed')
    failures = [check for check in checks if check['passed'] is not True]
    if report['failed_checks'] != failures or report['all_checks_passed'] != (not failures):
        raise ValueError('Scientific success/failure summary is inconsistent')
    if report['passed'] != (not failures):
        raise ValueError('Completed study acceptance flag is inconsistent')
    return {'executed_runs': len(runs), 'ippl_backend': execution_space,
            'completed_stages': expected_stages, 'failed_checks': len(failures),
            'scientific_acceptance': report['passed'], 'qualification': report['qualification']}


## @brief Check every recorded archive and retained artifact without altering journals.
# @see cosmology_tools
#
# @param report_path Retained report path, including the provenance needed by the audit.
# @param report Structured campaign/audit report; recorded failures are not retroactively changed.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def verify_saved_files(report_path, report):
    """Check every recorded archive and retained artifact without altering journals."""
    hashes = {}
    def include(path, digest):
        if path in hashes and hashes[path] != digest:
            raise ValueError('Conflicting hashes for one artifact')
        hashes[path] = digest
    for mapping in (report['provenance'], report['source_snapshot']):
        if not mapping:
            raise ValueError('Missing source provenance')
        for path, digest in mapping.items():
            include(path, digest)
    for fixture in report['fixtures'].values():
        include(fixture['path'], fixture['sha256'])
    storage = json.loads((report_path.parent / 'storage/storage.json').read_text())
    if set(storage['runs']) != {run['name'] for run in report['runs']}:
        raise ValueError('Storage and analyzed run sets differ')
    archives = 0
    for name, run in storage['runs'].items():
        if run['state'] != 'complete' or run.get('storage_complete') is not True or run['return_code'] != 0:
            raise ValueError('An archived execution is not complete')
        if sorted(snapshot['checkpoint'] for snapshot in run['snapshots']) != list(range(9)):
            raise ValueError('Missing or duplicate archived checkpoint')
        for path, digest in run['retained_artifacts'].items():
            include(path, digest)
        for snapshot in run['snapshots']:
            include(snapshot['path'], snapshot['sha256'])
            archives += 1
        metadata_path = Path(run['descriptor']['output']) / 'metadata.txt'
        metadata = dict(line.split('=', 1) for line in metadata_path.read_text().splitlines() if '=' in line)
        recorded = next(item for item in report['runs'] if item['name'] == name)
        if metadata != recorded['metadata']:
            raise ValueError('Recorded metadata differs from the retained execution output')
    for path, digest in hashes.items():
        if file_hash(path) != digest:
            raise ValueError(f'Artifact changed: {path}')
    return {'verified_file_hashes': len(hashes), 'verified_snapshot_archives': archives,
            'scope': 'Byte integrity and completed matrix; not independent recomputation of physics metrics'}


## @brief Join actual adapter argv to one verified launch, excluding CTests/smokes.
# @see cosmology_tools
#
# @param report_path Retained report path, including the provenance needed by the audit.
# @param report Structured campaign/audit report; recorded failures are not retroactively changed.
# @param evidence_directory Directory of retained allocation/binding/launch evidence; all referenced hashes must match.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def verify_launch_evidence(report_path, report, evidence_directory):
    """Join actual adapter argv to one verified launch, excluding CTests/smokes."""
    manifests = [(path, json.loads(path.read_text())) for path in sorted(evidence_directory.glob('*.json'))]
    storage = json.loads((report_path.parent / 'storage/storage.json').read_text())
    joined = []
    for run in report['runs']:
        archived = storage['runs'][run['name']]
        config_path, ranks, target_args = launcher.parse_arguments(archived['descriptor']['command'][1:])
        candidates = [(path, manifest) for path, manifest in manifests
                      if manifest.get('requested_target_arguments') == target_args]
        if len(candidates) != 1:
            raise ValueError('Each executed study run requires exactly one matching launcher manifest')
        path, manifest = candidates[0]
        config, target, kind, config_digest = launcher.load_config(config_path, target_args)
        allocated = launcher.allocation_evidence(config['allocation_evidence'])
        if (ranks != run['ranks'] or manifest['ranks'] != ranks
                or manifest['status'] != 'launch_complete' or manifest['return_code'] != 0
                or manifest['wrapper_return_code'] != 0 or manifest['binding_errors']
                or manifest['target'] != target or manifest['target_kind'] != kind
                or kind != ('gpu' if run['code'] == 'ippl' else 'cpu')
                or manifest['command'] != launcher.build_command(config, ranks, kind, target_args)
                or manifest['allocation_evidence'] != allocated):
            raise ValueError('Launcher execution/topology contract differs from the archived run')
        required_hashes = {**config['sha256'], str(config_path): config_digest,
                           allocated['path']: allocated['sha256']}
        if any(manifest['hashes'].get(p) != h for p, h in required_hashes.items()):
            raise ValueError('Launcher manifest lacks required pinned artifact/allocation hashes')
        launcher.verify_hashes(manifest['hashes'])
        if (file_hash(manifest['log']) != manifest['log_sha256']
                or file_hash(archived['log_path']) != manifest['log_sha256']):
            raise ValueError('Launcher log and archived execution log differ')
        records = [json.loads(line[len(launcher.BindingPrefix):])
                   for line in Path(manifest['log']).read_bytes().splitlines()
                   if line.startswith(launcher.BindingPrefix)]
        if records != manifest['bindings']:
            raise ValueError('Recorded GPU bindings differ from actual launch output')
        launcher.validate_bindings(records, ranks, kind, config, target, target_args[0], allocated)
        launcher.allocation(manifest['slurm'])
        if manifest['thread_environment'].get('OMP_NUM_THREADS') != '1':
            raise ValueError('Full nonlinear study must retain one host thread per rank')
        joined.append(dict(run=run['name'], ranks=ranks, kind=kind, manifest=str(path.resolve()),
                           manifest_sha256=file_hash(path), observed_pci=[row['pci'] for row in records]))
    return {'scope': 'Per-run argv, hashes, original log, rank bindings and allocated full-A100 PCI membership',
            'launches': joined}


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu_report', type=Path)
    parser.add_argument('--cpu-report', type=Path, required=True)
    parser.add_argument('--launch-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Audit output must be new')
    result = {'schema': 'ippl-a100-completion-audit-v1',
              'scope': 'Execution scope/backend/provenance audit, NOT a declaration that physics gates passed'}
    for label, path, stage, backend in (('cpu', args.cpu_report, 'gaussian', 'OpenMP'),
                                      ('a100', args.gpu_report, 'all', 'Cuda')):
        before = file_hash(path)
        report = json.loads(path.read_text())
        result[label] = validate_scope(report, stage, backend)
        result[label].update(verify_saved_files(path, report))
        if file_hash(path) != before:
            raise ValueError('Report changed during audit')
        result[label].update(report_path=str(path.resolve()), report_sha256=before)
        if label == 'a100':
            result[label]['launch_evidence'] = verify_launch_evidence(path, report, args.launch_evidence)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


## @cond CLI_DISPATCH
if __name__ == '__main__':
    main()
## @endcond
