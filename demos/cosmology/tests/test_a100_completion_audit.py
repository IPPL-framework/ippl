"""Synthetic evidence contract checks; no hardware or physics pass is fabricated."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'merlin'))
import audit_a100_study as audit
import test_merlin_gpu_mpiexec as mock_launcher


def report_fixture(stage, backend):
    cases = audit.study_cases(stage, False)
    runs = []
    for case in cases:
        ai, af = audit.epochs(case, False)
        for code in audit.Codes:
            metadata = dict(ranks=str(case.ranks), threads='1', n_particles_grid=str(case.particles),
                n_grid=str(case.mesh), n_steps=str(case.steps), n_checkpoints='8',
                box_size=str(audit.Parameters['box_size']), omega_m=str(audit.Parameters['omega_m']),
                a_initial=str(ai), a_final=str(af))
            if code == 'ippl':
                metadata.update(execution_space=backend, memory_space='Cuda' if backend == 'Cuda' else 'Host',
                                host_threads='1', execution_concurrency='2048' if backend == 'Cuda' else '1')
                metadata['threads'] = metadata['execution_concurrency']
            runs.append(dict(name=case.name + '_' + code, code=code, metadata=metadata, **asdict(case)))
    return dict(state='complete', complete=True, integrity_passed=True,
        configuration=dict(stage=stage, smoke=False), expected_runs=len(runs), runs=runs,
        planned_cases=[asdict(case) for case in cases],
        completed_stages=['spatial', 'gaussian'] if stage == 'all' else [stage],
        parameters=audit.canonical(audit.Parameters), budgets=audit.canonical(audit.Budgets),
        checks=[dict(name='synthetic-integrity', category='integrity', passed=True),
                dict(name='synthetic-scope-only', passed=False)],
        failed_checks=[dict(name='synthetic-scope-only', passed=False)],
        all_checks_passed=False, passed=False, qualification={'scope': 'synthetic test only'})


class CompletionAuditTests(unittest.TestCase):
    def test_launch_manifest_join_checks_actual_log_and_refuses_duplicates(self):
        fixture = mock_launcher.LauncherTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = fixture.root
        storage = root / 'storage'
        storage.mkdir()
        target = fixture.files['CompareCosmologyEvolution']
        args = [target, '8', '8', '168.75', '.31', '.02', '.2', '16', '8',
                str(root / 'input.csv'), str(storage / 'synthetic_ippl')]
        rows = [fixture.binding(rank) for rank in range(2)]
        for row in rows:
            row.update(executable=target, requested_executable=target)
        status, manifest, _, _, raw = fixture.mocked_launch(rows, target='CompareCosmologyEvolution', arguments=args)
        self.assertEqual(status, 0)
        log = storage / 'synthetic_ippl.log'
        log.write_bytes(raw)
        journal = {'runs': {'synthetic_ippl': dict(log_path=str(log), descriptor=dict(command=[
            str(audit.launcher.__file__), '--config', str(fixture.configPath), '-n', '2', *args]))}}
        (storage / 'storage.json').write_text(json.dumps(journal))
        report = {'runs': [dict(name='synthetic_ippl', code='ippl', ranks=2)]}
        evidence = root / 'evidence'
        result = audit.verify_launch_evidence(root / 'results.json', report, evidence)
        self.assertEqual(len(result['launches']), 1)
        self.assertEqual(len(result['launches'][0]['observed_pci']), 2)
        log.write_bytes(raw + b'changed outer log')
        with self.assertRaisesRegex(ValueError, 'log'):
            audit.verify_launch_evidence(root / 'results.json', report, evidence)
        log.write_bytes(raw)
        (evidence / 'duplicate.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            audit.verify_launch_evidence(root / 'results.json', report, evidence)

    def test_complete_execution_does_not_relabel_scientific_failures(self):
        result = audit.validate_scope(report_fixture('all', 'Cuda'), 'all', 'Cuda')
        self.assertEqual(result['executed_runs'], 52)
        self.assertEqual(result['failed_checks'], 1)
        self.assertFalse(result['scientific_acceptance'])
        self.assertEqual(audit.validate_scope(report_fixture('gaussian', 'OpenMP'),
                                             'gaussian', 'OpenMP')['executed_runs'], 18)

    def test_cpu_fallback_wrong_memory_and_mismatched_parameters_rejected(self):
        for field, value in (('execution_space', 'OpenMP'), ('memory_space', 'CudaUVM'),
                             ('n_grid', '128'), ('host_threads', '2')):
            with self.subTest(field=field):
                report = report_fixture('all', 'Cuda')
                report['runs'][0]['metadata'][field] = value
                with self.assertRaises(ValueError):
                    audit.validate_scope(report, 'all', 'Cuda')

    def test_incomplete_duplicate_reduced_and_relabelled_reports_rejected(self):
        for change in ('incomplete', 'duplicate', 'smoke', 'budget', 'relabel', 'missing_stage'):
            with self.subTest(change=change):
                report = report_fixture('all', 'Cuda')
                if change == 'incomplete': report['complete'] = False
                if change == 'duplicate': report['runs'][-1] = deepcopy(report['runs'][0])
                if change == 'smoke': report['configuration']['smoke'] = True
                if change == 'budget': report['budgets']['time_shell_complex'] *= 2
                if change == 'relabel': report['passed'] = True
                if change == 'missing_stage': report['completed_stages'] = ['gaussian']
                with self.assertRaises(ValueError):
                    audit.validate_scope(report, 'all', 'Cuda')

    def test_integrity_summary_cannot_override_missing_or_failed_checks(self):
        for change in ('missing', 'failed', 'not_boolean'):
            with self.subTest(change=change):
                report = report_fixture('all', 'Cuda')
                if change == 'missing': report['checks'].pop(0)
                if change == 'failed':
                    report['checks'][0]['passed'] = False
                    report['failed_checks'] = deepcopy(report['checks'])
                if change == 'not_boolean': report['checks'][0]['passed'] = 1
                with self.assertRaises(ValueError):
                    audit.validate_scope(report, 'all', 'Cuda')

    def test_storage_hashes_and_checkpoint_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / 'storage'
            storage.mkdir()
            source = root / 'source.txt'
            source.write_text('synthetic source')
            metadata = storage / 'metadata.txt'
            metadata.write_text('ranks=1\nthreads=2048\n')
            snapshots = []
            for checkpoint in range(9):
                path = storage / f'checkpoint{checkpoint}.npz'
                path.write_bytes(f'synthetic archive bytes {checkpoint}'.encode())
                snapshots.append(dict(path=str(path), sha256=audit.file_hash(path), checkpoint=checkpoint))
            report = dict(provenance={str(source): audit.file_hash(source)},
                          source_snapshot={str(source): audit.file_hash(source)}, fixtures={},
                          runs=[dict(name='synthetic', metadata={'ranks': '1', 'threads': '2048'})])
            journal = dict(runs={'synthetic': dict(state='complete', storage_complete=True, return_code=0,
                retained_artifacts={str(metadata): audit.file_hash(metadata)}, snapshots=snapshots,
                descriptor={'output': str(storage)})})
            journal_path = storage / 'storage.json'
            journal_path.write_text(json.dumps(journal))
            result = audit.verify_saved_files(root / 'results.json', report)
            self.assertEqual(result['verified_snapshot_archives'], 9)
            self.assertEqual(result['verified_file_hashes'], 11)
            Path(snapshots[0]['path']).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                audit.verify_saved_files(root / 'results.json', report)
            journal['runs']['synthetic']['snapshots'].pop()
            journal_path.write_text(json.dumps(journal))
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                audit.verify_saved_files(root / 'results.json', report)


if __name__ == '__main__':
    unittest.main()
