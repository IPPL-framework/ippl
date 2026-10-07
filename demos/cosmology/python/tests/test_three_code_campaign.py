#!/usr/bin/env python3
## @file test_three_code_campaign.py
# @brief Regression checks for shared ICs, cached builds and the Figure A11 denominator.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Regression checks for shared ICs, cached builds and the Figure A11 denominator."""
from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import gaussian_fixture
import run_three_code_campaign as campaign
from gadget2.convert_shared_ic import gadget_header, write_record


## @brief Regression suite for ThreeCodeCampaign.
# @see cosmology_tools
class ThreeCodeCampaignTests(unittest.TestCase):
    ## @brief Verify broadband matches existing low band convention.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_broadband_matches_existing_low_band_convention(self):
        actual,metadata=campaign.make_broadband_fixture(8,2,20261003)
        expected,original=gaussian_fixture.make_gaussian_fixture(8,99,20261003,cutoff=2)
        np.testing.assert_array_equal(actual.to_numpy(),expected.to_numpy())
        self.assertEqual(metadata['coefficient_sha256'],original['coefficient_sha256'])
        self.assertEqual(metadata['phase_space_sha256'],original['phase_space_sha256'])

    ## @brief Verify invalid nyquist and grid rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_invalid_nyquist_and_grid_rejected(self):
        for grid,cutoff in [(8,4),(7,2),(4,0)]:
            with self.assertRaises(ValueError):campaign.make_broadband_fixture(grid,cutoff,1)

    ## @brief Verify existing executables are not rebuilt.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_existing_executables_are_not_rebuilt(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(ippl_build=root/'ippl',fastpm_build=root/'fastpm',
                                 gadget_build=root/'gadget',build_logs=root/'logs',jobs=2,grid=128)
            paths=[args.ippl_build/'demos/cosmology/Cosmology',
                   args.ippl_build/'demos/cosmology/CompareCosmologyEvolution',
                   args.fastpm_build/'evolution/FastPMEvolution',
                   args.gadget_build/'Gadget2-TreePM-128-double']
            for path in paths:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o755)
            timestamps=[p.stat().st_mtime_ns for p in paths]
            with patch.object(campaign,'run_command',side_effect=AssertionError('Unexpected build')):
                _,records=campaign.ensure_builds(args)
            self.assertEqual(records,[])
            self.assertEqual(timestamps,[p.stat().st_mtime_ns for p in paths])

    ## @brief Verify both pm checkpoint formats and bad epoch.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_both_pm_checkpoint_formats_and_bad_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoints.csv'
            frame=pd.DataFrame({'checkpoint':[0,1],'step':[0,24],'a':[.01,1.],
                                'mass_error':[0.,1e-15],'max_inverse_imaginary':[0.,1e-16]})
            frame.to_csv(path,index=False)
            campaign.validate_pm_diagnostics(path,1,24,16)
            frame['particle_count']=4096
            frame.to_csv(path,index=False)
            campaign.validate_pm_diagnostics(path,1,24,16)
            frame.loc[1,'a']=.5
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'epoch'):
                campaign.validate_pm_diagnostics(path,1,24,16)

    ## @brief Verify gadget paths fit legacy filename buffers.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_gadget_paths_fit_legacy_filename_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'runs').mkdir()
            path=campaign.gadget_parameters(root,128,120)
            parameters=dict(line.split(maxsplit=1) for line in path.read_text().splitlines())
            for key in ('InitCondFile','OutputDir','OutputListFilename'):
                self.assertFalse(Path(parameters[key]).is_absolute())
                self.assertLess(len(parameters[key]),100)
            self.assertEqual(parameters['OutputListOn'],'0')

    ## @brief Verify gadget endpoint and ids verified.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_gadget_endpoint_and_ids_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'snapshot'
            def snapshot(a,ids):
                with path.open('wb') as stream:
                    write_record(stream,gadget_header(count=64,mass_code=1,a=a,box_kpc_h=168750,
                                                     omega_m=.31,omega_lambda=.69,hubble=.675))
                    write_record(stream,np.zeros((64,3),dtype='<f4').tobytes())
                    write_record(stream,np.zeros((64,3),dtype='<f4').tobytes())
                    write_record(stream,ids.astype('<u4').tobytes())
            snapshot(1,np.arange(64))
            self.assertEqual(campaign.read_gadget_snapshot(path,4).shape,(64,3))
            snapshot(.5,np.arange(64))
            with self.assertRaisesRegex(ValueError,'epoch'):campaign.read_gadget_snapshot(path,4)
            snapshot(1,np.zeros(64))
            with self.assertRaisesRegex(ValueError,'IDs'):campaign.read_gadget_snapshot(path,4)

    ## @brief Verify plot offsets use fastpm and nonpositive bins are not floored.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_plot_offsets_use_fastpm_and_nonpositive_bins_are_not_floored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'analysis.json').write_text('{}')
            powers={'ippl':[2.,-1.],'fastpm':[1.,2.],'gadget2':[3.,4.],'ippl_gpu1':[1.5,3.],'ippl_gpu4':[1.,2.]}
            data={'configuration':{'grid':8,'steps':24,'cutoff':2,'seed':1,'smoke':True},
                  'spectra':{code:[{'k_h_per_mpc':.1*(i+1),'P_shot_subtracted':power}
                                   for i,power in enumerate(values)] for code,values in powers.items()}}
            campaign.render_figure(root,data)
            frame=pd.read_csv(root/'figures/plotted_values.csv')
            self.assertEqual(frame.ippl_offset_vs_fastpm_percent.iloc[0],100.)
            self.assertEqual(frame.gadget2_offset_vs_fastpm_percent.iloc[0],200.)
            self.assertEqual(frame.ippl_gpu1_offset_vs_fastpm_percent.iloc[0],50.)
            self.assertEqual(frame.ippl_gpu4_offset_vs_fastpm_percent.iloc[0],0.)
            self.assertTrue(np.isnan(frame.ippl_offset_vs_fastpm_percent.iloc[1]))
            self.assertEqual(frame.ippl_P_shot_subtracted.iloc[1],-1.)
            manifest=json.loads((root/'figures/manifest.json').read_text())
            self.assertEqual(manifest['ratio_denominator'],'FastPM')

    ## @brief Verify rank parsing rejects duplicates and invalid counts.
    # @return None; assertions verify CLI rank contracts.
    def test_rank_list(self):
        self.assertEqual(campaign.parse_rank_list('1'),[1])
        self.assertEqual(campaign.parse_rank_list('1,4'),[1,4])
        for value in ('0','1,1','1,','gpu','-4'):
            with self.assertRaises(ValueError):campaign.parse_rank_list(value)

    ## @brief Verify omitted cluster selects local execution with one rank.
    # @return None; assertions verify resolved CLI defaults without any builds.
    def test_default_local_and_cluster_plan(self):
        import contextlib,io
        stream=io.StringIO()
        with patch.object(sys,'argv',['runner','--plan']),contextlib.redirect_stdout(stream):
            self.assertEqual(campaign.main(),0)
        resolved=json.loads(stream.getvalue());self.assertIsNone(resolved['cluster']);self.assertEqual(resolved['ranks'],1)
        stream=io.StringIO()
        with patch.object(sys,'argv',['runner','--cluster','merlin6','--rank','1,4','--shared-ic','fixture.csv','--plan']),contextlib.redirect_stdout(stream):
            self.assertEqual(campaign.main(),0)
        resolved=json.loads(stream.getvalue());self.assertEqual(resolved['rank'],[1,4]);self.assertEqual(resolved['scheduler_cluster'],'gmerlin6')

    ## @brief Require exact baseline IC identity before accepting supplementary GPU powers.
    # @return None; an otherwise complete GPU record with a different IC is rejected.
    def test_gpu_merge_rejects_different_shared_ic(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';gpu=root/'gpu';base.mkdir();gpu.mkdir()
            data={'configuration':{'grid':8,'cutoff':2,'steps':24,'checkpoints':1,'seed':1,'smoke':True},'spectra':{},'input_hashes':{}}
            campaign.write_json(base/'analysis.json',data)
            campaign.write_json(base/'campaign.json',{'complete':True,'analysis_sha256':campaign.sha256(base/'analysis.json'),'provenance':{str((base/'ics/shared-z99.csv').resolve()):'original'}})
            config={**data['configuration'],'ranks':1,'shared_ic_sha256':'different'}
            campaign.write_json(gpu/'gpu-analysis.json',{'complete':True,'configuration':config})
            with self.assertRaisesRegex(ValueError,'IC'):
                campaign.extend_figure(base,[gpu],root/'new')

    ## @brief Verify GPU jobs depend on a successful build and request one physical GPU per rank.
    # @return None; mocked Slurm calls are checked without submitting any jobs.
    def test_slurm_dependencies_and_gpu_requests(self):
        from types import SimpleNamespace
        import contextlib,io
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ic=root/'shared-z99.csv';ic.write_text('fixture')
            campaign.write_json(root/'ic-manifest.json',{'csv_sha256':'digest','particle_grid':8,'seed':1,'cutoff_fundamental':2,'redshift_initial':99})
            args=SimpleNamespace(output=root/'remote',plan=False,shared_ic=ic,grid=8,seed=1,cutoff=2,steps=24,checkpoints=1,smoke=True,timeout=300,remote_python='/python',rank_list=[1,4],build_only=False)
            identifiers=[SimpleNamespace(stdout=f'{value};gmerlin6\n') for value in (101,102,103)]
            with patch.object(campaign.socket,'gethostname',return_value='merlin-l-001'),patch.object(campaign,'sha256',return_value='digest'),patch.object(campaign,'executable',return_value=True),patch.object(campaign.subprocess,'run',side_effect=identifiers) as submit,contextlib.redirect_stdout(io.StringIO()):
                campaign.submit_merlin(args)
            commands=[call.args[0] for call in submit.call_args_list]
            self.assertEqual(len(commands),3)
            self.assertIn('--gres=gpu:1',commands[0]);self.assertIn('--cpus-per-task=4',commands[0])
            for rank,command in zip((1,4),commands[1:]):
                self.assertIn(f'--gres=gpu:{rank}',command);self.assertIn(f'--ntasks={rank}',command)
                self.assertIn('--dependency=afterok:101',command)
            self.assertEqual(json.loads((args.output/'submission.json').read_text())['jobs']['gpu4']['job_id'],'103')

    ## @brief Reject several MPI ranks bound to the same physical GPU before computing a spectrum.
    # @return None; aliased PCI binding evidence raises a validation error.
    def test_gpu_analysis_rejects_shared_physical_device(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);root=parent/'gpu4';root.mkdir();ic=parent/'ic';ic.write_text('fixture')
            campaign.write_json(root/'configuration.json',{'ranks':4,'source_sha256':{},'shared_ic':str(ic),'shared_ic_sha256':campaign.sha256(ic)})
            campaign.write_json(parent/'build-manifest.json',{'artifacts':{}})
            (root/'allocated-gpus.csv').write_text(''.join(f'NVIDIA A100, GPU-{rank}, 00000000:{rank+1:02x}:00.0, Disabled\n' for rank in range(4)))
            records=[{'rank':rank,'pci':'0000:01:00.0','host':'merlin-g-100','world_size':4,'local_size':4,'runtime_device_count':1,'visible_device_ordinal':0} for rank in range(4)]
            (root/'solver.log').write_text(''.join('GPU_BINDING '+json.dumps(row)+'\n' for row in records))
            with self.assertRaisesRegex(ValueError,'distinct physical GPU'):
                campaign.analyze_gpu(root)

    ## @brief Verify downloaded-path relocation, GPU ratios and rejection of altered downloaded evidence.
    # @return None; five-series outputs are checked and tampering is rejected.
    def test_downloaded_gpu_figure_merge_and_hash_verification(self):
        import contextlib,io
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir()
            config={'grid':8,'cutoff':2,'steps':24,'checkpoints':1,'seed':1,'smoke':True}
            rows=[{'k_h_per_mpc':.1*(i+1),'P_shot_subtracted':float(i+1)} for i in range(2)]
            data={'configuration':config,'spectra':{code:rows for code in campaign.Codes},'input_hashes':{}}
            campaign.write_json(base/'analysis.json',data)
            campaign.write_json(base/'campaign.json',{'complete':True,'analysis_sha256':campaign.sha256(base/'analysis.json'),'provenance':{str((base/'ics/shared-z99.csv').resolve()):'shared'}})
            directories=[]
            for rank in (1,4):
                gpu=root/f'gpu{rank}';gpu.mkdir();(gpu/'solver.log').write_text('retained log')
                remote=f'/remote/gpu{rank}'
                record={'complete':True,'configuration':{**config,'ranks':rank,'shared_ic_sha256':'shared','result_root':remote},'spectra':rows,'artifacts':{remote+'/solver.log':campaign.sha256(gpu/'solver.log')}}
                campaign.write_json(gpu/'gpu-analysis.json',record);directories.append(gpu)
            with contextlib.redirect_stdout(io.StringIO()):campaign.extend_figure(base,directories,root/'comparison')
            actual=pd.read_csv(root/'comparison/figures/plotted_values.csv')
            np.testing.assert_array_equal(actual.ippl_gpu1_offset_vs_fastpm_percent,[0.,0.])
            np.testing.assert_array_equal(actual.ippl_gpu4_offset_vs_fastpm_percent,[0.,0.])
            self.assertEqual(len(json.loads((root/'comparison/analysis.json').read_text())['spectra']),5)
            (directories[0]/'solver.log').write_text('altered log')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                campaign.extend_figure(base,directories,root/'bad')

    ## @brief Reject an existing remote output directory before any shared IC upload.
    # @return None; a failed exclusive directory creation stops SCP before touching evidence.
    def test_remote_existing_output_stops_before_upload(self):
        from types import SimpleNamespace
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            ic=Path(directory)/'shared-z99.csv';ic.write_text('fixture')
            args=SimpleNamespace(output=Path('/remote/existing'),plan=False,shared_ic=ic)
            with patch.object(campaign.socket,'gethostname',return_value='local'),patch.object(campaign.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['ssh'])) as run:
                with self.assertRaises(subprocess.CalledProcessError):campaign.submit_merlin(args)
            self.assertEqual(run.call_count,1)
            command=run.call_args.args[0];self.assertEqual(command[:2],['ssh','merlin6'])
            self.assertIn('mkdir /remote/existing && mkdir /remote/existing/input',command[2])

    ## @brief Permit cache reuse after launcher changes while rejecting changed native compilation inputs.
    # @return None; hash-scope comparisons preserve the original native build contract.
    def test_native_cache_and_launch_hash_scopes(self):
        before={'/src/Field.h':'header','/src/CMakeLists.txt':'configure','/tools/nvcc_wrapper':'compiler','/src/runner.py':'python','/src/a11_job.sh':'shell','/src/tuning.csv':'runtime-data'}
        runtime={**before,'/src/runner.py':'new-python','/src/a11_job.sh':'new-shell','/src/tuning.csv':'new-runtime-data'}
        self.assertEqual(campaign.native_source_hashes(before),campaign.native_source_hashes(runtime))
        changed={**runtime,'/src/Field.h':'changed-header'}
        self.assertNotEqual(campaign.native_source_hashes(before),campaign.native_source_hashes(changed))
        self.assertEqual(set(campaign.native_source_hashes(before)),{'/src/Field.h','/src/CMakeLists.txt','/tools/nvcc_wrapper'})


## @cond CLI_DISPATCH
if __name__=='__main__':unittest.main()
## @endcond
