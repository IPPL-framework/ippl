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
            powers={'ippl':[2.,-1.],'fastpm':[1.,2.],'gadget2':[3.,4.]}
            data={'configuration':{'grid':8,'steps':24,'cutoff':2,'seed':1,'smoke':True},
                  'spectra':{code:[{'k_h_per_mpc':.1*(i+1),'P_shot_subtracted':power}
                                   for i,power in enumerate(values)] for code,values in powers.items()}}
            campaign.render_figure(root,data)
            frame=pd.read_csv(root/'figures/plotted_values.csv')
            self.assertEqual(frame.ippl_offset_vs_fastpm_percent.iloc[0],100.)
            self.assertEqual(frame.gadget2_offset_vs_fastpm_percent.iloc[0],200.)
            self.assertTrue(np.isnan(frame.ippl_offset_vs_fastpm_percent.iloc[1]))
            self.assertEqual(frame.ippl_P_shot_subtracted.iloc[1],-1.)
            manifest=json.loads((root/'figures/manifest.json').read_text())
            self.assertEqual(manifest['ratio_denominator'],'FastPM')


## @cond CLI_DISPATCH
if __name__=='__main__':unittest.main()
## @endcond
