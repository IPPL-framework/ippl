#!/usr/bin/env python3
## @file run_three_code_campaign.py
# @brief Build, run and analyze local and Slurm three-code comparisons of paper Figure A11.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
"""Build, run and analyze local and Slurm three-code comparisons of paper Figure A11.

Default: shared 128^3 Gaussian 1LPT ICs, z=99 to 0, 2400 plain-PM steps.
The IPPL import adapter runs the production Cosmology kernels. Existing
executables are reused. --smoke uses 16^3 and 24 steps for engineering checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType

import numpy as np
import pandas as pd

import gaussian_fixture as gaussian
from analyze_zeldovich_benchmark import cic_power
from gadget2.convert_shared_ic import convert
from validate_linear import bbks_spectrum, growth_reference

## @var Repository
# @brief Named Repository protocol/schema value; the source initializer records its exact contents.
Repository = Path(__file__).resolve().parents[3]
## @var CosmologySource
# @brief Named CosmologySource protocol/schema value; the source initializer records its exact contents.
CosmologySource = Repository / 'demos/cosmology'
## @var Schema
# @brief Named Schema protocol/schema value; the source initializer records its exact contents.
Schema = 'ippl-fastpm-gadget2-local-a11-v1'
## @var Codes
# @brief Named Codes protocol/schema value; the source initializer records its exact contents.
Codes = ('ippl', 'fastpm', 'gadget2')


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


## @brief Atomically publish a complete finite-valued JSON report.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


## @brief Wait on only this task's process group; preserve logs and exit status.
# @see cosmology_tools
#
# @param command Argument-vector command; no shell interpolation is required by the Python launcher.
# @param log Path retaining combined subprocess output and failure evidence.
# @param cwd Working directory for the launched task; relative runtime files are resolved there.
# @param timeout Finite positive subprocess timeout in seconds.
# @param environment Explicit subprocess environment; it does not by itself qualify an execution backend.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def run_command(command, log, *, cwd=None, timeout=86400, environment=None):
    """Wait on only this task's process group; preserve logs and exit status."""
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    started = time.monotonic()
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            # The process group was created above solely for this command.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
    record = {'command': command, 'cwd': str(cwd) if cwd else None,
              'log': str(Path(log).resolve()), 'exit_status': status,
              'wall_seconds': time.monotonic() - started}
    if status:
        tail = '\n'.join(Path(log).read_text(errors='replace').splitlines()[-20:])
        raise RuntimeError(f'Command exited {status}; log: {log}\n{tail}')
    return record


## @brief Report whether a selected path is a regular executable file.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return True only for a regular executable file.
def executable(path):
    return Path(path).is_file() and os.access(path, os.X_OK)


## @brief Skip each existing executable; never touch a cached binary's timestamps.
# @see cosmology_tools
#
# @param args Parsed command-line options; see main/--help and the module workflow contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def ensure_builds(args):
    """Skip each existing executable; never touch a cached binary's timestamps."""
    environment = {**os.environ, 'BUILD_JOBS': str(args.jobs), 'FASTPM_JOBS': str(args.jobs)}
    records = []
    args.build_logs.mkdir(parents=True, exist_ok=True)
    targets = ('Cosmology', 'CompareCosmologyEvolution')
    ippl = {target: args.ippl_build / 'demos/cosmology' / target for target in targets}
    cache = args.ippl_build / 'CMakeCache.txt'
    if cache.exists() and 'IPPL_USE_STANDARD_FOLDERS:BOOL=ON' in cache.read_text():
        ippl = {target: args.ippl_build / 'bin' / target for target in targets}
    missing = [target for target, path in ippl.items() if not executable(path)]
    if missing:
        if not cache.exists():
            command = ['cmake', '-S', Repository, '-B', args.ippl_build,
                       '-DCMAKE_BUILD_TYPE=Release', '-DIPPL_PLATFORMS=OPENMP',
                       '-DIPPL_ENABLE_COSMOLOGY=ON', '-DIPPL_ENABLE_FFT=ON',
                       '-DIPPL_ENABLE_SOLVERS=ON', '-DBUILD_TESTING=ON',
                       '-DKokkos_VERSION=git.5.2.0', '-DHeffte_VERSION=git.v2.4.1']
            command.extend(args.cmake_arg)
            records.append(run_command(command, args.build_logs/'ippl-configure.log', environment=environment))
        records.append(run_command(['cmake', '--build', args.ippl_build, '--target', *missing,
                                    '-j', str(args.jobs)], args.build_logs/'ippl-build.log', environment=environment))
    fastpm = args.fastpm_build / 'evolution/FastPMEvolution'
    if not executable(fastpm):
        if not executable(args.fastpm_build / 'FastPMForce'):
            records.append(run_command(['bash', CosmologySource/'reference/build_fastpm.sh', args.fastpm_build],
                                       args.build_logs/'fastpm-build.log', environment=environment))
        records.append(run_command(['bash', CosmologySource/'reference/build_fastpm_evolution.sh', args.fastpm_build],
                                   args.build_logs/'fastpm-evolution-build.log', environment=environment))
    gadget = args.gadget_build / f'Gadget2-TreePM-{args.grid}-double'
    if not executable(gadget):
        gsl = args.gsl_prefix or args.fastpm_build / 'install'
        if not (gsl/'include/gsl/gsl_math.h').exists():
            raise ValueError('GADGET needs GSL: supply --gsl-prefix or build the FastPM dependencies first')
        records.append(run_command(['bash', CosmologySource/'reference/build_gadget2.sh',
                                    args.gadget_build, str(args.grid), gsl],
                                   args.build_logs/'gadget-build.log', environment=environment))
    paths = {'cosmology': ippl['Cosmology'], 'ippl': ippl['CompareCosmologyEvolution'],
             'fastpm': fastpm, 'gadget2': gadget}
    for name, path in paths.items():
        if not executable(path):
            raise FileNotFoundError(f'{name} executable missing after build: {path}')
        print(f'Ready: {name}: {path}', flush=True)
    return paths, records


## @brief Extend the existing keyed-mode convention without relaxing its cutoff-12 API.
# @see cosmology_tools
#
# @param grid Mesh/lattice size per dimension, as specified by this module's contract.
# @param cutoff Positive spherical integer-mode ceiling, strictly below the sampling Nyquist where required.
# @param seed Unsigned 64-bit realization seed; the RNG domain/key contract is module-specific.
# @return Shared canonical phase-space DataFrame and its initial-map/provenance metadata.
def make_broadband_fixture(grid, cutoff, seed):
    """Extend the existing keyed-mode convention without relaxing its cutoff-12 API."""
    if grid < 4 or grid % 2 or not 0 < cutoff < grid / 2:
        raise ValueError('Require even grid >=4 and a positive cutoff strictly below Nyquist')
    if not 0 <= seed < 2**64:
        raise ValueError('Seed must be an unsigned 64-bit integer')
    parameters = dict(gaussian.DefaultCosmology)
    modes = np.asarray([mode for mode in itertools.product(range(-cutoff, cutoff+1), repeat=3)
                        if 0 < sum(v*v for v in mode) <= cutoff*cutoff], dtype=np.int32)
    waveNumbers = 2*math.pi/parameters['box_size'] * np.linalg.norm(modes, axis=1)
    power = bbks_spectrum(waveNumbers, parameters)
    coefficients = np.asarray([gaussian.mode_gaussian(seed, mode) for mode in modes]) * np.sqrt(
        power / parameters['box_size']**3)
    if not np.isfinite(coefficients).all() or not np.isfinite(power).all():
        raise ValueError('Nonfinite Gaussian spectrum')
    digest = hashlib.sha256(gaussian.RngDomain + modes.astype('<i4').tobytes()
                            + coefficients.astype('<c16').tobytes()).hexdigest()
    realization = gaussian.GaussianRealization(seed, cutoff, MappingProxyType(parameters),
                                              modes, coefficients, power, digest)
    a = .01
    growth, rate = growth_reference(a, parameters['Omega_m'])
    expansion = math.sqrt(parameters['Omega_m']/a**3 + 1-parameters['Omega_m'])
    displacement = gaussian.sample_displacement(realization, grid)
    eigenvalues = 1 + growth*np.linalg.eigvalsh(gaussian.sample_deformation(realization, grid))
    minimumEigenvalue = float(eigenvalues.min())
    minimumJacobian = float(np.prod(eigenvalues, axis=1).min())
    if minimumEigenvalue <= 0:
        raise ValueError('Initial 1LPT map has a nonpositive sampled eigenvalue')
    del eigenvalues
    positions = np.remainder(gaussian.lattice_positions(grid, parameters['box_size'])
                             + growth*displacement, parameters['box_size'])
    momentum = (a*a*expansion*rate*growth*displacement).astype(np.float32).astype(np.float64)
    frame = pd.DataFrame({'id': np.arange(grid**3, dtype=np.uint64)})
    frame[['x','y','z']] = positions
    frame[['px','py','pz']] = momentum
    frame['mass'] = 1.0
    metadata = realization.metadata()
    metadata.update(particle_grid=grid, particle_count=grid**3, redshift_initial=99,
                    a_initial=a, minimum_sampled_initial_map_eigenvalue=minimumEigenvalue,
                    minimum_sampled_initial_map_jacobian=minimumJacobian,
                    phase_space_sha256=gaussian.phase_space_sha256(frame),
                    momentum_contract='rounded once to float32, represented as exact float64')
    return frame, metadata


## @brief Write the declared TreePM controls using short relative paths compatible with legacy filename buffers.
# @see cosmology_tools
#
# @param root Campaign or artifact root following this module's ownership contract.
# @param grid Mesh/lattice size per dimension, as specified by this module's contract.
# @param timeout Finite positive subprocess timeout in seconds.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def gadget_parameters(root, grid, timeout):
    output = root/'runs/gadget2'
    output.mkdir(parents=True, exist_ok=False)
    outputs = root/'gadget-outputs.txt'
    outputs.write_text('1.0\n')
    softening = 168750.0/grid*.02
    parameters = {
        'InitCondFile': 'ics/shared-gadget2.bin', 'OutputDir': 'runs/gadget2/',
        'EnergyFile':'energy.txt','InfoFile':'info.txt','TimingsFile':'timings.txt',
        'CpuFile':'cpu.txt','RestartFile':'restart','SnapshotFileBase':'snapshot',
        'OutputListFilename':'gadget-outputs.txt', 'TimeLimitCPU':int(timeout)+3600,
        'ResubmitOn':0,'ResubmitCommand':'none','ICFormat':1,'SnapFormat':1,
        'ComovingIntegrationOn':1,'TypeOfTimestepCriterion':0,'OutputListOn':0,
        'PeriodicBoundariesOn':1,'TimeBegin':.01,'TimeMax':1.0,'Omega0':.31,
        'OmegaLambda':.69,'OmegaBaryon':.0487,'HubbleParam':.675,'BoxSize':168750.,
        'TimeBetSnapshot':2.0,'TimeOfFirstSnapshot':2.0,'CpuTimeBetRestartFile':36000.,
        'TimeBetStatistics':.05,'CourantFac':.15,'NumFilesPerSnapshot':1,
        'NumFilesWrittenInParallel':1,'ErrTolIntAccuracy':.025,'MaxRMSDisplacementFac':.2,
        'MaxSizeTimestep':.025,'MinSizeTimestep':0.,'ErrTolTheta':.5,
        'TypeOfOpeningCriterion':1,'ErrTolForceAcc':.005,'TreeDomainUpdateFrequency':.1,
        'DesNumNgb':33,'MaxNumNgbDeviation':2,'ArtBulkViscConst':.8,'InitGasTemp':0.,
        'MinGasTemp':0.,'PartAllocFactor':1.6,'TreeAllocFactor':.8,'BufferSize':64,
        'UnitLength_in_cm':3.085678e21,'UnitMass_in_g':1.989e43,
        'UnitVelocity_in_cm_per_s':1e5,'GravityConstantInternal':0.,'MinGasHsmlFractional':.25,
    }
    for kind in ('Gas','Halo','Disk','Bulge','Stars','Bndry'):
        parameters['Softening'+kind] = softening if kind=='Halo' else 0.
        parameters['Softening'+kind+'MaxPhys'] = softening if kind=='Halo' else 0.
    path = root/'gadget.param'
    path.write_text(''.join(f'{key} {value:.17g}\n' if isinstance(value,float)
                            else f'{key} {value}\n' for key,value in parameters.items()))
    return path


## @brief Read and validate a complete GADGET Fortran record and its expected byte count.
# @see cosmology_tools
#
# @param stream Open binary or text stream following the routine's declared record/output contract.
# @param size Expected allocation/record byte size as defined by the caller.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def read_record(stream, size):
    marker = stream.read(4)
    if len(marker)!=4 or struct.unpack('<i',marker)[0]!=size:
        raise ValueError('Unexpected GADGET format-1 record size')
    data = stream.read(size)
    trailer = stream.read(4)
    if len(data)!=size or len(trailer)!=4 or struct.unpack('<i',trailer)[0]!=size:
        raise ValueError('Truncated or mismatched GADGET record')
    return data


## @brief Validate the final GADGET format-1 cosmology, IDs and finite state, then return periodic positions.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @param grid Mesh/lattice size per dimension, as specified by this module's contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def read_gadget_snapshot(path, grid):
    count = grid**3
    with Path(path).open('rb') as stream:
        header = read_record(stream,256)
        raw = read_record(stream,count*3*4)
        velocities = np.frombuffer(read_record(stream,count*3*4), dtype='<f4')
        ids = np.frombuffer(read_record(stream,count*4), dtype='<u4')
        if stream.read(1):
            raise ValueError('Unexpected trailing GADGET snapshot blocks')
    counts = struct.unpack_from('<6i',header,0)
    totals = struct.unpack_from('<6I',header,96)
    a,z = struct.unpack_from('<2d',header,72)
    box,om,ol,h = struct.unpack_from('<4d',header,128)
    if counts!=(0,count,0,0,0,0) or totals!=counts:
        raise ValueError('GADGET final particle count differs')
    if not np.allclose([a,z,box,om,ol,h],[1,0,168750,.31,.69,.675],rtol=0,atol=1e-10):
        raise ValueError('GADGET final epoch or cosmology differs')
    if not np.array_equal(np.sort(ids),np.arange(count,dtype=np.uint32)):
        raise ValueError('GADGET IDs are incomplete or duplicated')
    positions = np.frombuffer(raw,dtype='<f4').reshape(count,3).astype(np.float64)*.001
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError('Nonfinite GADGET output')
    if np.any((positions<0)|(positions>168.75)):
        raise ValueError('GADGET positions outside the periodic box')
    return np.remainder(positions,168.75)


## @brief Validate and concatenate per-rank PM snapshots in global-ID order.
# @see cosmology_tools
#
# @param directory Artifact directory following this module's ownership/freshness contract.
# @param checkpoint Synchronized saved epoch index, with zero denoting imported initial state.
# @param grid Mesh/lattice size per dimension, as specified by this module's contract.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def read_pm_snapshot(directory, checkpoint, grid, ranks):
    paths = sorted(Path(directory).glob(f'particles_checkpoint{checkpoint:04d}_rank*.csv'))
    expected = {f'particles_checkpoint{checkpoint:04d}_rank{rank}.csv' for rank in range(ranks)}
    if {p.name for p in paths}!=expected:
        raise ValueError('Incomplete PM rank snapshot shards')
    frames = [pd.read_csv(p,dtype={'id':np.uint64},float_precision='round_trip') for p in paths]
    frame = pd.concat(frames,ignore_index=True).sort_values('id',kind='stable').reset_index(drop=True)
    if list(frame.columns)!=['id','x','y','z','px','py','pz']:
        raise ValueError('Unexpected PM snapshot columns')
    if not np.array_equal(frame.id.to_numpy(),np.arange(grid**3,dtype=np.uint64)):
        raise ValueError('Incomplete or duplicated PM particle IDs')
    values = frame[['x','y','z','px','py','pz']].to_numpy()
    if not np.isfinite(values).all() or np.any((values[:,:3]<0)|(values[:,:3]>=168.75)):
        raise ValueError('Invalid PM phase space')
    return frame, paths


## @brief Validate the differing IPPL/FastPM checkpoint schemas and synchronized epoch schedule.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @param checkpoints Saved synchronized interval count; the PM step count must be divisible by it.
# @param steps Positive PM step count; endpoints are uniform in log(a) for the imported drivers.
# @param grid Mesh/lattice size per dimension, as specified by this module's contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def validate_pm_diagnostics(path, checkpoints, steps, grid):
    table = pd.read_csv(path)
    required = {'checkpoint', 'step', 'a', 'mass_error'}
    if (not required.issubset(table.columns) or len(table) != checkpoints + 1
            or not np.isfinite(table.to_numpy(dtype=float)).all()
            or not np.array_equal(table.checkpoint, np.arange(checkpoints + 1))
            or not np.array_equal(table.step, np.arange(checkpoints + 1) * (steps // checkpoints))
            or not np.allclose(table.a, .01 * np.exp(np.arange(checkpoints + 1)
                               * math.log(100) / checkpoints), rtol=0, atol=1e-12)):
        raise ValueError('Incomplete/nonfinite PM checkpoint diagnostics or wrong epoch schedule')
    # IPPL records count through snapshot IDs; FastPM additionally stores it here.
    if 'particle_count' in table and not np.all(table.particle_count == grid**3):
        raise ValueError('PM diagnostic particle count differs')
    return table


## @brief Analyze the documented module workflow.
# @see cosmology_tools
#
# @param root Campaign or artifact root following this module's ownership contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def analyze(root):
    root = Path(root).resolve()
    report = json.loads((root/'campaign.json').read_text())
    if report.get('state') not in ('evolved','complete'):
        raise ValueError('All three solver runs must finish before analysis')
    for name, digest in report['provenance'].items():
        if sha256(name)!=digest:
            raise ValueError(f'Frozen source, executable or input changed: {name}')
    config = report['configuration']
    grid,ranks,checkpoints = (config[key] for key in ('grid','ranks','checkpoints'))
    spectra, hashes, roundtrips = {}, {}, {}
    initial = pd.read_csv(root/'ics/shared-z99.csv', dtype={'id':np.uint64}, float_precision='round_trip')
    for code in ('ippl','fastpm'):
        directory = root/'runs'/code
        frame,paths = read_pm_snapshot(directory,checkpoints,grid,ranks)
        table = validate_pm_diagnostics(directory/'checkpoints.csv', checkpoints, config['steps'], grid)
        imported, initialPaths = read_pm_snapshot(directory,0,grid,ranks)
        delta = imported[['x','y','z']].to_numpy() - initial[['x','y','z']].to_numpy()
        delta -= 168.75*np.rint(delta/168.75)
        positionError = float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))/(168.75/grid))
        momentumDelta = imported[['px','py','pz']].to_numpy() - initial[['px','py','pz']].to_numpy()
        momentumScale = np.linalg.norm(initial[['px','py','pz']].to_numpy())
        momentumError = float(np.linalg.norm(momentumDelta)/momentumScale)
        if positionError > 1e-9 or momentumError > 1e-6:
            raise ValueError(f'{code}: imported initial state differs from the shared fixture')
        roundtrips[code] = {'position_rms_cells':positionError,
                            'momentum_relative_l2':momentumError,
                            'maximum_checkpoint_mass_error':float(table.mass_error.abs().max())}
        paths.extend(initialPaths)
        del imported, delta, momentumDelta
        spectra[code] = cic_power(frame[['x','y','z']].to_numpy(),particle_grid=grid,
                                 mesh_grid=grid,box_size=168.75,cutoff=config['cutoff'])
        for path in paths+[directory/'checkpoints.csv',directory/'metadata.txt']:
            hashes[str(path)] = sha256(path)
        del frame
    snapshots = sorted((root/'runs/gadget2').glob('snapshot_*'))
    if len(snapshots)!=1:
        raise ValueError('Expected exactly one final GADGET snapshot')
    positions = read_gadget_snapshot(snapshots[0],grid)
    spectra['gadget2'] = cic_power(positions,particle_grid=grid,mesh_grid=grid,
                                 box_size=168.75,cutoff=config['cutoff'])
    hashes[str(snapshots[0])] = sha256(snapshots[0])
    data = {'schema':Schema,'configuration':config,'spectra':spectra,'input_hashes':hashes,'initial_state_roundtrips':roundtrips,
            'limitations':['One realization and one resolution, not continuum truth or a convergence qualification.',
                'GADGET TreePM short-range force and adaptive timesteps differ from the plain-PM models.',
                'CIC deconvolution does not remove all aliasing; high-k bins are characterization.']}
    write_json(root/'analysis.json',data)
    render_figure(root,data)
    report.update(state='complete',complete=True,analysis_sha256=sha256(root/'analysis.json'),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'campaign.json',report)
    return data


## @brief Plot measured powers and FastPM-relative offsets, retaining negative values in the data rather than flooring them.
# @see cosmology_tools
#
# @param root Campaign or artifact root following this module's ownership contract.
# @param data Finite numerical or serialized record data in the routine's explicit schema.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def render_figure(root,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    labels = {'ippl':'IPPL · PM','fastpm':'FastPM · plain PM','gadget2':'GADGET-2 · TreePM'}
    colors = {'ippl':'#0072B2','fastpm':'#D55E00','gadget2':'#009E73','ippl_gpu1':'#CC79A7','ippl_gpu4':'#6B51A3'}
    labels.update(ippl_gpu1='IPPL · A100 ×1',ippl_gpu4='IPPL · A100 ×4')
    plotCodes=tuple(data['spectra'])
    if not set(Codes).issubset(plotCodes) or not set(plotCodes).issubset(labels):
        raise ValueError('Unknown or missing spectrum series')
    rows=data['spectra']; config=data['configuration']
    k=np.asarray([row['k_h_per_mpc'] for row in rows['fastpm']])
    powers={code:np.asarray([row['P_shot_subtracted'] for row in rows[code]]) for code in plotCodes}
    for code in plotCodes:
        if len(rows[code])!=config['cutoff'] or not np.array_equal(k,[row['k_h_per_mpc'] for row in rows[code]]):
            raise ValueError('All three spectra must share the same Fourier shells')
    fig,(axis,ratio)=plt.subplots(2,1,figsize=(9.2,7.2),sharex=True,
        gridspec_kw={'height_ratios':[2.1,1],'hspace':.08})
    plotted=pd.DataFrame({'k_h_per_mpc':k,'shell':np.arange(1,len(k)+1)})
    for code in plotCodes:
        positive=powers[code]>0
        axis.plot(k[positive],powers[code][positive],color=colors[code],lw=1.7,
                  marker='^' if code=='ippl_gpu1' else 's' if code=='ippl_gpu4' else 'o',
                  ms=3,markevery=(2,4) if code=='ippl_gpu4' else 4,
                  linestyle=':' if code=='ippl_gpu4' else '--' if 'gpu' in code else '-',label=labels[code])
        plotted[code+'_P_shot_subtracted']=powers[code]
    for code in (code for code in plotCodes if code!='fastpm'):
        valid=(powers[code]>0)&(powers['fastpm']>0)
        offset=np.full(len(k),np.nan)
        offset[valid]=100*(powers[code][valid]/powers['fastpm'][valid]-1)
        ratio.plot(k[valid],offset[valid],color=colors[code],lw=1.5,
                   marker='^' if code=='ippl_gpu1' else 's' if code=='ippl_gpu4' else 'o',ms=3,
                   markevery=(2,4) if code=='ippl_gpu4' else 4,
                   linestyle=':' if code=='ippl_gpu4' else '--' if 'gpu' in code else '-',label=labels[code]+' / FastPM - 1')
        plotted[code+'_offset_vs_fastpm_percent']=offset
    axis.set(xscale='log',yscale='log',ylabel=r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    title='z = 0 matter power · shared Zel’dovich ICs at zᵢ = 99'
    if config['smoke']: title+=' · engineering smoke'
    axis.set_title(title,pad=12)
    ratio.set(xscale='log',xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',ylabel='Power offset [%]')
    ratio.axhline(0,color='#555555',lw=.8)
    for panel in (axis,ratio):
        panel.grid(True,which='major',color='#dddddd',lw=.6)
        panel.legend(loc='best',fontsize=9)
    note=(f"{config['grid']}³ particles; {config['grid']}³ force/analysis mesh; L=168.75 Mpc/h; "
          f"Gaussian 1LPT seed {config['seed']}.\n"
          f"Ωₘ=0.31, σ₈=0.82; PM {config['steps']} steps. Common CIC estimator, window correction and shot-noise subtraction.\n"
          f"GADGET-2 TreePM uses adaptive steps and a short-range tree. |n|≤{config['cutoff']}: characterization, not convergence.")
    fig.text(.5,.025,note,ha='center',va='bottom',fontsize=7.7,linespacing=1.3)
    fig.subplots_adjust(left=.12,right=.97,top=.92,bottom=.21)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    outputs=[]
    for ext in ('png','svg'):
        path=figures/f'figure-A11.{ext}'
        fig.savefig(path,dpi=240,metadata={'Date':None} if ext=='svg' else None)
        outputs.append(path)
    plt.close(fig)
    values=figures/'plotted_values.csv';plotted.to_csv(values,index=False,float_format='%.17g');outputs.append(values)
    write_json(figures/'manifest.json',{'schema':Schema,'analysis_sha256':sha256(root/'analysis.json'),
        'source_sha256':sha256(__file__),'outputs':{str(p):sha256(p) for p in outputs},
        'reference':'ippl-cosmology-paper/main.pdf, Figure A.11 (page 32)',
        'ratio_denominator':'FastPM','nonpositive_power':'omitted, no positive floor substituted'})


## @brief Evaluate the campaign helper in the documented module workflow.
# @see cosmology_tools
#
# @param args Parsed command-line options; see main/--help and the module workflow contract.
# @param paths Selected input/source/executable paths retained in the provenance registry.
# @param builds Recorded build commands/logs; cached executables need not be rebuilt.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def campaign(args,paths,builds):
    root=args.output
    if root is None:
        parent=CosmologySource/'results';parent.mkdir(exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix=f'a11-local-{args.grid}-',dir=parent))
    else:
        root.mkdir(parents=True,exist_ok=False)
    root=root.resolve();(root/'logs').mkdir();(root/'ics').mkdir();(root/'runs').mkdir()
    config={key:getattr(args,key) for key in ('grid','cutoff','seed','steps','checkpoints','ranks','smoke')}
    config.update(box_size=168.75,omega_m=.31,hubble=.675,sigma8=.82,n_s=.965,
                  redshift_initial=99,redshift_final=0,host=socket.gethostname())
    report={'schema':Schema,'state':'preparing','complete':False,'configuration':config,
            'started_utc':datetime.now(timezone.utc).isoformat(),'builds':builds,'runs':{},'provenance':{}}
    write_json(root/'campaign.json',report)
    print(f'Campaign directory: {root}',flush=True)
    try:
        # Conservative CSV/restart allowance; reserve separately for other user work.
        required=args.grid**3*(240+400*(args.checkpoints+1)+320)+args.reserve_gib*1024**3
        if shutil.disk_usage(root).free<required:
            raise RuntimeError(f'Insufficient free disk: need approximately {required/1024**3:.1f} GiB including reserve')
        frame,metadata=make_broadband_fixture(args.grid,args.cutoff,args.seed)
        ic=root/'ics/shared-z99.csv'
        frame.to_csv(ic,index=False,float_format='%.17g')
        del frame
        metadata.update(csv_sha256=sha256(ic),csv=str(ic))
        write_json(root/'ics/ic-manifest.json',metadata)
        convert(ic,root/'ics/shared-gadget2.bin',redshift=99,expected_sha256=metadata['csv_sha256'])
        parameter=gadget_parameters(root,args.grid,args.timeout)
        sources=[Path(__file__),Path(gaussian.__file__),CosmologySource/'python/validate_linear.py',
                 CosmologySource/'python/analyze_zeldovich_benchmark.py',
                 CosmologySource/'python/gadget2/convert_shared_ic.py',
                 CosmologySource/'CosmologySimulation.h',CosmologySource/'CosmologyPhysics.h',
                 CosmologySource/'CosmologyConfig.h',CosmologySource/'ExecutionMetadata.h',
                 CosmologySource/'tests/CompareCosmologyEvolution.cpp',
                 CosmologySource/'reference/FastPMEvolution.c',CosmologySource/'reference/build_gadget2.sh',
                 ic,parameter,root/'ics/ic-manifest.json',root/'ics/shared-gadget2.bin',*paths.values()]
        for manifest in (args.fastpm_build/'evolution/build-manifest.txt',
                         Path(str(paths['gadget2'])+'.manifest.txt')):
            if manifest.exists():sources.append(manifest)
        report['provenance']={str(p):sha256(p) for p in sources}
        report['state']='running';write_json(root/'campaign.json',report)
        environment={**os.environ,'OMP_NUM_THREADS':'1','OMP_PROC_BIND':'false',
                     'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        launcher=[args.mpiexec,*args.mpi_arg,args.numproc_flag,str(args.ranks)]
        for code in Codes:
            tail=([str(args.grid),str(args.grid),'168.75','0.31','0.01','1',str(args.steps),
                   str(args.checkpoints),str(ic),str(root/'runs'/code)]
                  if code!='gadget2' else ['gadget.param'])
            report['current_code']=code;write_json(root/'campaign.json',report)
            record=run_command([*launcher,paths[code],*tail],root/'logs'/f'{code}.log',
                               cwd=root,timeout=args.timeout,environment=environment)
            report['runs'][code]=record;write_json(root/'campaign.json',report)
        report.update(state='evolved');report.pop('current_code',None);write_json(root/'campaign.json',report)
        analyze(root)
        print(f'Figure A11: {root}/figures/figure-A11.png',flush=True)
        return root
    except BaseException as error:
        report.update(state='failed',complete=False,error=str(error))
        write_json(root/'campaign.json',report)
        raise


## @brief Select native compilation inputs independently of frozen Python/Slurm launch inputs.
# @param hashes Absolute source/compiler paths mapped to their recorded SHA256 values.
# @return Hash registry for C/C++/CUDA/CMake inputs and the selected NVCC wrapper.
# Cached binaries retain their original complete build provenance; launcher changes
# do not force recompilation when native inputs and binary/cache hashes match.
def native_source_hashes(hashes):
    return {name:digest for name,digest in hashes.items()
            if Path(name).suffix in ('.h','.hpp','.cpp','.c','.cu','.cuh','.cmake')
            or Path(name).name in ('CMakeLists.txt','nvcc_wrapper')}


## @brief Parse distinct positive MPI counts without silently dropping duplicate cases.
# @param value Single decimal count or comma-separated counts, for example 1,4.
# @return Ordered list of distinct positive rank counts.
def parse_rank_list(value):
    try:
        ranks=[int(part) for part in str(value).split(',')]
    except ValueError as error:
        raise ValueError('rank must be a positive integer or comma-separated integers') from error
    if not ranks or any(rank<1 for rank in ranks) or len(set(ranks))!=len(ranks):
        raise ValueError('rank counts must be positive and distinct')
    return ranks


## @brief Prepare exact shared ICs and submit a dependency-linked A100 build and GPU runs.
# @param args Validated CLI options; cluster is merlin6, rank counts are 1 or 4.
# @return None; submission.json retains Slurm IDs and all resolved settings.
# No science or CUDA build is executed on a login node. Invocation from a local host
# transfers the input and invokes this same controller on Merlin6 through SSH.
def submit_merlin(args):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=args.output or Path('/data/user/adelmann')/f'a11-merlin6-{stamp}'
    if not root.is_absolute():raise ValueError('Merlin --output must be an absolute remote path')
    if args.plan:
        print(json.dumps({'cluster':'merlin6','scheduler_cluster':'gmerlin6',
            'partition':'gwendolen','account':'gwendolen','rank':args.rank_list,
            'output':str(root),'shared_ic':str(args.shared_ic),
            'grid':args.grid,'steps':args.steps,'cutoff':args.cutoff},indent=2))
        return
    if not socket.gethostname().startswith('merlin'):
        if not args.shared_ic.is_file():raise FileNotFoundError(args.shared_ic)
        setup=' && '.join((shlex.join(['mkdir','-p',str(root.parent)]),shlex.join(['mkdir',str(root)]),shlex.join(['mkdir',str(root/'input')])))
        subprocess.run(['ssh','merlin6',setup],check=True)
        subprocess.run(['scp',str(args.shared_ic),f'merlin6:{root}/input/shared-z99.csv'],check=True)
        manifest=args.shared_ic.parent/'ic-manifest.json'
        if not manifest.is_file():raise ValueError('Shared IC requires sibling ic-manifest.json')
        subprocess.run(['scp',str(manifest),f'merlin6:{root}/input/ic-manifest.json'],check=True)
        command=[args.remote_python,'-B',str(Path(args.remote_root)/'demos/cosmology/python/run_three_code_campaign.py'),
            '--cluster','merlin6','--rank',','.join(map(str,args.rank_list)),
            '--shared-ic',str(root/'input/shared-z99.csv'),'--output',str(root),
            '--grid',str(args.grid),'--cutoff',str(args.cutoff),'--steps',str(args.steps),
            '--checkpoints',str(args.checkpoints),'--seed',str(args.seed),'--timeout',str(args.timeout),
            '--remote-python',args.remote_python]
        if args.ippl_build!=Repository/'build_openmp':command += ['--ippl-build',str(args.ippl_build)]
        if args.nvcc_wrapper is not None:command += ['--nvcc-wrapper',str(args.nvcc_wrapper)]
        if args.smoke:command.append('--smoke')
        if args.build_only:command.append('--build-only')
        subprocess.run(['ssh','merlin6','module unload Python/3.14.4; module load Python/3.11.11; '+shlex.join(command)],check=True)
        return
    root.mkdir(parents=True,exist_ok=True)
    if (root/'submission.json').exists():raise ValueError('Refusing duplicate submission into existing campaign')
    inputDir=root/'input';inputDir.mkdir(exist_ok=True)
    ic=inputDir/'shared-z99.csv';manifest=inputDir/'ic-manifest.json'
    if args.shared_ic.resolve()!=ic.resolve():
        shutil.copy2(args.shared_ic,ic)
        shutil.copy2(args.shared_ic.parent/'ic-manifest.json',manifest)
    metadata=json.loads(manifest.read_text())
    if metadata['csv_sha256']!=sha256(ic) or metadata['particle_grid']!=args.grid or metadata['seed']!=args.seed:
        raise ValueError('Shared IC digest, particle grid or seed differs')
    if metadata['cutoff_fundamental']!=args.cutoff or metadata['redshift_initial']!=99:
        raise ValueError('Shared IC cutoff or epoch differs')
    sources=[CosmologySource/name for name in ('CosmologySimulation.h','CosmologyPhysics.h',
        'CosmologyConfig.h','ExecutionMetadata.h','Cosmology.cpp','tests/CompareCosmologyEvolution.cpp',
        'tests/CompareCosmologyForce.cpp','merlin/a11_job.sh','python/merlin/gpu_rank.py',
        'python/run_three_code_campaign.py')]
    sources += [CosmologySource/'python'/name for name in ('gaussian_fixture.py','validate_linear.py','runtime_metadata.py','analyze_zeldovich_benchmark.py')]
    sources += [path for path in (Repository/'src').rglob('*') if path.is_file() and path.suffix in ('.h','.hpp','.cpp','.c')]
    sources += [Repository/'CMakeLists.txt',CosmologySource/'CMakeLists.txt']
    sources += list((Repository/'cmake/auto_tune/sm_80').glob('*.csv'))
    selectedBuild=getattr(args,'ippl_build',None)
    if selectedBuild is None or selectedBuild==Repository/'build_openmp':selectedBuild=Repository/'build_a11_a100'
    if not selectedBuild.is_absolute():raise ValueError('Merlin build directory must be absolute')
    wrapper=getattr(args,'nvcc_wrapper',None) or Repository/'build_a100/_deps/kokkos-src/bin/nvcc_wrapper'
    if not executable(wrapper):raise FileNotFoundError(f'Expected cached Kokkos CUDA compiler wrapper: {wrapper}')
    sources.append(wrapper)
    config={'schema':'ippl-a11-merlin6-v1','cluster':'merlin6','scheduler_cluster':'gmerlin6',
        'grid':args.grid,'cutoff':args.cutoff,'seed':args.seed,'steps':args.steps,
        'checkpoints':args.checkpoints,'smoke':args.smoke,'shared_ic':str(ic),
        'shared_ic_sha256':sha256(ic),'source_root':str(Repository),'python':args.remote_python,
        'build_dir':str(selectedBuild),'nvcc_wrapper':str(wrapper),'timeout':args.timeout,
        'mpi_environment':{'OMPI_MCA_pml':'ucx','OMPI_MCA_pml_ucx_tls':'any','OMPI_MCA_pml_ucx_devices':'any','UCX_TLS':'sm,self,cuda_copy,cuda_ipc'},
        'source_sha256':{str(path):sha256(path) for path in sources}}
    write_json(root/'configuration.json',config)
    jobs={}
    def submit(action,rank,dependency=None):
        command=['sbatch','--parsable','-M','gmerlin6','-A','gwendolen','-p','gwendolen',
            '--nodes=1',f'--ntasks={rank}',f'--cpus-per-task={4 if action=="build" else 1}',
            f'--gres=gpu:{1 if action=="build" else rank}','--mem=32G','--time=04:00:00',
            f'--job-name=a11-{action}-{rank}',f'--output={root}/slurm-{action}-{rank}-%j.log']
        if dependency:command.append(f'--dependency=afterok:{dependency}')
        command += [str(CosmologySource/'merlin/a11_job.sh'),action,str(root),str(rank)]
        result=subprocess.run(command,text=True,capture_output=True,check=True)
        identifier=result.stdout.strip().split(';')[0]
        if not identifier.isdecimal():raise RuntimeError(f'Unexpected sbatch result: {result.stdout}')
        return {'job_id':identifier,'command':command}
    # Build job also verifies a cached binary against its retained source manifest.
    jobs['build']=submit('build',1)
    write_json(root/'submission.json',{'configuration':config,'jobs':jobs})
    if not args.build_only:
        for rank in args.rank_list:
            jobs[f'gpu{rank}']=submit('run',rank,jobs['build']['job_id'])
            write_json(root/'submission.json',{'configuration':config,'jobs':jobs})
    print(json.dumps({'output':str(root),'jobs':jobs},indent=2),flush=True)


## @brief Validate A100 bindings and exact imported state, then measure the common CIC spectrum.
# @param root Completed rank-result directory, containing configuration.json and GPU snapshots.
# @return None; gpu-analysis.json retains spectra, source/input/output hashes and binding evidence.
def analyze_gpu(root):
    root=Path(root);config=json.loads((root/'configuration.json').read_text());rank=config['ranks']
    for path,digest in config['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError(f'GPU frozen source changed: {path}')
    if sha256(config['shared_ic'])!=config['shared_ic_sha256']:raise ValueError('GPU IC changed')
    build=json.loads((root.parent/'build-manifest.json').read_text())
    for path,digest in build['artifacts'].items():
        if sha256(path)!=digest:raise ValueError(f'GPU build artifact changed: {path}')
    allocation=pd.read_csv(root/'allocated-gpus.csv',header=None,skipinitialspace=True)
    if len(allocation)!=rank or allocation.shape[1]!=4 or not allocation[0].str.contains('A100').all() or not (allocation[3]=='Disabled').all():
        raise ValueError('Require allocated full A100 GPUs with disabled MIG')
    bindings=[json.loads(line.split('GPU_BINDING ',1)[1]) for line in (root/'solver.log').read_text().splitlines() if 'GPU_BINDING ' in line]
    if (len(bindings)!=rank or {row['rank'] for row in bindings}!=set(range(rank))
        or len({row['pci'] for row in bindings})!=rank or len({row['host'] for row in bindings})!=1
        or any(row['world_size']!=rank or row['local_size']!=rank or row['runtime_device_count']!=1
               or row['visible_device_ordinal']!=0 for row in bindings)):
        raise ValueError('GPU bindings do not prove one distinct physical GPU per rank')
    def pci(value):return tuple(int(part,16) for part in value.replace('.',':').split(':'))
    if {pci(row['pci']) for row in bindings}!={pci(value) for value in allocation[2]}:
        raise ValueError('Bound GPUs differ from scheduler allocation')
    metadata=(root/'run/metadata.txt').read_text()
    metadataFields=dict(line.split('=',1) for line in metadata.splitlines() if '=' in line)
    if metadataFields.get('execution_space')!='Cuda' or metadataFields.get('memory_space') not in ('Cuda','CudaSpace'):
        raise ValueError('Expected CUDA execution and memory spaces')
    frame,paths=read_pm_snapshot(root/'run',config['checkpoints'],config['grid'],rank)
    imported,initialPaths=read_pm_snapshot(root/'run',0,config['grid'],rank)
    initial=pd.read_csv(config['shared_ic'],dtype={'id':np.uint64},float_precision='round_trip')
    delta=imported[['x','y','z']].to_numpy()-initial[['x','y','z']].to_numpy();delta-=168.75*np.rint(delta/168.75)
    positionError=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))/(168.75/config['grid']))
    momentumError=float(np.linalg.norm(imported[['px','py','pz']].to_numpy()-initial[['px','py','pz']].to_numpy())/np.linalg.norm(initial[['px','py','pz']].to_numpy()))
    if positionError>1e-9 or momentumError>1e-6:raise ValueError('GPU imported state differs from exact shared fixture')
    table=validate_pm_diagnostics(root/'run/checkpoints.csv',config['checkpoints'],config['steps'],config['grid'])
    if table.mass_error.abs().max()>2e-12:raise ValueError('GPU mass conservation failed')
    spectra=cic_power(frame[['x','y','z']].to_numpy(),particle_grid=config['grid'],mesh_grid=config['grid'],box_size=168.75,cutoff=config['cutoff'])
    artifacts=paths+initialPaths+[root/'run/checkpoints.csv',root/'run/metadata.txt',root/'solver.log',root/'allocated-gpus.csv',root/'configuration.json']
    write_json(root/'gpu-analysis.json',{'schema':'ippl-a11-gpu-analysis-v1','complete':True,'configuration':config,
        'spectra':spectra,'bindings':bindings,'source_sha256':config['source_sha256'],
        'artifacts':{str(path):sha256(path) for path in artifacts},'build':build,
        'initial_state_roundtrip':{'position_rms_cells':positionError,'momentum_relative_l2':momentumError},
        'maximum_checkpoint_mass_error':float(table.mass_error.abs().max())})


## @brief Extend verified local spectra with completed GPU spectra in a fresh figure directory.
# @param baseline Completed local three-code campaign directory; recorded analysis hash must match.
# @param gpu_results Downloaded GPU rank directories, each with complete gpu-analysis.json.
# @param output New comparison directory; original baseline evidence is preserved.
# @return None; analysis.json and Figure A11 retain all five measured series and provenance.
def extend_figure(baseline,gpu_results,output):
    baseline=Path(baseline);report=json.loads((baseline/'campaign.json').read_text())
    if not report.get('complete') or sha256(baseline/'analysis.json')!=report['analysis_sha256']:
        raise ValueError('Baseline is incomplete or its analysis changed')
    data=json.loads((baseline/'analysis.json').read_text())
    inputs={str(baseline/'analysis.json'):sha256(baseline/'analysis.json'),str(baseline/'campaign.json'):sha256(baseline/'campaign.json')}
    for path,digest in data['input_hashes'].items():
        if sha256(path)!=digest:raise ValueError(f'Baseline output changed: {path}')
    icDigest=report['provenance'][str((baseline/'ics/shared-z99.csv').resolve())]
    for directory in gpu_results:
        directory=Path(directory);path=directory/'gpu-analysis.json';gpu=json.loads(path.read_text())
        config=gpu['configuration'];rank=config['ranks'];code=f'ippl_gpu{rank}'
        if not gpu.get('complete') or rank not in (1,4) or code in data['spectra']:
            raise ValueError('Invalid, duplicate or incomplete GPU result')
        if config['shared_ic_sha256']!=icDigest or any(config[key]!=data['configuration'][key] for key in ('grid','cutoff','steps','checkpoints','seed','smoke')):
            raise ValueError('GPU IC or campaign settings differ from baseline')
        for remote,digest in gpu['artifacts'].items():
            # Download the complete rank directory without rewriting recorded remote paths.
            local=directory/Path(remote).relative_to(Path(config['result_root']))
            if sha256(local)!=digest:raise ValueError(f'Downloaded GPU artifact changed: {local}')
        data['spectra'][code]=gpu['spectra'];inputs[str(path.resolve())]=sha256(path)
        data.setdefault('gpu_results',{})[code]=gpu
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    data['comparison_inputs']=inputs
    write_json(output/'analysis.json',data);render_figure(output,data)
    print(f'Extended Figure A11: {output}/figures/figure-A11.png')


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='New campaign directory; default unique directory under cosmology/results')
    parser.add_argument('--analyze-only',type=Path,metavar='CAMPAIGN',help='Verify and replot an already evolved campaign')
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--plan',action='store_true',help='Print resolved settings without building or running')
    parser.add_argument('--smoke',action='store_true',help='16^3, cutoff 6, 24 steps; engineering check only')
    parser.add_argument('--grid',type=int,default=128)
    parser.add_argument('--cutoff',type=int,default=48)
    parser.add_argument('--steps',type=int,default=2400)
    parser.add_argument('--checkpoints',type=int,default=1,help='Saved intervals, default initial/final only to limit disk use')
    parser.add_argument('--seed',type=int,default=20261003)
    parser.add_argument('--rank','--ranks',dest='rank',default='1',help='MPI ranks; comma-separated 1,4 submits both Merlin GPU cases; default 1')
    parser.add_argument('--cluster',choices=['merlin6'],help='Submit A100 jobs via Slurm on Merlin6; omitted means local')
    parser.add_argument('--nvcc-wrapper',type=Path,help='Cluster-local cached Kokkos nvcc_wrapper; default remote build_a100/_deps/kokkos-src/bin/nvcc_wrapper')
    parser.add_argument('--shared-ic',type=Path,help='Existing exact shared-z99.csv for Merlin; use the completed local campaign IC')
    parser.add_argument('--remote-root',default='/data/user/adelmann/ippl')
    parser.add_argument('--remote-python',default='/data/user/adelmann/cosmology-cpu-login-20261004/python/bin/python')
    parser.add_argument('--extend-figure',type=Path,metavar='BASELINE',help='Verified complete local baseline to extend without changing its evidence')
    parser.add_argument('--gpu-result',type=Path,action='append',default=[],help='Downloaded completed Merlin rank result; repeat for 1 and 4 GPUs')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=86400,help='Per-solver limit in seconds')
    parser.add_argument('--reserve-gib',type=float,default=2)
    parser.add_argument('--mpiexec',default='mpiexec')
    parser.add_argument('--numproc-flag',default='-n')
    parser.add_argument('--mpi-arg',action='append',default=[])
    parser.add_argument('--cmake-arg',action='append',default=[],help='Fresh IPPL configure argument; repeat, using --cmake-arg=-D...')
    parser.add_argument('--ippl-build',type=Path,default=Repository/'build_openmp')
    parser.add_argument('--fastpm-build',type=Path,default=Repository/'build_fastpm')
    parser.add_argument('--gadget-build',type=Path,default=Repository/'build_gadget2')
    parser.add_argument('--gsl-prefix',type=Path)
    args=parser.parse_args()
    try:
        args.rank_list=parse_rank_list(args.rank)
    except ValueError as error:
        parser.error(str(error))
    args.ranks=args.rank_list[0]
    if args.extend_figure:
        if args.output is None or not args.gpu_result:parser.error('Figure extension needs --output NEW_DIR and --gpu-result')
        extend_figure(args.extend_figure,args.gpu_result,args.output)
        return 0
    if args.analyze_only:
        analyze(args.analyze_only)
        return 0
    if args.smoke:args.grid,args.cutoff,args.steps=16,6,24
    if (args.grid<4 or args.grid%2 or not 0<args.cutoff<args.grid/2 or args.grid>128
            or args.steps<1 or args.checkpoints<1 or args.steps%args.checkpoints
            or args.ranks<1 or args.jobs<1 or not math.isfinite(args.timeout) or args.timeout<=0
            or not math.isfinite(args.reserve_gib) or args.reserve_gib<0 or not 0<=args.seed<2**64):
        parser.error('Invalid grid/cutoff/schedule/ranks/jobs/time/disk/seed; this local runner supports grids <=128')
    if args.cluster:
        if args.shared_ic is None:parser.error('--cluster merlin6 requires --shared-ic from the completed baseline')
        if any(rank not in (1,4) for rank in args.rank_list):parser.error('Merlin supports rank 1 or 4')
        if args.grid!=128 and not args.smoke:parser.error('Merlin science campaign is 128 cubed; use --smoke for an engineering check')
        submit_merlin(args)
        return 0
    if len(args.rank_list)!=1:parser.error('Local runs accept one --rank; Merlin accepts comma-separated ranks')
    for key in ('output','ippl_build','fastpm_build','gadget_build','gsl_prefix'):
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).expanduser().resolve())
    args.build_logs=args.gadget_build/'controller-logs'
    if args.plan:
        print(json.dumps({key:str(value) if isinstance(value,Path) else value
                          for key,value in vars(args).items()},indent=2))
        return 0
    paths,records=ensure_builds(args)
    if not args.build_only:campaign(args,paths,records)
    return 0


## @cond CLI_DISPATCH
if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,RuntimeError,OSError,subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
        print(f'Campaign error: {error}',file=sys.stderr)
        sys.exit(1)
## @endcond
