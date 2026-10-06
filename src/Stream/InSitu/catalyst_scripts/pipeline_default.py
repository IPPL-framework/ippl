r"""! \file pipeline_default.py
\brief Main ParaView Catalyst pipeline: live visualization, VTK extracts, PNG extracts, and steering.
\details Discovers channel proxies, wires optional extractors, updates live views,
and forwards/fetches steerable parameters between the simulation and ParaView.

PNG extracts replicate the rendering of the former per-channel extractor scripts
(catalyst_extractors/png_ext_particle.py, png_ext_sfield.py and
png_ext_vfield.py): every channel - or every array of a mesh channel - gets its
own dedicated render view, styled and extracted exactly like those scripts did,
written to 'data_png_extracts_<experiment_name>'.
"""

# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html
# BUG: if pv client is opened after Simulation has completed first execute
# the created PNG extractor show up in the clients pipeline browser
# Iam not sure how to avoid this. This will break the View needed for
# the extractor to work properly and png extaction starrts failing ca
# 3 timesteps after the client was opened.
########################################################
########################################################

import paraview
from paraview.simple import *
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 12
import paraview.catalyst
from paraview import catalyst

import paraview.simple as pvs
from paraview.simple import (
    CreateSteerableParameters,
    PVTrivialProducer,
    Show,
    MergeBlocks,
    ExtractBlock,
    CellDatatoPointData,
    Glyph,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetTransferFunction2D,
    GetScalarBar,
    ColorBy,
    CreateView,
    SetActiveView,
    GetMaterialLibrary,
    ResampleToImage,
    Threshold,
    # LoadPlugin,
    # ExtractSubset,
    # AdaptiveResampleToImage,
)
from paraview import servermanager
from paraview import servermanager as sm
from paraview import print_info

import argparse
import sys
import time
import os

import string

sys.path.append(os.path.dirname(__file__))


from catalystSubroutines import (
    print_proxy_overview,
    get_global_spatial_bounds,
    get_global_range,
    nice_bounds,
    nice_bounds_sym,
    auto_camera_from_bounds,
    compute_bounding_box_scale
)
#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


from paraview.simple import CreateExtractor

# ------------------------------------------------------------------------------
# Extractor factory functions
# ------------------------------------------------------------------------------
# IMPORTANT: Match writer type to Conduit data structure:
#   - Conduit "mesh" (uniform/structured) → VTPD (homogeneous partitions)
#   - Conduit "multimesh" (heterogeneous blocks) → VTM (MultiBlock)
# Using wrong writer causes: "Can not execute ... without output ports" error
# ------------------------------------------------------------------------------

def create_VTPD_extractor(name, object, fr = 10):
    """Create a VTPD extractor for simple mesh data (uniform/structured grids).

    Use for: scalar fields, vector fields (Conduit type: "mesh")
    Output: .vtpd files (XML Partitioned Dataset)
    """
    safe_name = name.replace("/", "_")  # nested channels are named parent/block
    vTPD = CreateExtractor('VTPD', object, registrationName='VTPD_'+ safe_name)
    vTPD.Trigger.Frequency = fr
    vTPD.Writer.FileName = 'ippl_'+safe_name+'_{timestep:06d}.vtpd'
    return vTPD

def create_VTM_extractor(name, object, fr = 10):
    """Create a VTPC extractor for multimesh particle data (Partitioned Dataset Collection).

    Use for: particle data with multiple block types (Conduit type: "multimesh")
    Output: .vtpc files (XML Partitioned Dataset Collection)
    Note: Conduit multimesh → vtkPartitionedDataSetCollection (not vtkMultiBlockDataSet)
    """
    # Use VTPC (Partitioned Dataset Collection) not VTM (MultiBlock)
    safe_name = name.replace("/", "_")  # nested channels are named parent/block
    vTPC = CreateExtractor('VTPC', object, registrationName='VTPC_'+ safe_name)
    vTPC.Trigger.Frequency = fr
    vTPC.Writer.FileName = 'ippl_'+safe_name+'_{timestep:06d}.vtpc'
    return vTPC


_CATALYST_INFO_LEVEL = 4


def _log(msg):
    """Logs a message from the Catalyst script."""

    controller = sm.vtkProcessModule.GetProcessModule().GetGlobalController()
    rank = 0
    if controller:
        rank = controller.GetLocalProcessId()
        msg =  "[rank " + str(rank) + "]:  " + msg


    if verbosity >= _CATALYST_INFO_LEVEL:
        print(msg)

def print_info_(s, level=_CATALYST_INFO_LEVEL):
    global verbosity
    if verbosity >= max(level, _CATALYST_INFO_LEVEL):
        print_info(s)




#### disable automatic camera rest on 'Show'
# paraview.simple._DisableFirstRenderCameraReset()
colorPalette = GetSettingsProxy('ColorPalette')


colorPalette.BackgroundColorMode = 'Gradient'

colorPalette.Background = [0.0, 0.0, 0.0]
# colorPalette.Background = [0.0, 0.8, 1.0]
# colorPalette.Background = [0.3333333333333333, 1.0, 1.0]


colorPalette.Background2 = [0.9, 0.9, 0.9]
# colorPalette.Background2 = [0.97, 0.97, 0.97]

# # Properties modified on colorPalette
colorPalette.Foreground = [0.8, 0.0, 0.0]
colorPalette.Edges = [1.0, 0.0, 0.0]
colorPalette.Surface = [1.0, 0.0, 0.0]


# ----------------------------------------------------------------
# Parse arguments received via conduit node
# ----------------------------------------------------------------
arg_list = paraview.catalyst.get_args()
# print_info_(f"Arguments received: {arg_list}")
parser = argparse.ArgumentParser()
parser.add_argument("--channel_names", nargs="*",
                     help="Pass All Channel Names for which we need to update the privial producer each round")
parser.add_argument("--channel_types", nargs="*",
                     help="Conduit type for each channel (mesh or multimesh)")
parser.add_argument("--steer_channel_names", nargs="*",
                     help="Pass All Channel Names for Steering scalar parameters")
parser.add_argument("--channel_ghosts", nargs="*", type=int, default=[],
                    help="Ghost layers included per channel export (informational; ghost handling in the pipelines detects vtkGhostType itself)")

parser.add_argument("--verbosity", type=int, default=0, help="Communicate the Catalyst output level from the simulation")
parser.add_argument("--VTKextract", default="OFF", help="Enable the VTK extracts of all incoming channels")
parser.add_argument("--png",        default="OFF", help="Enable PNG screenshot extracts of the render view")
parser.add_argument("--extracts_output_dir", default="",
                    help="Extract output directory passed from the simulation (overrides the experiment-name default)")
parser.add_argument("--live",       default="OFF", help="Enable options.CatalystLive")
parser.add_argument("--steer",      default="OFF", help="Enable steering from catalyst python side")
parser.add_argument("--show_forward_channels", default="OFF", help="Show forward steerable channels in the GUI (PVTrivialProducer)")
parser.add_argument("--experiment_name", default="_", help="Needed to correctly for safe folder.")

parsed = parser.parse_args(arg_list)

exp_string = parsed.experiment_name
verbosity = parsed.verbosity
png_enabled = parsed.png == "ON"


# ------------------------------------------------------------------------------
print_info_("==========================================================0"[0:55]+"|")
print_info_("======= EXECUTING catalyst_pipeline GLOBAL SCOPE =========0"[0:55]+"|")
print_info_("==========================================================0"[0:55]+"|")
# ------------------------------------------------------------------------------



print_info_(f"Parsed steer_channel_names:     {parsed.steer_channel_names}")
print_info_(f"Parsed channel_names:           {parsed.channel_names}")
print_info_(f"Parsed channel_types:           {parsed.channel_types}")
print_info_(f"Parsed channel_ghosts:          {parsed.channel_ghosts}")
print_info_(f"Parsed verbosity level:         {parsed.verbosity}")
print_info_(f"Parsed VTK extract options:     {parsed.VTKextract}")
print_info_(f"Parsed PNG extract option:      {parsed.png}")
print_info_(f"Parsed extracts output dir:     {parsed.extracts_output_dir}")
print_info_(f"Parsed steering option:         {parsed.steer}")




# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
options = catalyst.Options()

if parsed.live == "ON":
    options.EnableCatalystLive = 1
else:
    options.EnableCatalystLive = 0



options.GlobalTrigger = 'Time Step'
options.CatalystLiveTrigger = 'Time Step'
options.CatalystLiveURL = 'localhost:22222' #is also default
if png_enabled:
    # The former png_ext_*.py scripts wrote their PNG extracts here; keep the
    # same location so output remains comparable across the redesign.
    options.ExtractsOutputDirectory = 'data_png_extracts_' + exp_string
elif parsed.extracts_output_dir:
    # Directory passed from the simulation (below IPPL_CATALYST_OUTPUT_DIR)
    options.ExtractsOutputDirectory = parsed.extracts_output_dir
else:
    options.ExtractsOutputDirectory = 'data_vtk_extracts_' + exp_string


# Steering requires a Live connection: the backward-channel senders only
# make sense when a client can drive the parameters.
steer_enabled = parsed.live == "ON" and parsed.steer == "ON"
if parsed.steer == "ON" and parsed.live != "ON":
    print_info_("Steering requested (--steer ON) but Live is OFF: steering requires Live and stays disabled.")

# ------------------------------------------------------------------------------
# Proactively remove the forward 0D mesh source from the GUI if it was auto-created
# ------------------------------------------------------------------------------
if steer_enabled and parsed.show_forward_channels == "OFF":
    try:
        src = paraview.simple.FindSource("steerable_channel_0D_mesh")
        if src is not None:
            print_info_("Hiding auto-created steerable_channel_0D_mesh from GUI")
            paraview.simple.Delete(src)
    except Exception as e:
        print_info_(f"Could not hide steerable_channel_0D_mesh: {e}")

    # Additionally hide any per-array forward meshes inferred from steer_channel_names
    try:
        if parsed.steer_channel_names:
            for sname in parsed.steer_channel_names:
                # Deduce forward mesh registration name for std::vector-based steerables
                # Labels are normalized on the C++ side to start with 'array:' for vectors.
                reg_name = None
                if isinstance(sname, str) and sname.startswith("array:"):
                    # namespace prefix before first '.' (e.g., 'array:LinMaps.time' -> 'LinMaps')
                    # ns = sname[6:]
                    ns = sname

                    dot = ns.find('.')
                    if dot != -1:
                        ns = ns[:dot]
                    reg_name = f"steerable_channel_1D_mesh_{ns}"

                if reg_name:
                    src_arr = paraview.simple.FindSource(reg_name)
                    if src_arr is not None:
                                print_info_(f"Hiding per-array forward mesh '{reg_name}' from GUI")
                                paraview.simple.Delete(src_arr)
    except Exception as e:
        print_info_(f"Error while hiding inferred per-array forward meshes: {e}")



# Global dictionaries to store proxies, filters etc
_extractors = {}
_sources = {}
_filters = {}
# _shows={}


# Build a (name, type) list.  If --channel_types was not passed (e.g. old
# C++ side), fall back to name-based heuristics for backward compatibility.
_channel_types = parsed.channel_types or []
_channels = list(zip(parsed.channel_names or [], _channel_types))
if not _channel_types:
    _channels = []
    for cname in (parsed.channel_names or []):
        if "particles" in cname:
            _channels.append((cname, "multimesh"))
        else:
            _channels.append((cname, "mesh"))

# Live view for Catalyst Live connections. Plain default styling, exactly like
# the live branch of the former pipeline_default.py: the fancy per-channel
# rendering lives in the dedicated PNG views below.
live_view = None
if options.EnableCatalystLive:
    live_view = CreateView('RenderView')
    SetActiveView(live_view)


# ------------------------------------------------------------------------------
# Particle channel PNG machinery (recipe: catalyst_extractors/png_ext_particle.py)
# ------------------------------------------------------------------------------

_BUILTIN_PARTICLE_ATTRIBUTES = {
    "position",
    "particleid",
    "particleids",
    "rankid",
    "rankids",
}


def _normalized_attribute_name(name):
    """Normalize spelling variants such as ParticleIDs and particle_ids."""
    return "".join(character for character in name.lower() if character.isalnum())


def _first_custom_particle_attribute(point_data_info):
    """Return (name, component count) for the first non-built-in point array."""
    if point_data_info is None:
        return None, 0

    for index in range(point_data_info.GetNumberOfArrays()):
        array_info = point_data_info.GetArrayInformation(index)
        if array_info is None:
            continue
        name = array_info.GetName()
        if not name or _normalized_attribute_name(name) in _BUILTIN_PARTICLE_ATTRIBUTES:
            continue
        components = array_info.GetNumberOfComponents()
        if components > 0:
            return name, components

    return None, 0


def _configure_particle_coloring(state, point_data_info):
    """Select the first custom attribute, or configure solid red coloring.

    Ported from png_ext_particle.py; coloring state is kept per channel.
    """
    array_name, components = _first_custom_particle_attribute(point_data_info)
    if (state['color_initialized']
            and array_name == state['color_array_name']
            and components == state['color_components']):
        return

    if state['color_lut'] is not None:
        try:
            GetScalarBar(state['color_lut'], state['view']).Visibility = 0
        except Exception:
            pass

    state['color_array_name'] = array_name
    state['color_components'] = components
    state['color_lut'] = None
    state['color_initialized'] = True

    display = state['display']
    if array_name is None:
        display.ColorArrayName = ['POINTS', '']
        display.AmbientColor = [1.0, 0.0, 0.0]
        display.DiffuseColor = [1.0, 0.0, 0.0]
        print_info_("No custom particle attribute found; using solid red coloring.")
        return

    color_spec = ("POINTS", array_name)
    component_title = ""
    if components > 1:
        color_spec = ("POINTS", array_name, "Magnitude")
        component_title = "Magnitude"

    ColorBy(display, color_spec)
    lut = GetColorTransferFunction(array_name)
    color_bar = GetScalarBar(lut, state['view'])
    color_bar.Title = array_name
    color_bar.ComponentTitle = component_title
    color_bar.Visibility = 1
    display.SetScalarBarVisibility(state['view'], True)
    state['color_lut'] = lut

    mode = "magnitude" if components > 1 else "scalar values"
    print_info_(f"Coloring particles by {mode} of custom attribute '{array_name}'.")


def _setup_particle_png_channel(cname, producer, particles, helper):
    """Dedicated PNG view for one particle channel (png_ext_particle.py recipe).

    Point-gaussian bunch with a data axes grid, solid red until the first
    execute discovers custom attributes, yellow outline helper box, adaptive
    camera and a PNG extractor writing '<label>_Particles_NNNNNN.png'.
    """
    label = cname.replace("/", "_")

    view = CreateView('RenderView', registrationName=f"View_{label}")
    view.ViewSize = [2000, 1500]
    view.AxesGrid = 'GridAxes3DActor'
    view.HiddenLineRemoval = 1
    view.StereoType = 'Crystal Eyes'
    view.BackEnd = 'OSPRay raycaster'
    view.OSPRayMaterialLibrary = GetMaterialLibrary()
    view.AxesGrid.Visibility = 1
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = 'Gradient'
    SetActiveView(view)

    # Initial camera from the (still empty) producer bounds; re-framed on
    # every catalyst_execute once data has arrived.
    auto_camera_from_bounds(view,
                            get_global_spatial_bounds(producer.GetDataInformation().GetBounds()))

    particlesDisplay = Show(particles, view, 'UnstructuredGridRepresentation')
    particlesDisplay.Representation = 'Point Gaussian'
    particlesDisplay.DataAxesGrid = 'GridAxesRepresentation'
    particlesDisplay.SelectInputVectors = ['POINTS', 'position']
    # Point-array metadata may not be available while the Catalyst script is
    # first loaded. Start with the guaranteed fallback and discover custom
    # attributes in catalyst_execute after the producer is updated.
    particlesDisplay.ColorArrayName = ['POINTS', '']
    particlesDisplay.AmbientColor = [1.0, 0.0, 0.0]
    particlesDisplay.DiffuseColor = [1.0, 0.0, 0.0]

    helperDisplay = Show(helper, view, 'GeometryRepresentation')
    helperDisplay.Representation = 'Outline'
    helperDisplay.AmbientColor = [1.0, 1.0, 0.0]
    helperDisplay.DiffuseColor = [1.0, 1.0, 0.0]
    helperDisplay.LineWidth = 2.0
    helperDisplay.Opacity = 0.1
    # Solid Color mode: no array, just DiffuseColor
    helperDisplay.ColorArrayName = ['POINTS', '']

    pNG1 = CreateExtractor('PNG', view, registrationName='PNG_' + label)
    pNG1.Trigger = 'Time Step'
    pNG1.Trigger.Frequency = 1
    pNG1.Writer.FileName = label + '_Particles_{timestep:06d}{camera}.png'
    pNG1.Writer.ImageResolution = [2000, 1500]
    pNG1.Writer.TransparentBackground = 0
    pNG1.Writer.Format = 'PNG'

    _png_particle_channels[cname] = {
        'producer': producer,
        'particles': particles,
        'view': view,
        'display': particlesDisplay,
        'color_array_name': None,
        'color_components': 0,
        'color_lut': None,
        'color_initialized': False,
    }

    if live_view is not None:
        SetActiveView(live_view)


# ------------------------------------------------------------------------------
# Mesh channel PNG machinery (recipes: png_ext_sfield.py / png_ext_vfield.py)
# ------------------------------------------------------------------------------

def _setup_sfield_view(cname, mesh_src, name, ghost_info):
    """Dedicated PNG view for one scalar mesh array (png_ext_sfield.py recipe).

    Ghost cutting (Threshold) and grid restoration (ResampleToImage), volume
    rendering with the blue-white-red transfer function and the 'zero wall'
    opacity curve, camera framed on the original domain bounds.
    """
    label = cname.replace("/", "_")

    info = mesh_src.GetDataInformation()
    local_bounds = info.GetBounds()
    local_extent = info.GetExtent()
    global_bounds = get_global_spatial_bounds(local_bounds)

    # Derive uniform spacing from local bounds and local extent.
    # For ImageData, bounds length ≈ (num_points - 1) * spacing.
    nx = (local_extent[1] - local_extent[0] + 1)
    ny = (local_extent[3] - local_extent[2] + 1)
    nz = (local_extent[5] - local_extent[4] + 1)

    lx = (local_bounds[1] - local_bounds[0])
    ly = (local_bounds[3] - local_bounds[2])
    lz = (local_bounds[5] - local_bounds[4])

    spacing_x = lx / max(nx - 1, 1)
    spacing_y = ly / max(ny - 1, 1)
    spacing_z = lz / max(nz - 1, 1)

    dx = max(spacing_x, 1e-12)
    dy = max(spacing_y, 1e-12)
    dz = max(spacing_z, 1e-12)

    # Keep a copy of the original global bounds for camera framing to avoid
    # visual "zoom" when the ghost ring is trimmed from the resampling bounds.
    original_global_bounds = tuple(global_bounds)

    # Optionally trim one layer to exclude the outer ghost ring from resampling
    cut_layers = 1 if ghost_info else 0
    extra_dims = 0
    global_bounds = (
        global_bounds[0] + cut_layers * dx,
        global_bounds[1] - cut_layers * dx,
        global_bounds[2] + cut_layers * dy,
        global_bounds[3] - cut_layers * dy,
        global_bounds[4] + cut_layers * dz,
        global_bounds[5] - cut_layers * dz,
    )
    dim_x = int(round((global_bounds[1] - global_bounds[0]) / dx)) + extra_dims
    dim_y = int(round((global_bounds[3] - global_bounds[2]) / dy)) + extra_dims
    dim_z = int(round((global_bounds[5] - global_bounds[4]) / dz)) + extra_dims
    global_extent = [max(dim_x, 1), max(dim_y, 1), max(dim_z, 1)]

    print_info_(f"Local spacing: dx={spacing_x}, dy={spacing_y}, dz={spacing_z}")
    print_info_(f"Global bounds (after modification)): {global_bounds}")
    print_info_(f"Global extent (after modification)): {global_extent}")

    # Ghost Handling & Restore Grid (Threshold -> Resample)
    if ghost_info:
        # Use Threshold to physically remove ghost cells; this converts
        # ImageData -> UnstructuredGrid.
        print_info_("vtkGhostType found; processing ghosts...", level=1)
        ippl_scalar_t = Threshold(registrationName=f"{cname}.{name}.RemoveGhosts",
                                  Input=mesh_src)
        ippl_scalar_t.Scalars = ['CELLS', 'vtkGhostType']
        ippl_scalar_t.AllScalars = 0
        ippl_scalar_t.ThresholdMethod = 'Between'
        ippl_scalar_t.LowerThreshold = 0.0
        ippl_scalar_t.UpperThreshold = 0.0

        # Convert BACK to Uniform Grid using ResampleToImage; this restores
        # the high-quality GPU Volume Mapper.
        print_info_("Resampling back to ImageData...", level=1)
        ippl_resampled = ResampleToImage(registrationName=f"{cname}.{name}.ResampleToGrid",
                                         Input=ippl_scalar_t)
        ippl_resampled.UseInputBounds = 1
        ippl_resampled.SamplingDimensions = global_extent
        _filters[f"{cname}.{name}.ghosts"] = ippl_scalar_t
    else:
        print_info_("vtkGhostType not found; skipping ghost filtering", level=1)
        ippl_resampled = ResampleToImage(registrationName=f"{cname}.{name}.ResampleToGrid",
                                         Input=mesh_src)
        ippl_resampled.UseInputBounds = 0
        ippl_resampled.SamplingBounds = list(global_bounds)
        ippl_resampled.SamplingDimensions = global_extent

    # ResampleToImage produces Point Data by default
    associate = "POINTS"
    ippl_scalar = ippl_resampled
    _filters[f"{cname}.{name}.resample"] = ippl_resampled

    # Setup View
    view = CreateView('RenderView', registrationName=f"View_{label}_{name}")
    view.BackEnd = 'OSPRay raycaster'
    view.OSPRayMaterialLibrary = GetMaterialLibrary()
    view.ViewSize = [2000, 1500]
    view.AxesGrid = 'Grid Axes 3D Actor'
    view.AxesGrid.Visibility = 1
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = 'Gradient'
    SetActiveView(view)

    # Initial Camera (pre-trim bounds keep camera framing stable, no zoom)
    ippl_scalar.UpdatePipeline()
    auto_camera_from_bounds(view, original_global_bounds)

    # Show Data
    ippl_scalarDisplay = Show(ippl_scalar, view)

    # Transfer Functions
    densityLUT = GetColorTransferFunction(name)
    densityLUT.RGBPoints = [-2.00, 0.231373, 0.298039, 0.752941,
                             0.00, 0.865003, 0.865003, 0.865003,
                             2.00, 0.705882, 0.0156863, 0.14902]
    densityLUT.ScalarRangeInitialized = 1.0

    densityPWF = GetOpacityTransferFunction(name)
    densityPWF.Points = [-2.00, 1.00, 0.5, 0.0,
                         -1.20, 0.75, 0.5, 0.0,
                         -0.80, 0.25, 0.5, 0.0,
                         # --- The Zero Wall ---
                         -0.01, 0.00, 0.5, 0.0,
                          0.00, 1.00, 0.5, 0.0,
                          0.01, 0.00, 0.5, 0.0,
                         # ---------------------
                          0.80, 0.25, 0.5, 0.0,
                          1.20, 0.75, 0.5, 0.0,
                          2.00, 1.00, 0.5, 0.0]
    densityPWF.ScalarRangeInitialized = 1

    # Configure Display
    ippl_scalarDisplay.Representation = 'Volume'
    ippl_scalarDisplay.LookupTable = densityLUT
    ippl_scalarDisplay.ScaleTransferFunction = 'Piecewise Function'
    ippl_scalarDisplay.ScaleFactor = 2.0
    ippl_scalarDisplay.GaussianRadius = 0.1

    ippl_scalarDisplay.ColorArrayName = [associate, name]

    # Explicit Opacity Mapping
    ippl_scalarDisplay.OpacityArrayName = [associate, name]
    ippl_scalarDisplay.OpacityTransferFunction = 'Piecewise Function'
    ippl_scalarDisplay.ScalarOpacityUnitDistance = 4.00
    ippl_scalarDisplay.ScalarOpacityFunction = densityPWF

    # Scalar Bar
    densityLUTColorBar = GetScalarBar(densityLUT, view)
    densityLUTColorBar.Title = name
    densityLUTColorBar.ComponentTitle = 'Magnitude'
    densityLUTColorBar.Visibility = 1
    ippl_scalarDisplay.SetScalarBarVisibility(view, True)

    # Extractors
    pNG1 = CreateExtractor('PNG', view, registrationName=f"PNG_{label}_{name}")
    pNG1.Trigger = 'Time Step'
    pNG1.Trigger.Frequency = 1
    pNG1.Writer.FileName = f"{name}_ScalarField_" + "{timestep:06d}{camera}.png"
    pNG1.Writer.ImageResolution = [2000, 1500]
    pNG1.Writer.TransparentBackground = 0
    pNG1.Writer.Format = 'PNG'

    state = {
        'name': name,
        'src': ippl_scalar,
        'view': view,
        'lut': densityLUT,
        'pwf': densityPWF,
        'associate': associate,
        'original_bounds': original_global_bounds,
        'show_proxy': ippl_scalar,
    }
    _png_sfield_channels.append(state)

    if live_view is not None:
        SetActiveView(live_view)
    return state


def _setup_vfield_view(cname, mesh_src, name, assoc, single_vector):
    """Dedicated PNG view for one vector mesh array (png_ext_vfield.py recipe).

    Arrow glyphs oriented by the field, magnitude-driven color/opacity with
    the blue-white-red transfer function, camera and ranges re-fit every 10th
    cycle. Output 'VectorField_NNNNNN.png' (label-prefixed when the channel
    carries more than one vector array, to avoid filename collisions).
    """
    label = cname.replace("/", "_")

    # Setup View
    view = CreateView('RenderView', registrationName=f"View_{label}_{name}")
    view.ViewSize = [1247, 1176]
    view.BackEnd = 'OSPRay raycaster'
    view.StereoType = 'Crystal Eyes'
    view.LegendGrid = 'Legend Grid Actor'
    view.AxesGrid = 'Grid Axes 3D Actor'
    view.AxesGrid.Visibility = 1
    view.OSPRayMaterialLibrary = GetMaterialLibrary()
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = 'Gradient'
    SetActiveView(view)

    # Initial adaptive Camera set
    ippl_vector_info = mesh_src.GetDataInformation()
    local_bounds = ippl_vector_info.GetBounds()
    global_bounds = get_global_spatial_bounds(local_bounds)
    auto_camera_from_bounds(view, global_bounds)
    # needed for glyph scale ... use global diagonal
    diag = compute_bounding_box_scale(global_bounds)

    # Vector Field glyphs
    glyph1 = Glyph(registrationName=f"{cname}.{name}.Glyph",
                   Input=mesh_src, GlyphType='Arrow')
    glyph1.OrientationArray = [assoc, name]
    glyph1.GlyphTransform = 'Transform2'
    glyph1.ScaleFactor = diag/30
    _filters[f"{cname}.{name}.glyph"] = glyph1

    # setup initial transfer function for colouring and opacity
    fieldStrengthTF2D = GetTransferFunction2D(name)
    fieldStrengthTF2D.ScalarRangeInitialized = 1
    fieldStrengthTF2D.Range = [0.00, 2.00, 0.0, 1.0]
    fieldStrengthLUT = GetColorTransferFunction(name)
    fieldStrengthLUT.TransferFunction2D = fieldStrengthTF2D
    fieldStrengthLUT.ScalarRangeInitialized = 1
    fieldStrengthLUT.RGBPoints = [0.00, 0.231373, 0.298039, 0.752941,
                                  1.00, 0.865003, 0.865003, 0.865003,
                                  2.00, 0.705882, 0.0156863, 0.14902]
    fieldStrengthLUTColorBar = GetScalarBar(fieldStrengthLUT, view)
    fieldStrengthLUTColorBar.Title = 'fieldStrength'
    fieldStrengthLUTColorBar.ComponentTitle = 'Magnitude'
    fieldStrengthLUTColorBar.Visibility = 1
    fieldStrengthLUT.EnableOpacityMapping = True
    fieldStrengthPWF = GetOpacityTransferFunction(name)
    fieldStrengthPWF.ScalarRangeInitialized = 1
    fieldStrengthPWF.Points = [0.00, 0.0, 0.5, 0.0,
                               0.50, 0.2, 0.5, 0.0,
                               2.00, 1.0, 0.5, 0.0]

    # configure displayed data
    glyph1Display = Show(glyph1, view, 'GeometryRepresentation')
    glyph1Display.Representation = 'Surface'
    glyph1Display.LookupTable = fieldStrengthLUT
    glyph1Display.ColorArrayName = ['POINTS', name]
    glyph1Display.OpacityTransferFunction = fieldStrengthPWF
    glyph1Display.DataAxesGrid = 'Grid Axes Representation'
    glyph1Display.SetScalarBarVisibility(view, True)

    # setup extractors
    pNG1 = CreateExtractor('PNG', view, registrationName=f"pNG_{label}_{name}")
    pNG1.Trigger = 'Time Step'
    pNG1.Trigger.Frequency = 1
    if single_vector:
        pNG1.Writer.FileName = 'VectorField_{timestep:06d}{camera}.png'
    else:
        pNG1.Writer.FileName = f"{name}_VectorField_" + "{timestep:06d}{camera}.png"
    pNG1.Writer.ImageResolution = [1247, 1176]
    pNG1.Writer.TransparentBackground = 0
    pNG1.Writer.Format = 'PNG'

    state = {
        'name': name,
        'source': mesh_src,
        'view': view,
        'lut': fieldStrengthLUT,
        'pwf': fieldStrengthPWF,
        'assoc': assoc,
        'show_proxy': glyph1,
    }
    _png_vfield_channels.append(state)

    if live_view is not None:
        SetActiveView(live_view)
    return state


# Mesh channel views are deferred to the first catalyst_execute: array names
# and associations are only known once the producer actually delivered data.
_pending_mesh = []
_png_particle_channels = {}
_png_sfield_channels = []
_png_vfield_channels = []


def _setup_mesh_channels():
    """Build the per-array render pipelines for all mesh channels.

    A mesh channel can carry several arrays (the former design had one
    channel per field); each array gets its own view, styling and PNG
    extractor exactly like the former per-field png_ext_*.py scripts.
    """
    global _pending_mesh
    for cname, mesh_src in _pending_mesh:
        info = mesh_src.GetDataInformation()
        cell_info = info.GetCellDataInformation()
        point_info = info.GetPointDataInformation()
        ghost_info = cell_info.GetArrayInformation("vtkGhostType") if cell_info else None

        arrays = []
        # RankID is a per-rank ProcessIds field the C++ export attaches to
        # every mesh channel; the former scripts never rendered it.
        _skip = ("vtkGhostType", "RankID")
        for index in range(cell_info.GetNumberOfArrays()):
            array_info = cell_info.GetArrayInformation(index)
            array_name = array_info.GetName() if array_info else None
            if array_name and array_name not in _skip:
                arrays.append((array_name, array_info.GetNumberOfComponents(), "CELLS"))
        for index in range(point_info.GetNumberOfArrays()):
            array_info = point_info.GetArrayInformation(index)
            array_name = array_info.GetName() if array_info else None
            if array_name and array_name not in _skip:
                arrays.append((array_name, array_info.GetNumberOfComponents(), "POINTS"))

        vector_count = sum(1 for (_, comps, _) in arrays if comps > 1)

        # Live view: merged blocks shown plainly (former live behaviour)
        if live_view is not None:
            _log(f"   -> Using MergeBlocks for mesh channel '{cname}'")
            merged = MergeBlocks(registrationName=cname + '.MergedBlocks',
                                 Input=mesh_src)
            merged.MergePartitionsOnly = 1
            Show(merged, live_view)
            _filters[cname + "_merge"] = merged

        for array_name, comps, assoc in arrays:
            try:
                if comps > 1:
                    state = _setup_vfield_view(cname, mesh_src, array_name, assoc,
                                               single_vector=(vector_count == 1))
                else:
                    state = _setup_sfield_view(cname, mesh_src, array_name, ghost_info)
                # Also show the array pipeline plainly in the live view
                if live_view is not None:
                    Show(state['show_proxy'], live_view)
            except Exception as e:
                print_info_(f"PNG view setup failed for '{cname}' array '{array_name}': {e}")
    _pending_mesh = []


# Find proxies and set up pipelines in the global scope
pm = servermanager.ProxyManager()
for idx, (cname, ctype) in enumerate(_channels):
    if ctype == "mesh" and "/" in cname:
        # Mesh block nested inside a parent multimesh channel (registered via
        # addMeshBlock): the data is delivered inside the parent's producer,
        # so extract the block from it instead of creating a data-less
        # producer of our own.
        parent_name, block_name = cname.split("/", 1)
        proxy = _sources.get(parent_name)
        if proxy is None:
            proxy = PVTrivialProducer(registrationName=parent_name)
            proxy.UpdatePipeline()
            _sources[parent_name] = proxy
        mesh_src = ExtractBlock(
            registrationName=f"{parent_name}.{block_name}",
            Input=proxy,
            Selectors=[f"//{block_name}"]
        )
        mesh_src.UpdatePipeline()
        _filters[cname+"_block"] = mesh_src
    else:
        proxy = PVTrivialProducer(registrationName=cname)
        proxy.UpdatePipeline()
        _sources[cname] = proxy
        mesh_src = proxy

    if ctype == "multimesh":
        # Particle channel: multimesh with particles, boundingbox, transform
        particles = ExtractBlock(
            registrationName=f"{cname}.bunch",
            Input=proxy,
            Selectors=['//particles']
        )
        helper = ExtractBlock(
            registrationName=f"{cname}.box",
            Input=proxy,
            Selectors=['//boundingbox']
        )
        particles.UpdatePipeline()
        helper.UpdatePipeline()

        _filters[cname+"_main"] = particles
        _filters[cname+"_help"] = helper

        # Always expose the transform block as a separate source so it can be
        # selected as the second input of a Programmable Filter.
        transform = ExtractBlock(
            registrationName=f"{cname}.transform",
            Input=proxy,
            Selectors=['//transform']
        )
        transform.UpdatePipeline()
        _filters[cname+"_transform"] = transform

        if live_view is not None:
            # Plain default styling in the live view (former live behaviour)
            Show(particles, live_view)
            Show(helper, live_view)

        if png_enabled:
            _log(f"Creating PNG view for particle channel '{cname}'")
            _setup_particle_png_channel(cname, proxy, particles, helper)

        # Particles come as multimesh -> vtkPartitionedDataSetCollection -> VTPC
        if parsed.VTKextract == "ON":
            _log(f"Attaching VTPC extractor to complete multimesh particle proxy '{cname}'")
            _extractors[cname] = create_VTM_extractor(cname, proxy, 1)

    elif ctype == "mesh":
        # Mesh channel: uniform grid with multiple cell-data arrays.
        # Nested blocks (parent/block) work on the ExtractBlock of the parent
        # producer; top-level channels use their own producer.
        if live_view is not None or png_enabled:
            _pending_mesh.append((cname, mesh_src))

        if parsed.VTKextract == "ON":
            _extractors[cname] = create_VTPD_extractor(cname, mesh_src, 1)

# ------------------------------------------------------------------------------
# Setup steering channels
# ------------------------------------------------------------------------------

steer_channel_readers = {}
steer_channel_senders = {}
steer_channels = {}

if steer_enabled:
    if parsed.steer_channel_names :
        print_info_("===CREATING STEERABLES============="[0:40]+"|0")

    else:
        print_info_("No channel names provided in parsed.channel_names.")


    # ------------------------------------------------------------------------------
    # backward / outgoing steering channels
    # ------------------------------------------------------------------------------
    print_info_("BACKWARD")
    try:
        # Unified sender: one proxy carrying one property per channel, single result mesh
        sender_all = CreateSteerableParameters(
            steerable_proxy_type_name           = "SteerableParameters_SCALARS",
            steerable_proxy_registration_name   = "SteeringParameters_SCALARS",
            result_mesh_name                    = "steerable_channel_backward_all"
        )

        if sender_all is None:
            print_info_("Error: SteerableParameters_ALL proxy not found (CreateSteerableParameters returned None).")
        else:
            print_info_("SteerableParameters_ALL loaded successfully.")


        # Dynamically detect struct-array namespaces from steer_channel_names with prefix 'array:'
        # Example entries: 'array:LinMaps.time', 'array:simpVec.temperature'
        struct_array_senders = {}
        detected_namespaces = set()
        try:
            for entry in (parsed.steer_channel_names or []):
                if not isinstance(entry, str):
                    continue
                if not entry.startswith("array:"):
                    continue
                # Extract namespace between ':' and first '.'; fallback to full tail if no '.'
                tail = entry.split(":", 1)[1]
                ns = tail.split(".", 1)[0] if "." in tail else tail
                if ns:
                    detected_namespaces.add(ns)
        except Exception as e:
            print_info_(f"[DEBUG][WARN] Failed to parse struct-array namespaces: {e}", level=2)

        # Create one sender per detected namespace (e.g., 'LinMaps', 'simpVec')
        for ns in sorted(detected_namespaces):
            sender = CreateSteerableParameters(
                steerable_proxy_type_name         = f"SteerableParameters_{ns}",
                steerable_proxy_registration_name = f"Steering_{ns}",
                result_mesh_name                  = "steerable_channel_backward_all"
            )
            struct_array_senders[ns] = sender

            if sender is None:
                print_info_(f"[DEBUG][WARN] Failed to create struct array sender '{ns}'", level=2)
            else:
                print_info_(f"[DEBUG] Created struct array sender '{ns}'", level=2)

    except Exception as e:
        print_info_(f"Exception while loading (backward) SteerableParameters: {e}")

    print_info_("===CREATING STEERABLES=============="[0:40]+"|1")



# ------------------------------------------------------------------------------
print_info_("=== Printing Proxy Overview ============"[0:40]+"0")
if verbosity >= _CATALYST_INFO_LEVEL:
    print_proxy_overview()
print_info_("=== Printing Proxy Overview ============"[0:40]+"1")
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
def catalyst_initialize():
    print_info_("catalyst_initialize()"+exp_string)

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Manually update the proxy's pipeline.# This is necessary to make the new data
# available for live and to downstream filters (like the extractors). Not sure when why vtk
# update is needed. Extractors objects themselves don't need to be updated i think.
# Inputs also don't need to be reset. Backwards channels are entirely managed by the
# proxy file so? no update calls needed?...
def catalyst_execute(info):
    print_info_("_________executing (cycle={}, time={})___________".format(info.cycle, info.time))
    print_info_("catalyst_execute()::"+exp_string)
    pm = servermanager.ProxyManager()

    global parsed
    global steer_channels
    global _extractors
    global _sources
    global _filters
    global _pending_mesh


    # Loop over all registered vis channels
    for name, proxy in _sources.items():
        proxy.UpdatePipeline()
        proxy.UpdateVTKObjects()


    for name_, filter in _filters.items():
            filter.UpdatePipeline()
            filter.UpdateVTKObjects()

    # Create deferred mesh channel views now that data (and array names) arrived
    if _pending_mesh:
        try:
            _setup_mesh_channels()
        except Exception as e:
            print_info_(f"Mesh channel view setup failed: {e}")

    # Adapt particle PNG views: coloring, point size and camera (every cycle,
    # as in png_ext_particle.py)
    for cname, state in _png_particle_channels.items():
        try:
            point_data_info = state['particles'].GetDataInformation().GetPointDataInformation()
            _configure_particle_coloring(state, point_data_info)

            if state['color_array_name'] is not None:
                color_array_info = point_data_info.GetArrayInformation(state['color_array_name'])
                if color_array_info:
                    component = -1 if state['color_components'] > 1 else 0
                    local_min, local_max = color_array_info.GetComponentRange(component)
                    gmin, gmax = get_global_range(local_min, local_max)
                    nice_min, nice_max = nice_bounds(gmin, gmax)
                    state['color_lut'].RescaleTransferFunction(nice_min, nice_max)

            pos_array_info = point_data_info.GetArrayInformation('position')
            if pos_array_info:
                bounds = get_global_spatial_bounds(state['producer'].GetDataInformation().GetBounds())
                auto_camera_from_bounds(state['view'], bounds)
                diagonal = compute_bounding_box_scale(bounds)
                state['display'].GaussianRadius = diagonal/500
        except Exception as e:
            print_info_(f"Particle PNG view update failed for '{cname}': {e}")

    # Adapt scalar field PNG views: ranges and camera (every 10th cycle,
    # as in png_ext_sfield.py)
    if info.cycle % 10 == 0:
        for state in _png_sfield_channels:
            try:
                state['src'].UpdatePipeline()
                scalar_info = state['src'].GetDataInformation()
                if state['associate'] == "POINTS":
                    data_array_info = scalar_info.GetPointDataInformation().GetArrayInformation(state['name'])
                else:
                    data_array_info = scalar_info.GetCellDataInformation().GetArrayInformation(state['name'])

                if data_array_info:
                    local_vmin, local_vmax = data_array_info.GetComponentRange(-1)
                    global_vmin, global_vmax = get_global_range(local_vmin, local_vmax)
                    nice_min, nice_max = nice_bounds_sym(global_vmin, global_vmax)
                    # Keep camera framing using the original domain size to
                    # avoid visual zoom on trims/resampling
                    auto_camera_from_bounds(state['view'], state['original_bounds'])
                    state['lut'].RescaleTransferFunction(nice_min, nice_max)
                    state['pwf'].RescaleTransferFunction(nice_min, nice_max)
            except Exception as e:
                print_info_(f"Scalar field PNG view update failed for '{state['name']}': {e}")

        # Adapt vector field PNG views: ranges and camera (every 10th cycle,
        # as in png_ext_vfield.py)
        for state in _png_vfield_channels:
            try:
                vector_info = state['source'].GetDataInformation()
                if state['assoc'] == "CELLS":
                    field_array_info = vector_info.GetCellDataInformation().GetArrayInformation(state['name'])
                else:
                    field_array_info = vector_info.GetPointDataInformation().GetArrayInformation(state['name'])

                # Adjust camera dynamically using global bounds
                auto_camera_from_bounds(state['view'],
                                        get_global_spatial_bounds(vector_info.GetBounds()))
                if field_array_info:
                    # magnitude ...
                    local_vmin, local_vmax = field_array_info.GetComponentRange(-1)
                    gmin, gmax = get_global_range(local_vmin, local_vmax)
                    nice_min, nice_max = nice_bounds(gmin, gmax)
                    # Update color and opacity transfer function
                    state['lut'].RescaleTransferFunction(nice_min, nice_max)
                    state['pwf'].RescaleTransferFunction(nice_min, nice_max)
            except Exception as e:
                print_info_(f"Vector field PNG view update failed for '{state['name']}': {e}")

    if steer_enabled and  parsed.show_forward_channels == "ON":
        for name, reader in steer_channel_readers.items():
            reader.UpdatePipeline()
            reader.UpdateVTKObjects()


    if options.EnableCatalystLive:
        # time.sleep(1)
        pass

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
def catalyst_finalize():
    print_info_("catalyst_finalize()::" + exp_string)
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
print_info_("==========================================================="[0:55]+"|")
print_info_("========== END OF catalyst_pipeline GLOBAL SCOPE =========0"[0:55]+"|")
print_info_("==========================================================="[0:55]+"|\n\n")
# ------------------------------------------------------------------------------
