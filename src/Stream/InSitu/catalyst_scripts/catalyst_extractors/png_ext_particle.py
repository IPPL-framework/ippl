r"""! \file png_ext_particle.py
\brief Catalyst PNG extractor for 3D particles (ParticleContainer/ParticleBase).
\details Visualizes particle point data with an adaptive camera. Particles are
colored by their first custom point attribute (using magnitude for vectors), or
solid red when no custom attribute exists. Generates PNG extracts and supports
Catalyst Live.
"""

# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0

########################################################
######################################################## 
# PNG extractor script for paraview catalyst. 
# Visualizes 3D particles. (ParticleContainer/ParticleBase)
# 
# The position, particle-ID and rank-ID arrays are built-ins. Any first
# additional point-data array is selected for coloring. Scalar arrays are used
# directly and multi-component arrays are colored by magnitude.
# The camera, color range and particle scale adapt to the current frame.
# 
# 
# Relies on pipeline_default.py to update pipeline else might
# cause errors (i think)
# 
# 
# 
# Possible TODO:
#  - Customize extraction frequency
#  - Customize "rescale" frequency
#  - More
########################################################
########################################################


import paraview
from paraview.simple import *
from paraview import catalyst
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 12
from paraview.simple import (
    PVTrivialProducer,
    GetMaterialLibrary,
    CreateView,
    Show,
    ColorBy,
    GetColorTransferFunction,
    GetScalarBar,
    CreateExtractor,
    SetActiveView,
    SetActiveSource
)

from paraview.simple import *
from paraview import print_info
import argparse
import math
# ----------------------------------------------------------------
# helpers used for adaptive visualization
# ----------------------------------------------------------------
from catalystSubroutines import (
    nice_bounds,
    auto_camera_from_bounds,
    compute_bounding_box_scale,
    get_global_spatial_bounds,
    get_global_range,
    hide_source_from_gui
    # print_info_
)
def print_info_(s, level=0):
    global verbosity
    if verbosity>level:
        print_info(s)


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


particle_color_array_name = None
particle_color_components = 0
particle_color_lut = None
particle_coloring_initialized = False


def _configure_particle_coloring(point_data_info):
    """Select the first custom attribute, or configure solid red coloring."""
    global particle_color_array_name
    global particle_color_components
    global particle_color_lut
    global particle_coloring_initialized

    array_name, components = _first_custom_particle_attribute(point_data_info)
    if (particle_coloring_initialized
            and array_name == particle_color_array_name
            and components == particle_color_components):
        return

    if particle_color_lut is not None:
        try:
            GetScalarBar(particle_color_lut, renderView1).Visibility = 0
        except Exception:
            pass

    particle_color_array_name = array_name
    particle_color_components = components
    particle_color_lut = None
    particle_coloring_initialized = True

    if array_name is None:
        ippl_particleDisplay.ColorArrayName = ['POINTS', '']
        ippl_particleDisplay.AmbientColor = [1.0, 0.0, 0.0]
        ippl_particleDisplay.DiffuseColor = [1.0, 0.0, 0.0]
        print_info_("No custom particle attribute found; using solid red coloring.")
        return

    color_spec = ("POINTS", array_name)
    component_title = ""
    if components > 1:
        color_spec = ("POINTS", array_name, "Magnitude")
        component_title = "Magnitude"

    ColorBy(ippl_particleDisplay, color_spec)
    particle_color_lut = GetColorTransferFunction(array_name)
    color_bar = GetScalarBar(particle_color_lut, renderView1)
    color_bar.Title = array_name
    color_bar.ComponentTitle = component_title
    color_bar.Visibility = 1
    ippl_particleDisplay.SetScalarBarVisibility(renderView1, True)

    mode = "magnitude" if components > 1 else "scalar values"
    print_info_(f"Coloring particles by {mode} of custom attribute '{array_name}'.")
# ----------------------------------------------------------------
# ----------------------------------------------------------------
paraview.simple._DisableFirstRenderCameraReset()
SetActiveView(None)
# ----------------------------------------------------------------
# Parse arguments received via conduit node
# ----------------------------------------------------------------
arg_list = paraview.catalyst.get_args()
# print_info_(f"Arguments received: {arg_list}")
parser = argparse.ArgumentParser()
parser.add_argument("--channel_name", default="DEFAULT_CHANNEL", help="Needed to correctly setup association between script name and conduti channel.")
parser.add_argument("--label", default="DEFAULAAAAAAAAT_CHANNEL", help="Needed to correctly setup association between script name and conduti channel.")
parser.add_argument("--experiment_name", default="_", help="Needed to correctly for safe folder.")
parser.add_argument("--verbosity", type=int, default="1", help="Communicate the catalyst Output Level from the simulation")
parsed = parser.parse_args(arg_list)

label = parsed.label
exp_string = parsed.experiment_name
verbosity = parsed.verbosity
print_info_("_global__scope__()::" + parsed.channel_name)
# ----------------------------------------------------------------
# create a new 'XML Partitioned Dataset Reader'
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 1. The Source
# ----------------------------------------------------------------
ippl_producer = PVTrivialProducer(registrationName=parsed.channel_name)

# ----------------------------------------------------------------
# 2. Extract ONLY the parts you want
# ----------------------------------------------------------------
# Use ExtractBlock to filter down to the specific conduit node/block
# ippl_merged = MergeBlocks(registrationName='Merged_Subset', Input=ippl_producer)
# ippl_merged.MergePartitionsOnly = 0

# subset_extractor
# subset_extractor = ExtractBlock(registrationName='Selected_Parts', Input=ippl_producer)
# subset_extractor.Selectors = ['//block_main'] 
# ippl_particle = MergeBlocks(registrationName='Merged_Subset', Input=subset_extractor)
# ippl_particle.MergePartitionsOnly = 0








cname = parsed.channel_name
ippl_particle_p = PVTrivialProducer(registrationName = cname)

data_info = ippl_particle_p.GetDataInformation()
# print(data_info.__dict__.keys())
# print(data_info.DataInformation)
# print(data_info.Proxy)
# print(data_info.Idx)
p_info = data_info.GetPointDataInformation()
f_info = data_info.GetFieldDataInformation()
# print(p_info)
# print(f_info)


# ippl_particle_e = ExtractBlock(
#                 registrationName=f"{cname[15:]}_bunch_png_ext",
#                 Input=ippl_particle_p,
#                 Assembly = 'Hierarchy',
#                 # Selectors=['//block_main']
#                 Selectors=['//main']
#                 # Selectors=['/Root/block_main']
#                 # Selectors=['/Root/main']
#             )
# ippl_particle_e.UpdatePipeline()

# ippl_particle_m = MergeBlocks(registrationName=cname[12:]+'_MergedBlocks',
#                                  Input=ippl_particle_p)
# ippl_particle_m.MergePartitionsOnly = 1


# ippl_particle_m.UpdatePipeline()



# fetch proxy from live script instead of creating its own...?...
# might break the pipeline when selected so lets leave it like this ... 
ippl_particle_bunch = ExtractBlock(
                registrationName=f"{cname[15:]}_bunch_png_ext",
                Input=ippl_particle_p,
                Assembly = 'Hierarchy',
                Selectors=['//block_main']
                # Selectors=['//main']
                # Selectors=['/Root/block_main']
                # Selectors=['/Root/main']
            )
hide_source_from_gui(ippl_particle_bunch)

ippl_particle_box = ExtractBlock(
                registrationName=f"{cname[15:]}_box_png_ext",
                Input=ippl_particle_p,
                Assembly = 'Hierarchy',
                Selectors=['//block_help']
                # Selectors=['//main']
                # Selectors=['/Root/block_main']
                # Selectors=['/Root/main']
            )
hide_source_from_gui(ippl_particle_box)

# ippl_particle_e.UpdatePipeline()


ippl_particle = ippl_particle_bunch



# ----------------------------------------------------------------
# setup visualisation view for extraction pipeline in renderview1
# ----------------------------------------------------------------
view_name = f"View_{cname}"
# renderView1 = CreateView('RenderView', registrationName=view_name)
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2000, 1500]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [9.804888932121028, 10.012698468217557, 10.017046030145888]
renderView1.HiddenLineRemoval = 1
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [42.01243955103898, 42.22024908713551, 42.224596649063855]
renderView1.CameraFocalPoint = [9.804888932121028, 10.012698468217557, 10.017046030145888]
renderView1.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 14.438249951766423
renderView1.BackEnd = 'OSPRay raycaster'
materialLibrary1 = GetMaterialLibrary()
renderView1.OSPRayMaterialLibrary = materialLibrary1
renderView1.AxesGrid.Visibility = 1

renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
# renderView1.Background2 = [0.0, 0.6666666666666666, 1.0]
# renderView1.Background = [0.0, 0.0, 0.4980392156862745]
SetActiveView(renderView1)
# ----------------------------------------------------------------
# Initial adaptive Camera set
# ----------------------------------------------------------------
particle_info = ippl_particle_p.GetDataInformation()
local_bounds = particle_info.GetBounds()
bounds = get_global_spatial_bounds(local_bounds)
# print(particle_info.__dict__.keys())
# print(particle_info.Idx)
# print(particle_info.Proxy)
# print(particle_info.DataInformation)
# print(bounds)

auto_camera_from_bounds(renderView1, bounds)
# ----------------------------------------------------------------
# choose Data to visualize and show in renderView1
# ----------------------------------------------------------------
ippl_particleDisplay = Show(ippl_particle, renderView1, 'UnstructuredGridRepresentation')
# ippl_particleDisplay = Show(ippl_particle, renderView1, 'GeometryRepresentation')
# configure displayed data
# ----------------------------------------------------------------
# ippl_particleDisplay.Representation = 'Points'
ippl_particleDisplay.Representation = 'Point Gaussian'
# point size ...
# ippl_particleDisplay.GaussianRadius = 1
ippl_particleDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_particleDisplay.SelectInputVectors = ['POINTS', 'position']

# Point-array metadata may not be available while the Catalyst script is first
# loaded. Start with the guaranteed fallback and discover custom attributes in
# catalyst_execute after the producer and main particle block are updated.
ippl_particleDisplay.ColorArrayName = ['POINTS', '']
ippl_particleDisplay.AmbientColor = [1.0, 0.0, 0.0]
ippl_particleDisplay.DiffuseColor = [1.0, 0.0, 0.0]

# ----------------------------------------------------------------
# visualize helper box as yellow outline
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# visualize helper box as yellow outline
# ----------------------------------------------------------------
ippl_particle_boxDisplay = Show(ippl_particle_box, renderView1, 'GeometryRepresentation')
ippl_particle_boxDisplay.Representation = 'Outline'
ippl_particle_boxDisplay.AmbientColor = [1.0, 1.0, 0.0]
ippl_particle_boxDisplay.DiffuseColor = [1.0, 1.0, 0.0]
ippl_particle_boxDisplay.LineWidth = 2.0
ippl_particle_boxDisplay.Opacity = 0.1

# Correct way to set Solid Color mode in ParaView Python
# This tells ParaView "Don't use any array, just use DiffuseColor"
ippl_particle_boxDisplay.ColorArrayName = ['POINTS', ''] 

# Remove explicit LookupTable manipulation and Bar Visibility calls.
# Since we set it to Solid Color above, ParaView automatically hides the bar.
# --------------------------------------------------------------
# setup extractors
# --------------------------------------------------------------
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG_'+ cname)
pNG1.Trigger = 'Time Step'
pNG1.Writer.FileName = label+'_Particles_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [2000, 1500]
pNG1.Writer.TransparentBackground = 0
pNG1.Writer.Format = 'PNG'
SetActiveSource(pNG1)
# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 0
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'data_png_extracts_' + exp_string
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)





# ------------------------------------------------------------------------------
def catalyst_execute(info):
    print_info_("catalyst_execute()::"+parsed.channel_name)

    global ippl_particle_bunch
    global ippl_particle_box
    global ippl_particle
    global ippl_particle_p
    global renderView1
    global pNG1
    global particle_color_array_name
    global particle_color_components
    global particle_color_lut

    ippl_particle_p.UpdatePipeline()
    ippl_particle_bunch.UpdatePipeline()

    # SetActiveView(renderView1)
    # print(info)
    # print(info.__dict__.keys())


    if info.cycle % 1 == 0:
        particle_info = ippl_particle_p.GetDataInformation()

        bunch_info = ippl_particle_bunch.GetDataInformation()
        point_data_info = bunch_info.GetPointDataInformation()
        _configure_particle_coloring(point_data_info)

        color_array_info = None
        if particle_color_array_name is not None:
            color_array_info = point_data_info.GetArrayInformation(particle_color_array_name)
        pos_array_info = point_data_info.GetArrayInformation('position')

        if color_array_info:
            component = -1 if particle_color_components > 1 else 0
            local_min, local_max = color_array_info.GetComponentRange(component)
            gmin, gmax = get_global_range(local_min, local_max)
            nice_min, nice_max = nice_bounds(gmin, gmax)
            particle_color_lut.RescaleTransferFunction(nice_min, nice_max)
        if pos_array_info:
            local_bounds = particle_info.GetBounds()
            bounds = get_global_spatial_bounds(local_bounds)
            auto_camera_from_bounds(renderView1, bounds)
            # print(bounds)


            def nice_pair(vmin, vmax):
                # Use nice_bounds for each axis
                return nice_bounds(vmin, vmax)

            # Compute nice bounds for each axis
            x0, x1 = nice_pair(bounds[0], bounds[1])
            y0, y1 = nice_pair(bounds[2], bounds[3])
            z0, z1 = nice_pair(bounds[4], bounds[5])
            diagonal = compute_bounding_box_scale(bounds)
            """ size """
            ippl_particleDisplay.GaussianRadius = diagonal/500

        else:
            print_info_("Position array not found!")
