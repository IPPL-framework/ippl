"""! \file png_ext_particle.py
\brief Catalyst PNG extractor for 3D particles (ParticleContainer/ParticleBase).
\details Visualizes particle point data with adaptive camera and velocity-based
coloring. Expects 'position' and 'velocity' arrays and is orchestrated by
pipeline_default.py. Generates PNG extracts and supports Catalyst Live.
"""

# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0

########################################################
######################################################## 
# PNG extractor script for paraview catalyst. 
# Visualizes 3D particles. (ParticleContainer/ParticleBase)
# 
# Currently hard coded to rely on attributes:
# - 'position'
# - 'velocity'
# Is adaptive: Attempts to set Camera Angle and colouring 
# of particles (dependent on velocity magnitude) adaptive to 
# current frame, range and scale (every 10'th step).
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
    GetTransferFunction2D,
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


# Apply Threshold on 'velocity' (Magnitude)
# We set the range to essentially "All valid numbers"
# ippl_particle_t = Threshold(registrationName='Filter_Particles', Input=ippl_particle_m)
# ippl_particle_t.Scalars = ['POINTS', 'velocity']
# ippl_particle_t.ThresholdMethod = 'Above Upper Threshold' # Or 'Between'
# ippl_particle_t.UpperThreshold = -1.0 # Velocity magnitude is always >= 0, so this keeps everything valid
# Note: In newer ParaView versions (5.10+), properties might be:
# ippl_particle.ThresholdRange = [-1.0, 999999999.9]






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
# ----------------------------------------------------------------
# setup initial transfer function for colouring and opacity
# ----------------------------------------------------------------
velocityTF2D = GetTransferFunction2D('velocity')
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.TransferFunction2D = velocityTF2D
velocityLUT.RGBPoints = [0.050641224585373915, 0.231373, 0.298039, 0.752941, 
                         2.3924284143906274, 0.865003, 0.865003, 0.865003, 
                         4.734215604195881, 0.705882, 0.0156863, 0.14902]
# ----------------------------------------------------------------
# configure displayed data
# ----------------------------------------------------------------
# ippl_particleDisplay.Representation = 'Points'
ippl_particleDisplay.Representation = 'Point Gaussian'
ippl_particleDisplay.LookupTable = velocityLUT
# point size ...
# ippl_particleDisplay.GaussianRadius = 1
ippl_particleDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_particleDisplay.SelectInputVectors = ['POINTS', 'position']
ippl_particleDisplay.ColorArrayName = ['POINTS', 'velocity']
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.Title = 'velocity'
velocityLUTColorBar.ComponentTitle = 'Magnitude'
velocityLUTColorBar.Visibility = 1
ippl_particleDisplay.SetScalarBarVisibility(renderView1, True)

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

    ippl_particle_p.UpdatePipeline()
    # ippl_particle_bunch
    # ippl_particle_box
    # ippl_particle

    # SetActiveView(renderView1)
    # print(info)
    # print(info.__dict__.keys())


    if info.cycle % 1 == 0:
        particle_info = ippl_particle_p.GetDataInformation()

        point_data_info = particle_info.GetPointDataInformation()
        # point_data_info = particle_info.GetFieldDataInformation()
        # print(point_data_info)

        vel_array_info = point_data_info.GetArrayInformation('velocity')
        pos_array_info = point_data_info.GetArrayInformation('position')

        if vel_array_info:
            local_vmin, local_vmax = vel_array_info.GetComponentRange(-1)
            gmin, gmax = get_global_range(local_vmin, local_vmax)
            nice_min, nice_max = nice_bounds(gmin, gmax)
            vel_lut = GetColorTransferFunction('velocity')
            vel_lut.RescaleTransferFunction(nice_min, nice_max)
        else:
            print_info_("Velocity array not found!")
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

