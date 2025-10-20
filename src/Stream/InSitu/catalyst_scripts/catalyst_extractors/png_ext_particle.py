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
#  - Together with CPP don't rely on hard coded attributes
#  - additionally pass field string to have constistent bounds
#    and don't have to guess reference frame ...
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
from paraview import print_info
import argparse
import math

# ----------------------------------------------------------------
# helpers used for adaptive visualization
# ----------------------------------------------------------------

def nice_bounds(vmin, vmax):
    # Avoid zero range
    if vmin == vmax:
        return vmin, vmax
    # Find order of magnitude
    order = math.floor(math.log10(max(abs(vmin), abs(vmax), 1e-10)))
    scale = 10 ** order
    # Round down min, up max
    nice_min = math.floor(vmin / scale) * scale
    nice_max = math.ceil(vmax / scale) * scale
    return nice_min, nice_max



def set_camera(view, position=None, focal_point=None, view_up=None, parallel_scale=None):
    if position is not None:
        view.CameraPosition = position
    if focal_point is not None:
        view.CameraFocalPoint = focal_point
    if view_up is not None:
        view.CameraViewUp = view_up
    if parallel_scale is not None:
        view.CameraParallelScale = parallel_scale


def auto_camera_from_bounds(view, bounds):
    # bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            
    # we forced nice bounds via cpp passing additional dummy field
    # marking domain corners
    # def nice_pair(vmin, vmax):
    #     return nice_bounds(vmin, vmax)
    # x0, x1 = nice_pair(bounds[0], bounds[1])
    # y0, y1 = nice_pair(bounds[2], bounds[3])
    # z0, z1 = nice_pair(bounds[4], bounds[5])


    x0, x1, y0, y1, z0, z1 = bounds

    # Center and diagonal based on nice bounds
    center = [
        0.5 * (x0 + x1),
        0.5 * (y0 + y1),
        0.5 * (z0 + z1)
    ]
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)

    # Camera direction: from a diagonal (e.g., [1,1,1])
    direction = [1, 1.3, 0.6]
    norm = math.sqrt(sum(d*d for d in direction))
    direction = [d / norm for d in direction]

    # Camera position: center + direction * (distance)
    distance = 1.8 * diagonal  # adjust multiplier as needed
    cam_pos = [
        center[0] + direction[0] * distance,
        center[1] + direction[1] * distance,
        center[2] + direction[2] * distance
    ]
    set_camera(
        view,
        position=cam_pos,
        focal_point=center,
        view_up=[0, 0, 1],  # z-up
        parallel_scale=0.6 * diagonal
    )
    view.AxesGrid.UseCustomBounds = 1
    view.AxesGrid.CustomBounds = [x0, x1, y0, y1, z0, z1]





# ----------------------------------------------------------------
# ----------------------------------------------------------------
paraview.simple._DisableFirstRenderCameraReset()
SetActiveView(None)
# ----------------------------------------------------------------
# Parse arguments received via conduit node
# ----------------------------------------------------------------
arg_list = paraview.catalyst.get_args()
# print_info(f"Arguments received: {arg_list}")
parser = argparse.ArgumentParser()
parser.add_argument("--channel_name", default="DEFAULT_CHANNEL", help="Needed to correctly setup association between script name and conduti channel.")
parser.add_argument("--experiment_name", default="_", help="Needed to correctly for safe folder.")
parsed = parser.parse_args(arg_list)
exp_string = parsed.experiment_name
# channel_string = parsed.channel_name
print_info("_global__scope__()::" + parsed.channel_name)
# print_info(f"Parsed VTK extract options: {parsed.channel_name}")
# ----------------------------------------------------------------
# create a new 'XML Partitioned Dataset Reader'
# ----------------------------------------------------------------
ippl_particle = PVTrivialProducer(registrationName = parsed.channel_name)
# ----------------------------------------------------------------
# setup visualisation view for extraction pipeline in renderview1
# ----------------------------------------------------------------
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
particle_info = ippl_particle.GetDataInformation()
bounds = particle_info.GetBounds()
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
# --------------------------------------------------------------
# setup extractors
# --------------------------------------------------------------
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
pNG1.Trigger = 'TimeStep'
pNG1.Writer.FileName = 'Particles_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [2000, 1500]
pNG1.Writer.TransparentBackground = 0
pNG1.Writer.Format = 'PNG'
SetActiveSource(pNG1)
# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'
options.ExtractsOutputDirectory = 'data_png_extracts_' + exp_string
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)





# ------------------------------------------------------------------------------
def catalyst_execute(info):
    print_info("catalyst_execute()::"+parsed.channel_name)
    global ippl_particle
    global renderView1
    # print(info)
    # print(info.__dict__.keys())


    if info.cycle % 10 == 0:

        particle_info = ippl_particle.GetDataInformation()
        point_data_info = particle_info.GetPointDataInformation()

        vel_array_info = point_data_info.GetArrayInformation('velocity')
        pos_array_info = point_data_info.GetArrayInformation('position')

        if vel_array_info:
            vmin, vmax = vel_array_info.GetComponentRange(-1)
            nice_min, nice_max = nice_bounds(vmin, vmax)
            vel_lut = GetColorTransferFunction('velocity')
            vel_lut.RescaleTransferFunction(nice_min, nice_max)
        else:
            print_info("Velocity array not found!")
        if pos_array_info:
            bounds = particle_info.GetBounds()
            auto_camera_from_bounds(renderView1, bounds)
            # print(bounds)


            def nice_pair(vmin, vmax):
                # Use nice_bounds for each axis
                return nice_bounds(vmin, vmax)

            # Compute nice bounds for each axis
            x0, x1 = nice_pair(bounds[0], bounds[1])
            y0, y1 = nice_pair(bounds[2], bounds[3])
            z0, z1 = nice_pair(bounds[4], bounds[5])
            dx = x1 - x0
            dy = y1 - y0
            dz = z1 - z0
            diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)
            """ size """
            ippl_particleDisplay.GaussianRadius = diagonal/500

        else:
            print_info("Position array not found!")

