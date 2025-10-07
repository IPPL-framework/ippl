# script-version: 2.0
from paraview.simple import *
from paraview import print_info

import argparse
import math


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
    def nice_pair(vmin, vmax):
        # Use nice_bounds for each axis
        return nice_bounds(vmin, vmax)

    # Compute nice bounds for each axis
    x0, x1 = nice_pair(bounds[0], bounds[1])
    y0, y1 = nice_pair(bounds[2], bounds[3])
    z0, z1 = nice_pair(bounds[4], bounds[5])

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

""" ideally offer option to add a field object, from which we can extract the true positional bounds of the simulation """
print_info("==='%s'======================="[0:28]+">",__name__)
paraview.simple._DisableFirstRenderCameraReset()
SetActiveView(None)


arg_list = paraview.catalyst.get_args()
# print_info(f"Arguments received: {arg_list}")
parser = argparse.ArgumentParser()
parser.add_argument("--channel_name", default="DEFAULT_CHANNEL", help="Needed to correctly setup association between script name and conduti channel.")
parsed = parser.parse_args(arg_list)
print_info(f"Parsed VTK extract options:     {parsed.channel_name}")


# ----------------------------------------------------------------
# create a new 'XML Partitioned Dataset Reader'
""" should be of the form ippl_vField_SUFFIX """
ippl_particle = PVTrivialProducer(registrationName = parsed.channel_name)



# ----------------------------------------------------------------
# ----------------------------------------------------------------
# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
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
renderView1.OSPRayMaterialLibrary = materialLibrary1
renderView1.AxesGrid.Visibility = 1


# change background ...
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
# renderView1.Background2 = [0.0, 0.6666666666666666, 1.0]
# renderView1.Background = [0.0, 0.0, 0.4980392156862745]



# show data from ippl_particle
ippl_particleDisplay = Show(ippl_particle, renderView1, 'UnstructuredGridRepresentation')

velocityTF2D = GetTransferFunction2D('velocity')
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.TransferFunction2D = velocityTF2D
# colouring ...
velocityLUT.RGBPoints = [0.050641224585373915, 0.231373, 0.298039, 0.752941, 2.3924284143906274, 0.865003, 0.865003, 0.865003, 4.734215604195881, 0.705882, 0.0156863, 0.14902]


ippl_particleDisplay.Representation = 'Point Gaussian'
ippl_particleDisplay.LookupTable = velocityLUT
# point size ...
# ippl_particleDisplay.GaussianRadius = 1


ippl_particleDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_particleDisplay.SelectInputVectors = ['POINTS', 'position']
ippl_particleDisplay.ColorArrayName = ['POINTS', 'velocity']

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.Title = 'velocity'
velocityLUTColorBar.ComponentTitle = 'Magnitude'
velocityLUTColorBar.Visibility = 1
ippl_particleDisplay.SetScalarBarVisibility(renderView1, True)
# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'Particles_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [2000, 1500]
pNG1.Writer.TransparentBackground = 0
pNG1.Writer.Format = 'PNG'
# ----------------------------------------------------------------
# restore active source
SetActiveSource(pNG1)
# ----------------------------------------------------------------


# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'
options.ExtractsOutputDirectory = 'data_png_extracts2'
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)





def catalyst_execute(info):
    print_info("'%s::catalyst_execute()'", __name__)
    global ippl_particle
    global renderView1

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

