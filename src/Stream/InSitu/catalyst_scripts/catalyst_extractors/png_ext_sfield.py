# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'

from paraview import print_info
import math


print_info("==='%s'=============================="[0:30]+">",__name__)
paraview.simple._DisableFirstRenderCameraReset()


# ----------------------------------------------------------------
# helpers used for adaptive visualization
# ----------------------------------------------------------------
# Helper function to compute "nice" bounds
def nice_bounds(vmin, vmax):
    if vmin == vmax:
        return vmin, vmax
    order = math.floor(math.log10(max(abs(vmin), abs(vmax), 1e-10)))
    scale = 10 ** order
    nice_min = math.floor(vmin / scale) * scale
    nice_max = math.ceil(vmax / scale) * scale
    return nice_min, nice_max



def nice_bounds_sym(vmin, vmax):
    if vmin == vmax:
        return vmin, vmax
    order = math.floor(math.log10(max(abs(vmin), abs(vmax), 1e-10)))
    scale = 10 ** order
    nice_min = math.floor(vmin / scale) * scale
    nice_max = math.ceil(vmax / scale) * scale
    if -nice_min > nice_max:
        nice_max = -nice_min
    else: 
        nice_min = -nice_max

    return nice_min, nice_max


# Helper function to set the camera
def set_camera(view, position=None, focal_point=None, view_up=None, parallel_scale=None):
    if position is not None:
        view.CameraPosition = position
    if focal_point is not None:
        view.CameraFocalPoint = focal_point
        #  maybe default better ....
        view.CenterOfRotation = focal_point
    if view_up is not None:
        view.CameraViewUp = view_up
    if parallel_scale is not None:
        view.CameraParallelScale = parallel_scale

# Helper function to auto-adjust the camera based on bounds
def auto_camera_from_bounds(view, bounds):
    center = [
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5])
    ]
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    diagonal = math.sqrt(dx * dx + dy * dy + dz * dz)

    # "Nice" rounding for center and diagonal
    def nice_value(val):
        if val == 0:
            return 0
        order = math.floor(math.log10(abs(val)))
        scale = 10 ** order
        if val > 0:
            return math.ceil(val / scale) * scale
        else:
            return math.floor(val / scale) * scale

    nice_center = [nice_value(c) for c in center]
    nice_diagonal = nice_value(diagonal)

    # Camera position: look from a diagonal direction
    direction = [1, 1.3, 0.6]
    norm = math.sqrt(sum(d * d for d in direction))
    direction = [d / norm for d in direction]
    distance = 1.3 * nice_diagonal
    cam_pos = [
        nice_center[0] + direction[0] * distance,
        nice_center[1] + direction[1] * distance,
        nice_center[2] + direction[2] * distance
    ]

    set_camera(
        view,
        position=cam_pos,
        focal_point=nice_center,
        view_up=[0, 0, 1],  # z-up
        parallel_scale=0.6 * nice_diagonal
    )


# ----------------------------------------------------------------
# create a new 'XML Partitioned Dataset Reader'
ippl_scalar = PVTrivialProducer(registrationName='ippl_scalar')


# Create a new 'Render View'
SetActiveView(None) #prbbly not needed
renderView1 = CreateView('RenderView')

materialLibrary1 = GetMaterialLibrary()
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

renderView1.ViewSize = [2000, 1500]
renderView1.StereoType = 'Crystal Eyes'
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.AxesGrid.Visibility = 1

# change background ...
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
# renderView1.Background2 = [0.0, 0.6666666666666666, 1.0]
# renderView1.Background = [0.0, 0.0, 0.4980392156862745]


scalar_info = ippl_scalar.GetDataInformation()
bounds = scalar_info.GetBounds()
auto_camera_from_bounds(renderView1, bounds)
# restore active view
SetActiveView(renderView1)



# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------
ippl_scalarDisplay = Show(ippl_scalar, renderView1, 'UniformGridRepresentation')
# get 2D transfer function for 'density'
densityTF2D = GetTransferFunction2D('density')
densityTF2D.Range = [-2.00, 2.00, 0.0, 1.0]
densityTF2D.ScalarRangeInitialized = 1
# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.TransferFunction2D = densityTF2D
densityLUT.RGBPoints = [-2.00, 0.231373, 0.298039, 0.752941, 
                         0.00, 0.865003, 0.865003, 0.865003, 
                         2.00, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0
# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [-2.00, 1.00, 0.5, 0.0, 
                     -1.20, 0.75, 0.5, 0.0, 
                     -0.80, 0.25, 0.5, 0.0, 
                     -0.01, 0.00, 0.5, 0.0, 
                      0.00, 1.00, 0.5, 0.0, 
                      0.01, 0.00, 0.5, 0.0, 
                      0.80, 0.25, 0.5, 0.0, 
                      1.20, 0.75, 0.5, 0.0, 
                      2.00, 1.00, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1



# vmin, vmax = density_array_info.GetComponentRange(-1)
# nice_min, nice_max = nice_bounds_sym(vmin, vmax)
# densityLUT.RescaleTransferFunction(nice_min, nice_max)
# densityPWF.RescaleTransferFunction(nice_min, nice_max)



ippl_scalarDisplay.Representation = 'Volume'
ippl_scalarDisplay.LookupTable = densityLUT
ippl_scalarDisplay.OSPRayScaleFunction = 'Piecewise Function'
ippl_scalarDisplay.ScaleTransferFunction = 'Piecewise Function'
ippl_scalarDisplay.Assembly = 'Hierarchy'
ippl_scalarDisplay.ScaleFactor = 2.0
ippl_scalarDisplay.GaussianRadius = 0.1

ippl_scalarDisplay.DataAxesGrid = 'Grid Axes Representation'
ippl_scalarDisplay.TransferFunction2D = densityTF2D
ippl_scalarDisplay.ColorArrayName = ['CELLS', 'density']
ippl_scalarDisplay.ColorArray2Name = ['CELLS', 'density']

ippl_scalarDisplay.OpacityArrayName = ['CELLS', 'density']
ippl_scalarDisplay.OpacityTransferFunction = 'Piecewise Function'
ippl_scalarDisplay.ScalarOpacityUnitDistance = 4.00
ippl_scalarDisplay.ScalarOpacityFunction = densityPWF


densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.Title = 'density'
densityLUTColorBar.ComponentTitle = 'Magnitude'
densityLUTColorBar.Visibility = 1

densityLUTColorBar.DrawAnnotations = 1 
# ===> FIX PART 1: Link the opacity function directly to the color map.
densityLUT.EnableOpacityMapping = True

ippl_scalarDisplay.SetScalarBarVisibility(renderView1, True)


# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
pNG1.Trigger = 'Time Step'
pNG1.Trigger.Frequency = 1
pNG1.Writer.FileName = 'ScalarField_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [2000, 1500]
pNG1.Writer.Format = 'PNG'
SetActiveSource(pNG1)

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'data_png_extracts2'
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)



# ----------------------------------------------------------------
def catalyst_execute(info):
    print_info("'%s::catalyst_execute()'", __name__)

    global ippl_scalar
    global densityLUT
    global densityPWF
    # densityPWF = GetOpacityTransferFunction('density')
    # densityLUT = GetColorTransferFunction('density')



    if info.cycle % 10 == 0:
    # if info.cycle % 10 + 1 == 10:
        # Get scalar field bounds
        scalar_info = ippl_scalar.GetDataInformation()
        bounds = scalar_info.GetBounds()
        cell_data_info = scalar_info.GetCellDataInformation()
        density_array_info = cell_data_info.GetArrayInformation('density')
        # print(bounds)
        # print(cell_data_info)

        # if for any reason this changes, but unlikely...
        # bounds for fields dont vary normally...
        # Adjust camera dynamically;
        auto_camera_from_bounds(renderView1, bounds)
        # Adjust grid bounds dynamically, should happen automaically..
        # renderView1.AxesGrid.UseCustomBounds = 1
        # renderView1.AxesGrid.CustomBounds = bounds
        

        vmin, vmax = density_array_info.GetComponentRange(-1)
        nice_min, nice_max = nice_bounds_sym(vmin, vmax)
        # Update color and opacity transfer function
        # print_info("==RESCALING COLOUR AND OPACITY BAR")
        densityLUT.RescaleTransferFunction(nice_min, nice_max)
        densityPWF.RescaleTransferFunction(nice_min, nice_max)

        # print_info(f"Updated Opacity Map at cycle {info.cycle}: {densityPWF.Points}")