# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

from paraview import print_info
import math


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

def compute_bounding_box_scale(bounds):
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    diagonal = math.sqrt(dx * dx + dy * dy + dz * dz)
    return diagonal

# Helper function to set the camera
def set_camera(view, position=None, focal_point=None, view_up=None, parallel_scale=None):
    if position is not None:
        view.CameraPosition = position
    if focal_point is not None:
        view.CameraFocalPoint = focal_point
        renderView1.CameraFocalDisk = 1.0
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
# setup views used in the visualization
# ----------------------------------------------------------------
print_info("==='%s'=============================="[0:30]+">",__name__)
SetActiveView(None)

ippl_vector_field = PVTrivialProducer(registrationName='ippl_E')
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2000, 1500]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.StereoType = 'Crystal Eyes'
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.AxesGrid.Visibility = 1
materialLibrary1 = GetMaterialLibrary()
renderView1.OSPRayMaterialLibrary = materialLibrary1

# change background ...
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
# renderView1.Background2 = [0.0, 0.6666666666666666, 1.0]
# renderView1.Background = [0.0, 0.0, 0.4980392156862745]


ippl_vector_info = ippl_vector_field.GetDataInformation()
bounds = ippl_vector_info.GetBounds()
auto_camera_from_bounds(renderView1, bounds)
diag = compute_bounding_box_scale(bounds)

SetActiveView(renderView1)


# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------
glyph1 = Glyph(registrationName='Glyph1', Input=ippl_vector_field, GlyphType='Arrow')
glyph1.OrientationArray = ['CELLS', 'electrostatic']
glyph1.GlyphTransform = 'Transform2'
glyph1.ScaleFactor = diag/30


#MORE....
#  ## dimension dependent scaling... depends on data ...
# glyph1.ScaleFactor = [1.0, 0.5, 2.0] 
## field proportional scaling, sometimes turns graph illegible
# glyph1.ScaleArray = ['CELLS', 'electrostatic']

# init 'Arrow' selected for 'GlyphType' ... stay at defaults...
# print(f"Default TipResolution: {glyph1.GlyphType.TipResolution}")
# print(f"Default TipLength: {glyph1.GlyphType.TipLength}")
# print(f"Default ShaftResolution: {glyph1.GlyphType.ShaftResolution}")
# print(f"Default ShaftRadius: {glyph1.GlyphType.ShaftRadius}")
    
# print(f"Glyph Type: {glyph1.GlyphType}")
# print(f"Orientation Array: {glyph1.OrientationArray}")
# print(f"Scale Factor: {glyph1.ScaleFactor}")
# print(f"Opacity Transfer Function Points: {fieldStrengthPWF.Points}")
# print(f"Color Transfer Function Range: {fieldStrengthLUT.RGBPoints}")





fieldStrengthTF2D = GetTransferFunction2D('electrostatic')
fieldStrengthTF2D.ScalarRangeInitialized = 1
fieldStrengthTF2D.Range = [0.00, 2.00, 0.0, 1.0]

fieldStrengthLUT = GetColorTransferFunction('electrostatic')
fieldStrengthLUT.TransferFunction2D = fieldStrengthTF2D
fieldStrengthLUT.ScalarRangeInitialized = 1
fieldStrengthLUT.RGBPoints = [0.00, 0.231373, 0.298039, 0.752941, 
                              1.00, 0.865003, 0.865003, 0.865003, 
                              2.00, 0.705882, 0.0156863, 0.14902]
fieldStrengthLUTColorBar = GetScalarBar(fieldStrengthLUT, renderView1)
fieldStrengthLUTColorBar.Title = 'fieldStrength'
fieldStrengthLUTColorBar.ComponentTitle = 'Magnitude'
fieldStrengthLUTColorBar.Visibility = 1

fieldStrengthLUT.EnableOpacityMapping = True

fieldStrengthPWF = GetOpacityTransferFunction('electrostatic')
fieldStrengthPWF.ScalarRangeInitialized = 1
fieldStrengthPWF.Points = [0.00, 0.0, 0.5, 0.0, 
                           0.50, 0.2, 0.5, 0.0, 
                           2.00, 1.0, 0.5, 0.0]



glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
glyph1Display.Representation = 'Surface'
glyph1Display.LookupTable = fieldStrengthLUT
glyph1Display.ColorArrayName = ['POINTS', 'electrostatic']
glyph1Display.OpacityTransferFunction = fieldStrengthPWF
glyph1Display.DataAxesGrid = 'Grid Axes Representation'
glyph1Display.SetScalarBarVisibility(renderView1, True)



# # ------------------------------------------------------------
# setup extractors
# --------------------------------------------------------------
pNG3 = CreateExtractor('PNG', renderView1, registrationName='PNG3')
pNG3.Trigger = 'Time Step'
pNG3.Trigger.Frequency = 1
pNG3.Writer.FileName = 'VectorField_{timestep:06d}{camera}.png'
pNG3.Writer.ImageResolution = [1247, 1176]
pNG3.Writer.Format = 'PNG'
SetActiveSource(glyph1)



# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
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


    global ippl_vector_field
    global renderView1
    global fieldStrengthLUT
    global fieldStrengthPWF



    if info.cycle % 10 == 0:
        vector_info = ippl_vector_field.GetDataInformation()
        bounds = vector_info.GetBounds()
        cell_data_info = vector_info.GetCellDataInformation()
        fieldStrength_array_info = cell_data_info.GetArrayInformation('electrostatic')


        # bounds for fields dont vary normally,butmight ...-> Adjust camera dynamically;
        auto_camera_from_bounds(renderView1, bounds)
        # Adjust grid bounds dynamically, should happen automaically even if there are changes??...
        # renderView1.AxesGrid.UseCustomBounds = 1
        # renderView1.AxesGrid.CustomBounds = bounds
        vmin, vmax = fieldStrength_array_info.GetComponentRange(-1) # magnitude ...
        nice_min, nice_max = nice_bounds(vmin, vmax)
        # # Update color and opacity transfer function
        fieldStrengthLUT.RescaleTransferFunction(nice_min, nice_max)
        fieldStrengthPWF.RescaleTransferFunction(nice_min, nice_max)

