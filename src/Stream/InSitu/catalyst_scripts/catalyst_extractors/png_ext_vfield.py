"""! \file png_ext_vfield.py
\brief Catalyst PNG extractor for 3D vector fields (e.g., ippl::Field<Vector<T,3>,3>).
\details Uses glyph-based rendering with adaptive camera and magnitude-driven
color/opacity. Driven by pipeline_default.py and compatible with Catalyst Live.
"""

# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0


########################################################
######################################################## 
# PNG extractor script for paraview catalyst. 
# Visualizes 3D vector fields. eg:
# ippl::field<ippl::Vector<double, 3>,3> 
# ippl::field<ippl::Vector<float, 3>,3> 
# 
# Currently hard coded to rely on attributes:
# - 'position'
# - label
# Is adaptive: Attempts to set Camera Angle and colouring 
# of the glyph (dependent on the fieldStrength / -magnitude of
# electrostatic attribute) adaptive to 
# current frame, range and scale (every 10'th step).
# 
# 
# 
# Relies on pipeline_default.py to update pipeline else might
# cause errors (i think)
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
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 12
from paraview.simple import *
from paraview.simple import (
    PVTrivialProducer,
    CreateView,
    GetMaterialLibrary,
    Show,
    Glyph,
    GetTransferFunction2D,
    GetColorTransferFunction,
    GetScalarBar,
    GetOpacityTransferFunction,
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
from catalystSubroutines import (
    nice_bounds,
    set_camera,
    auto_camera_from_bounds,
    compute_bounding_box_scale,
    get_global_range,
    get_global_spatial_bounds,
    hide_source_from_gui 
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
parser.add_argument("--label", default="DEFAULT_CHANNEL", help="Needed to correctly setup association between script name and conduti channel.")
parser.add_argument("--channel_name", default="DEFAULT_CHANNEL", help="Needed to correctly setup association between script name and conduti channel.")
parser.add_argument("--experiment_name", default="_", help="Needed to correctly for safe folder.")
parser.add_argument("--verbosity", type=int, default="1", help="Communicate the catalyst Output Level from the simulation")

parsed = parser.parse_args(arg_list)
label = parsed.label
exp_string = parsed.experiment_name
verbosity = parsed.verbosity
cname  = parsed.channel_name
print_info_("_global__scope__()::" + parsed.channel_name)
# ----------------------------------------------------------------
# create a new 'XML Partitioned Dataset Reader'
# ----------------------------------------------------------------
ippl_vector_field = PVTrivialProducer(registrationName = parsed.channel_name)
hide_source_from_gui(ippl_vector_field)
# ----------------------------------------------------------------
# setup visualisation view for extraction pipeline in renderview1
# ----------------------------------------------------------------
view_name = f"View_{cname}"
renderView1 = CreateView('RenderView', registrationName=view_name)
# renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2000, 1500]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.StereoType = 'Crystal Eyes'
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.AxesGrid.Visibility = 1
materialLibrary1 = GetMaterialLibrary()
renderView1.OSPRayMaterialLibrary = materialLibrary1

renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
# renderView1.Background2 = [0.0, 0.6666666666666666, 1.0]
# renderView1.Background = [0.0, 0.0, 0.4980392156862750]
SetActiveView(renderView1)
# ----------------------------------------------------------------
# Initial adaptive Camera set
# ----------------------------------------------------------------
ippl_vector_info = ippl_vector_field.GetDataInformation()
local_bounds = ippl_vector_info.GetBounds()
global_bounds = get_global_spatial_bounds(local_bounds)
auto_camera_from_bounds(renderView1, global_bounds)
# needed for glyph scale ... use global diagonal
diag = compute_bounding_box_scale(global_bounds)
# ----------------------------------------------------------------
# setup the data processing pipelines, create filter for 
# Vector Field from Vector data
# ----------------------------------------------------------------
glyph1 = Glyph(registrationName='Glyph1', Input=ippl_vector_field, GlyphType='Arrow')
hide_source_from_gui(glyph1)
glyph1.OrientationArray = ['CELLS', label]
glyph1.GlyphTransform = 'Transform2'
glyph1.ScaleFactor = diag/30
# ----------------------------------------------------------------
# choose Data to visualize to show in renderView1
# ----------------------------------------------------------------
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
# ----------------------------------------------------------------
# setup initial transfer function for colouring and opacity
# ----------------------------------------------------------------
#MORE....
#  ## dimension dependent scaling... depends on data ...
# glyph1.ScaleFactor = [1.0, 0.5, 2.0] 
## field proportional scaling, sometimes turns graph illegible
# glyph1.ScaleArray = ['CELLS', label]

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

fieldStrengthTF2D = GetTransferFunction2D(label)
fieldStrengthTF2D.ScalarRangeInitialized = 1
fieldStrengthTF2D.Range = [0.00, 2.00, 0.0, 1.0]
fieldStrengthLUT = GetColorTransferFunction(label)
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
fieldStrengthPWF = GetOpacityTransferFunction(label)
fieldStrengthPWF.ScalarRangeInitialized = 1
fieldStrengthPWF.Points = [0.00, 0.0, 0.5, 0.0, 
                           0.50, 0.2, 0.5, 0.0, 
                           2.00, 1.0, 0.5, 0.0]
# ----------------------------------------------------------------
# configure displayed data
# ----------------------------------------------------------------
glyph1Display.Representation = 'Surface'
glyph1Display.LookupTable = fieldStrengthLUT
glyph1Display.ColorArrayName = ['POINTS', label]
# glyph1Display.ColorArrayName = ['CELLS', label]
glyph1Display.OpacityTransferFunction = fieldStrengthPWF
glyph1Display.DataAxesGrid = 'Grid Axes Representation'
glyph1Display.SetScalarBarVisibility(renderView1, True)
# # ------------------------------------------------------------
# setup extractors
# --------------------------------------------------------------
pNG1 = CreateExtractor('PNG', renderView1, registrationName='pNG_'+cname)
pNG1.Trigger = 'Time Step'
pNG1.Trigger.Frequency = 1
pNG1.Writer.FileName = 'VectorField_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1247, 1176]
pNG1.Writer.Format = 'PNG'
SetActiveSource(glyph1)
# ----------------------------------------------------------------
# Catalyst options
# ----------------------------------------------------------------
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

    global glyph1
    global ippl_vector_field
    global fieldStrengthLUT
    global fieldStrengthPWF
    global renderView1
    global pNG1

    ippl_vector_field.UpdatePipeline()
    # glyph1.UpdatePipeline()
    
    # SetActiveView(renderView1)
    # pNG1.UpdateVTKObjects()


    if info.cycle % 10 == 0:
        vector_info = ippl_vector_field.GetDataInformation()
        local_bounds = vector_info.GetBounds()
        global_bounds = get_global_spatial_bounds(local_bounds)
        cell_data_info = vector_info.GetCellDataInformation()
        fieldStrength_array_info = cell_data_info.GetArrayInformation(label)


        # Adjust camera dynamically using global bounds
        auto_camera_from_bounds(renderView1, global_bounds)
        # Adjust grid bounds dynamically, should happen automaically even if there are changes??...
        # renderView1.AxesGrid.UseCustomBounds = 1
        # renderView1.AxesGrid.CustomBounds = bounds
        local_vmin, local_vmax = fieldStrength_array_info.GetComponentRange(-1) # magnitude ...
        gmin, gmax = get_global_range(local_vmin, local_vmax)
        nice_min, nice_max = nice_bounds(gmin, gmax)
        # # Update color and opacity transfer function
        fieldStrengthLUT.RescaleTransferFunction(nice_min, nice_max)
        fieldStrengthPWF.RescaleTransferFunction(nice_min, nice_max)

