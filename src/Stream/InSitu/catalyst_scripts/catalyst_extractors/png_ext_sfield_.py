# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

print("====================================>")
print("===EXECUTING CATALYST SFIELD EXTRACTOR2======>")
print("====================================>")

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1254, 1131]
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.CenterOfRotation = [10.0, 10.0, 10.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [66.57411756836018, 43.09659003165251, 23.50725429984657]
renderView1.CameraFocalPoint = [10.0, 10.0, 10.0]
renderView1.CameraViewUp = [-0.34800051873318505, 0.7965927005782465, -0.4943032554484042]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 17.320508075688775
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1254, 1131)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Partitioned Dataset Reader'
ippl_scalar = PVTrivialProducer(registrationName='ippl_scalar')
# XMLPartitionedDatasetReader(registrationName='ippl_scalar', FileName=['/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000000.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000001.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000002.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000003.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000004.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000005.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000006.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000007.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000008.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000009.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000010.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000011.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000012.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000013.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000014.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000015.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000016.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000017.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000018.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000019.vtpd', '/home/klappi/Documents/Msc/ippl-frk/build/alpine/ippl_catalyst_output/ippl_field_s_000020.vtpd'])

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from ippl_scalar
ippl_scalarDisplay = Show(ippl_scalar, renderView1, 'UniformGridRepresentation')

# get 2D transfer function for 'density'
densityTF2D = GetTransferFunction2D('density')
densityTF2D.ScalarRangeInitialized = 1
densityTF2D.Range = [-1.8423090006664276, 1.9375303235980603, 0.0, 1.0]

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.TransferFunction2D = densityTF2D
densityLUT.RGBPoints = [-1.8423090006664276, 0.231373, 0.298039, 0.752941, 0.047610661465816495, 0.865003, 0.865003, 0.865003, 1.9375303235980605, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [-1.8423090006664276, 1.0, 0.5, 0.0, -1.166914463043213, 0.3080357313156128, 0.5, 0.0, -0.05, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.05, 0.0, 0.5, 0.0, 1.1554945707321167, 0.3482142984867096, 0.5, 0.0, 1.9375303235980605, 1.0, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
ippl_scalarDisplay.Representation = 'Volume'
ippl_scalarDisplay.ColorArrayName = ['CELLS', 'density']
ippl_scalarDisplay.LookupTable = densityLUT
ippl_scalarDisplay.SelectTCoordArray = 'None'
ippl_scalarDisplay.SelectNormalArray = 'None'
ippl_scalarDisplay.SelectTangentArray = 'None'
ippl_scalarDisplay.OSPRayScaleFunction = 'Piecewise Function'
ippl_scalarDisplay.Assembly = 'Hierarchy'
ippl_scalarDisplay.SelectOrientationVectors = 'None'
ippl_scalarDisplay.ScaleFactor = 2.0
ippl_scalarDisplay.SelectScaleArray = 'None'
ippl_scalarDisplay.GlyphType = 'Arrow'
ippl_scalarDisplay.GlyphTableIndexArray = 'None'
ippl_scalarDisplay.GaussianRadius = 0.1
ippl_scalarDisplay.SetScaleArray = [None, '']
ippl_scalarDisplay.ScaleTransferFunction = 'Piecewise Function'
ippl_scalarDisplay.OpacityArray = [None, '']
ippl_scalarDisplay.OpacityTransferFunction = 'Piecewise Function'
ippl_scalarDisplay.DataAxesGrid = 'Grid Axes Representation'
ippl_scalarDisplay.PolarAxes = 'Polar Axes Representation'
ippl_scalarDisplay.ScalarOpacityUnitDistance = 4.330127018922194
ippl_scalarDisplay.ScalarOpacityFunction = densityPWF
ippl_scalarDisplay.TransferFunction2D = densityTF2D
ippl_scalarDisplay.OpacityArrayName = ['CELLS', 'density']
ippl_scalarDisplay.ColorArray2Name = ['CELLS', 'density']
ippl_scalarDisplay.SliceFunction = 'Plane'
ippl_scalarDisplay.Slice = 4
ippl_scalarDisplay.SelectInputVectors = [None, '']
ippl_scalarDisplay.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
ippl_scalarDisplay.SliceFunction.Origin = [10.0, 10.0, 10.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for densityLUT in view renderView1
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.Title = 'density'
densityLUTColorBar.ComponentTitle = ''

# set color bar visibility
densityLUTColorBar.Visibility = 1

# show color legend
ippl_scalarDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# # get time animation track
# timeAnimationCue1 = GetTimeTrack()

# # initialize the animation scene

# # get the time-keeper
# timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track
""" 
# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0

 """
# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'Time Step'

pNG1.Trigger.Frequency = 1
# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'ScalarField_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1254, 1131]
pNG1.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(pNG1)
# ----------------------------------------------------------------

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
