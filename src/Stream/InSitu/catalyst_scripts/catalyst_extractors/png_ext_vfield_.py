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
print("===EXECUTING CATALYSt VFIELD EXTRACTOR2======>")
print("====================================>")


# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1247, 1176]
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.CenterOfRotation = [9.977132737636566, 9.960369408130646, 10.014306247234344]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [45.25156805313023, 36.870151913543715, 45.686363221485244]
renderView1.CameraFocalPoint = [9.977132737636566, 9.960369408130646, 10.014306247234344]
renderView1.CameraViewUp = [-0.13205443629762095, 0.8497247521302609, -0.5104208767980444]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 14.73432485141569
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
renderView1.Background2 = [0.0, 0.6666666666666666, 1.0]
renderView1.Background = [0.0, 0.0, 0.4980392156862745]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# init the 'Grid Axes 3D Actor' selected for 'AxesGrid'
renderView1.AxesGrid.Visibility = 1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1247, 1176)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PV Trivial Producer'
ippl_E = PVTrivialProducer(registrationName='ippl_E')

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=ippl_E,
    GlyphType='Arrow')
glyph1.OrientationArray = ['CELLS', 'electrostatic']
glyph1.ScaleArray = ['CELLS', 'None']
# glyph1.ScaleArray = ['CELLS', 'electrostatic']
# glyph1.ScaleFactor = 0.5
glyph1.GlyphTransform = 'Transform2'

# init the 'Arrow' selected for 'GlyphType'
glyph1.GlyphType.TipResolution = 20
glyph1.GlyphType.TipLength = 0.29
glyph1.GlyphType.ShaftResolution = 8
glyph1.GlyphType.ShaftRadius = 0.02

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from glyph1
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

# get 2D transfer function for 'electrostatic'
electrostaticTF2D = GetTransferFunction2D('electrostatic')
electrostaticTF2D.ScalarRangeInitialized = 1
electrostaticTF2D.Range = [0.3405590802851601, 2.3380367305185823, 0.0, 1.0]

# get color transfer function/color map for 'electrostatic'
electrostaticLUT = GetColorTransferFunction('electrostatic')
electrostaticLUT.TransferFunction2D = electrostaticTF2D
electrostaticLUT.RGBPoints = [0.3405590802851601, 0.231373, 0.298039, 0.752941, 1.3392979054018714, 0.865003, 0.865003, 0.865003, 2.3380367305185823, 0.705882, 0.0156863, 0.14902]
electrostaticLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = ['POINTS', 'electrostatic']
glyph1Display.LookupTable = electrostaticLUT
glyph1Display.SelectTCoordArray = 'None'
glyph1Display.SelectNormalArray = 'None'
glyph1Display.SelectTangentArray = 'None'
glyph1Display.OSPRayScaleFunction = 'Piecewise Function'
glyph1Display.Assembly = 'Hierarchy'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 2.2
glyph1Display.SelectScaleArray = 'None'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'None'
glyph1Display.GaussianRadius = 0.11
glyph1Display.SetScaleArray = [None, '']
glyph1Display.ScaleTransferFunction = 'Piecewise Function'
glyph1Display.OpacityArray = [None, '']
glyph1Display.OpacityTransferFunction = 'Piecewise Function'
glyph1Display.DataAxesGrid = 'Grid Axes Representation'
glyph1Display.PolarAxes = 'Polar Axes Representation'
glyph1Display.SelectInputVectors = [None, '']
glyph1Display.WriteLog = ''

# setup the color legend parameters for each legend in this view

# get color legend/bar for electrostaticLUT in view renderView1
electrostaticLUTColorBar = GetScalarBar(electrostaticLUT, renderView1)
electrostaticLUTColorBar.Title = 'electrostatic'
electrostaticLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
electrostaticLUTColorBar.Visibility = 1

# show color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'electrostatic'
electrostaticPWF = GetOpacityTransferFunction('electrostatic')
electrostaticPWF.Points = [0.3405590802851601, 0.0, 0.5, 0.0, 2.3380367305185823, 1.0, 0.5, 0.0]
electrostaticPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0

# ----------------------------------------------------------------
# setup extractors
# --------------------------------------------------------------
# create extractor
pNG3 = CreateExtractor('PNG', renderView1, registrationName='PNG3')
# trace defaults for the extractor.
pNG3.Trigger = 'Time Step'

# init the 'Time Step' selected for 'Trigger'
pNG3.Trigger.Frequency = 1

# init the 'PNG' selected for 'Writer'
pNG3.Writer.FileName = 'VectorField_{timestep:06d}{camera}.png'
pNG3.Writer.ImageResolution = [1247, 1176]
pNG3.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(glyph1)
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
