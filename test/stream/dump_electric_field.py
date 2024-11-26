# script-version: 2.0
# Catalyst state generated using paraview version 5.11.1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [637, 749]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [10.0, 10.0, 10.0]
renderView1.HiddenLineRemoval = 1
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [55.0073453709904, 55.00734537099039, 55.00734537099041]
renderView1.CameraFocalPoint = [10.0, 10.0, 10.0]
renderView1.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 20.365872132952735
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(637, 749)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
ippl_field = PVTrivialProducer(registrationName='ippl_field')

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=ippl_field,
    GlyphType='Arrow')
glyph1.OrientationArray = ['CELLS', 'electrostatic']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.ScaleFactor = 1.1
glyph1.GlyphTransform = 'Transform2'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from ippl_field
ippl_fieldDisplay = Show(ippl_field, renderView1, 'UniformGridRepresentation')

# get 2D transfer function for 'electrostatic'
electrostaticTF2D = GetTransferFunction2D('electrostatic')
electrostaticTF2D.ScalarRangeInitialized = 1
electrostaticTF2D.Range = [0.034176988881870436, 5.3579889614424, 0.0, 1.0]

# get color transfer function/color map for 'electrostatic'
electrostaticLUT = GetColorTransferFunction('electrostatic')
electrostaticLUT.TransferFunction2D = electrostaticTF2D
electrostaticLUT.RGBPoints = [0.034176988881870436, 0.231373, 0.298039, 0.752941, 2.6960829751621356, 0.865003, 0.865003, 0.865003, 5.3579889614424, 0.705882, 0.0156863, 0.14902]
electrostaticLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'electrostatic'
electrostaticPWF = GetOpacityTransferFunction('electrostatic')
electrostaticPWF.Points = [0.034176988881870436, 0.0, 0.5, 0.0, 5.3579889614424, 1.0, 0.5, 0.0]
electrostaticPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
ippl_fieldDisplay.Representation = 'Outline'
ippl_fieldDisplay.ColorArrayName = ['CELLS', 'electrostatic']
ippl_fieldDisplay.LookupTable = electrostaticLUT
ippl_fieldDisplay.SelectTCoordArray = 'None'
ippl_fieldDisplay.SelectNormalArray = 'None'
ippl_fieldDisplay.SelectTangentArray = 'None'
ippl_fieldDisplay.OSPRayScaleArray = 'electrostatic'
ippl_fieldDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
ippl_fieldDisplay.SelectOrientationVectors = 'None'
ippl_fieldDisplay.ScaleFactor = 2.0
ippl_fieldDisplay.SelectScaleArray = 'None'
ippl_fieldDisplay.GlyphType = 'Arrow'
ippl_fieldDisplay.GlyphTableIndexArray = 'None'
ippl_fieldDisplay.GaussianRadius = 0.1
ippl_fieldDisplay.SetScaleArray = ['POINTS', 'electrostatic']
ippl_fieldDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ippl_fieldDisplay.OpacityArray = ['POINTS', 'electrostatic']
ippl_fieldDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ippl_fieldDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_fieldDisplay.PolarAxes = 'PolarAxesRepresentation'
ippl_fieldDisplay.ScalarOpacityUnitDistance = 1.0825317547305484
ippl_fieldDisplay.ScalarOpacityFunction = electrostaticPWF
ippl_fieldDisplay.TransferFunction2D = electrostaticTF2D
ippl_fieldDisplay.OpacityArrayName = ['CELLS', 'electrostatic']
ippl_fieldDisplay.ColorArray2Name = ['CELLS', 'electrostatic']
ippl_fieldDisplay.SliceFunction = 'Plane'
ippl_fieldDisplay.Slice = 16
ippl_fieldDisplay.SelectInputVectors = ['POINTS', 'electrostatic']
ippl_fieldDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
ippl_fieldDisplay.ScaleTransferFunction.Points = [-3.4982625412984754, 0.0, 0.5, 0.0, 3.5997197159102856, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
ippl_fieldDisplay.OpacityTransferFunction.Points = [-3.4982625412984754, 0.0, 0.5, 0.0, 3.5997197159102856, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
ippl_fieldDisplay.SliceFunction.Origin = [10.0, 10.0, 10.0]

# show data from glyph1
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = ['POINTS', 'electrostatic']
glyph1Display.LookupTable = electrostaticLUT
glyph1Display.SelectTCoordArray = 'None'
glyph1Display.SelectNormalArray = 'None'
glyph1Display.SelectTangentArray = 'None'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 2.2
glyph1Display.SelectScaleArray = 'None'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'None'
glyph1Display.GaussianRadius = 0.11
glyph1Display.SetScaleArray = [None, '']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = [None, '']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.PolarAxes = 'PolarAxesRepresentation'
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
ippl_fieldDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'Electric_Field_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [3840, 2160]
pNG1.Writer.TransparentBackground = 1
pNG1.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(pNG1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.ExtractsOutputDirectory = 'datasets_electric'
options.GenerateCinemaSpecification = 1
options.GlobalTrigger = 'TimeStep'
#options.EnableCatalystLive = 1
#options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
