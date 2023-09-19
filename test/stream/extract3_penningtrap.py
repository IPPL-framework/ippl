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
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [371, 530]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.CenterOfRotation = [10.0, 10.0, 10.0]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [-42.55244880849627, 53.0225730665183, 75.27577212134747]
renderView2.CameraFocalPoint = [10.0, 10.0, 10.0]
renderView2.CameraViewUp = [0.3331761994480328, 0.8879714244143248, -0.3170179325289973]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 24.810457513824463
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.SplitHorizontal(0, 0.500000)
layout1.AssignView(1, renderView2)
layout1.SetSize(742, 530)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView2)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
ippl_field = PVTrivialProducer(registrationName='ippl_field')

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=ippl_field)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', '']

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [10.0, 10.0, 10.0]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [10.0, 10.0, 10.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from ippl_field
ippl_fieldDisplay = Show(ippl_field, renderView2, 'UniformGridRepresentation')

# get 2D transfer function for 'electrostatic'
electrostaticTF2D = GetTransferFunction2D('electrostatic')
electrostaticTF2D.ScalarRangeInitialized = 1
electrostaticTF2D.Range = [0.0, 1.1757813367477812e-38, 0.0, 1.0]

# get color transfer function/color map for 'electrostatic'
electrostaticLUT = GetColorTransferFunction('electrostatic')
electrostaticLUT.TransferFunction2D = electrostaticTF2D
electrostaticLUT.RGBPoints = [0.07470709971953278, 0.231373, 0.298039, 0.752941, 3.33273624725058, 0.865003, 0.865003, 0.865003, 6.590765394781628, 0.705882, 0.0156863, 0.14902]
electrostaticLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'electrostatic'
electrostaticPWF = GetOpacityTransferFunction('electrostatic')
electrostaticPWF.Points = [0.07470709971953278, 0.0, 0.5, 0.0, 6.590765394781628, 1.0, 0.5, 0.0]
electrostaticPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
ippl_fieldDisplay.Representation = 'Outline'
ippl_fieldDisplay.ColorArrayName = ['CELLS', 'electrostatic']
ippl_fieldDisplay.LookupTable = electrostaticLUT
ippl_fieldDisplay.SelectTCoordArray = 'None'
ippl_fieldDisplay.SelectNormalArray = 'None'
ippl_fieldDisplay.SelectTangentArray = 'None'
ippl_fieldDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
ippl_fieldDisplay.SelectOrientationVectors = 'None'
ippl_fieldDisplay.ScaleFactor = 2.0
ippl_fieldDisplay.SelectScaleArray = 'None'
ippl_fieldDisplay.GlyphType = 'Arrow'
ippl_fieldDisplay.GlyphTableIndexArray = 'None'
ippl_fieldDisplay.GaussianRadius = 0.1
ippl_fieldDisplay.SetScaleArray = [None, '']
ippl_fieldDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ippl_fieldDisplay.OpacityArray = [None, '']
ippl_fieldDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ippl_fieldDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_fieldDisplay.PolarAxes = 'PolarAxesRepresentation'
ippl_fieldDisplay.ScalarOpacityUnitDistance = 2.165063509461097
ippl_fieldDisplay.ScalarOpacityFunction = electrostaticPWF
ippl_fieldDisplay.TransferFunction2D = electrostaticTF2D
ippl_fieldDisplay.OpacityArrayName = ['CELLS', 'electrostatic']
ippl_fieldDisplay.ColorArray2Name = ['CELLS', 'electrostatic']
ippl_fieldDisplay.SliceFunction = 'Plane'
ippl_fieldDisplay.Slice = 8
ippl_fieldDisplay.SelectInputVectors = [None, '']
ippl_fieldDisplay.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
ippl_fieldDisplay.SliceFunction.Origin = [10.0, 10.0, 10.0]

# show data from clip1
clip1Display = Show(clip1, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['CELLS', 'electrostatic']
clip1Display.LookupTable = electrostaticLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 1.7627550836865113
clip1Display.SelectScaleArray = 'None'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'None'
clip1Display.GaussianRadius = 0.08813775418432555
clip1Display.SetScaleArray = [None, '']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = [None, '']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = electrostaticPWF
clip1Display.ScalarOpacityUnitDistance = 2.054619857832192
clip1Display.OpacityArrayName = ['CELLS', 'electrostatic']
clip1Display.SelectInputVectors = [None, '']
clip1Display.WriteLog = ''

# setup the color legend parameters for each legend in this view

# get color legend/bar for electrostaticLUT in view renderView2
electrostaticLUTColorBar = GetScalarBar(electrostaticLUT, renderView2)
electrostaticLUTColorBar.Title = 'electrostatic'
electrostaticLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
electrostaticLUTColorBar.Visibility = 1

# show color legend
ippl_fieldDisplay.SetScalarBarVisibility(renderView2, True)

# show color legend
clip1Display.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView2, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView2_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [371, 530]
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
options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
