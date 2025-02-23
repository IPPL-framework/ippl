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
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [936, 893]
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.CenterOfRotation = [0.5, 0.5, 0.5]
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraPosition = [5.357750529134775, 4.872085250907219, 5.280823925206746]
renderView3.CameraFocalPoint = [0.5, 0.5, 0.5]
renderView3.CameraViewUp = [-0.11343363691620603, 0.7878938103825996, -0.6052736187681985]
renderView3.CameraFocalDisk = 1.0
renderView3.CameraParallelScale = 1.7320508075688772
renderView3.BackEnd = 'OSPRay raycaster'
renderView3.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.SplitHorizontal(0, 0.500000)
layout1.AssignView(2, renderView3)
layout1.SetSize(1873, 893)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView3)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
# extractippl_field = PVTrivialProducer(registrationName='Extract: ippl_field')
extractippl_field = PVTrivialProducer(registrationName='ippl_field')

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=extractippl_field)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['CELLS', 'density']
clip1.Value = 0.5

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.5, 0.5, 0.5]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [0.5, 0.5, 0.5]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView3'
# ----------------------------------------------------------------

# show data from extractippl_field
extractippl_fieldDisplay = Show(extractippl_field, renderView3, 'UniformGridRepresentation')

# get 2D transfer function for 'density'
densityTF2D = GetTransferFunction2D('density')
densityTF2D.ScalarRangeInitialized = 1

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.TransferFunction2D = densityTF2D
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
extractippl_fieldDisplay.Representation = 'Outline'
extractippl_fieldDisplay.ColorArrayName = ['CELLS', 'density']
extractippl_fieldDisplay.LookupTable = densityLUT
extractippl_fieldDisplay.SelectTCoordArray = 'None'
extractippl_fieldDisplay.SelectNormalArray = 'None'
extractippl_fieldDisplay.SelectTangentArray = 'None'
extractippl_fieldDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
extractippl_fieldDisplay.SelectOrientationVectors = 'None'
extractippl_fieldDisplay.ScaleFactor = 0.2
extractippl_fieldDisplay.SelectScaleArray = 'None'
extractippl_fieldDisplay.GlyphType = 'Arrow'
extractippl_fieldDisplay.GlyphTableIndexArray = 'None'
extractippl_fieldDisplay.GaussianRadius = 0.01
extractippl_fieldDisplay.SetScaleArray = [None, '']
extractippl_fieldDisplay.ScaleTransferFunction = 'PiecewiseFunction'
extractippl_fieldDisplay.OpacityArray = [None, '']
extractippl_fieldDisplay.OpacityTransferFunction = 'PiecewiseFunction'
extractippl_fieldDisplay.DataAxesGrid = 'GridAxesRepresentation'
extractippl_fieldDisplay.PolarAxes = 'PolarAxesRepresentation'
extractippl_fieldDisplay.ScalarOpacityUnitDistance = 0.8660254037844386
extractippl_fieldDisplay.ScalarOpacityFunction = densityPWF
extractippl_fieldDisplay.TransferFunction2D = densityTF2D
extractippl_fieldDisplay.OpacityArrayName = ['CELLS', 'density']
extractippl_fieldDisplay.ColorArray2Name = ['CELLS', 'density']
extractippl_fieldDisplay.SliceFunction = 'Plane'
extractippl_fieldDisplay.Slice = 2
extractippl_fieldDisplay.SelectInputVectors = [None, '']
extractippl_fieldDisplay.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
extractippl_fieldDisplay.SliceFunction.Origin = [0.5, 0.5, 0.5]

# show data from clip1
clip1Display = Show(clip1, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['CELLS', 'density']
clip1Display.LookupTable = densityLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 0.2
clip1Display.SelectScaleArray = 'None'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'None'
clip1Display.GaussianRadius = 0.01
clip1Display.SetScaleArray = [None, '']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = [None, '']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = densityPWF
clip1Display.ScalarOpacityUnitDistance = 0.9449407874211548
clip1Display.OpacityArrayName = ['CELLS', 'density']
clip1Display.SelectInputVectors = [None, '']
clip1Display.WriteLog = ''

# setup the color legend parameters for each legend in this view

# get color legend/bar for densityLUT in view renderView3
densityLUTColorBar = GetScalarBar(densityLUT, renderView3)
densityLUTColorBar.WindowLocation = 'Any Location'
densityLUTColorBar.Position = [0.8376068376068376, 0.027995520716685318]
densityLUTColorBar.Title = 'density'
densityLUTColorBar.ComponentTitle = ''
densityLUTColorBar.ScalarBarLength = 0.32999999999999996

# set color bar visibility
densityLUTColorBar.Visibility = 1

# show color legend
extractippl_fieldDisplay.SetScalarBarVisibility(renderView3, True)

# show color legend
clip1Display.SetScalarBarVisibility(renderView3, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView3, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView3_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [936, 893]
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

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
