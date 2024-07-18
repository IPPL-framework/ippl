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
renderView1.ViewSize = [795, 749]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [10.0, 10.0, 10.0]
renderView1.HiddenLineRemoval = 1
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [48.63703305156274, 48.63703305156273, 48.637033051562746]
renderView1.CameraFocalPoint = [10.0, 10.0, 10.0]
renderView1.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 17.320508075688775
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(795, 749)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Partitioned Dataset Reader'
ippl_field = PVTrivialProducer(registrationName='ippl_field')

# create a new 'Ghost Cells Generator'
ghostCellsGenerator1 = GhostCellsGenerator(registrationName='GhostCellsGenerator1', Input=ippl_field)

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=ghostCellsGenerator1)
contour1.ContourBy = ['POINTS', 'density']
contour1.Isosurfaces = [0.0, -9.560476096448486, -9.057293144003829, -8.554110191559172, -8.050927239114515, -7.547744286669857, -7.0445613342252, -6.541378381780543, -6.038195429335886, -5.535012476891229, -5.031829524446572, -4.5286465720019144, -4.025463619557257, -3.5222806671126, -3.019097714667943, -2.515914762223286, -2.0127318097786286, -1.5095488573339715, -1.0063659048893143, -0.5031829524446572, 0.0]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=contour1)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'density']
clip1.Value = -3.773872137069702

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [10.619441449642181, 10.038045406341553, 10.0]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [10.619441449642181, 10.038045406341553, 10.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from ippl_field
ippl_fieldDisplay = Show(ippl_field, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
ippl_fieldDisplay.Representation = 'Outline'
ippl_fieldDisplay.ColorArrayName = [None, '']
ippl_fieldDisplay.SelectTCoordArray = 'None'
ippl_fieldDisplay.SelectNormalArray = 'None'
ippl_fieldDisplay.SelectTangentArray = 'None'
ippl_fieldDisplay.OSPRayScaleArray = 'density'
ippl_fieldDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
ippl_fieldDisplay.SelectOrientationVectors = 'None'
ippl_fieldDisplay.ScaleFactor = 2.0
ippl_fieldDisplay.SelectScaleArray = 'None'
ippl_fieldDisplay.GlyphType = 'Arrow'
ippl_fieldDisplay.GlyphTableIndexArray = 'None'
ippl_fieldDisplay.GaussianRadius = 0.1
ippl_fieldDisplay.SetScaleArray = ['POINTS', 'density']
ippl_fieldDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ippl_fieldDisplay.OpacityArray = ['POINTS', 'density']
ippl_fieldDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ippl_fieldDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_fieldDisplay.PolarAxes = 'PolarAxesRepresentation'
ippl_fieldDisplay.ScalarOpacityUnitDistance = 1.0825317547305484
ippl_fieldDisplay.OpacityArrayName = ['CELLS', 'density']
ippl_fieldDisplay.ColorArray2Name = ['CELLS', 'density']
ippl_fieldDisplay.SliceFunction = 'Plane'
ippl_fieldDisplay.Slice = 16
ippl_fieldDisplay.SelectInputVectors = [None, '']
ippl_fieldDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
ippl_fieldDisplay.ScaleTransferFunction.Points = [-9.560476096448486, 0.0, 0.5, 0.0, 0.1953125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
ippl_fieldDisplay.OpacityTransferFunction.Points = [-9.560476096448486, 0.0, 0.5, 0.0, 0.1953125, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
ippl_fieldDisplay.SliceFunction.Origin = [10.0, 10.0, 10.0]

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'density'
densityTF2D = GetTransferFunction2D('density')

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.TransferFunction2D = densityTF2D
densityLUT.RGBPoints = [-7.547744274139404, 0.231373, 0.298039, 0.752941, -3.773872137069702, 0.865003, 0.865003, 0.865003, 0.0, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [-7.547744274139404, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'density']
clip1Display.LookupTable = densityLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'Normals'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleArray = 'density'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 1.851871407032013
clip1Display.SelectScaleArray = 'density'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'density'
clip1Display.GaussianRadius = 0.09259357035160065
clip1Display.SetScaleArray = ['POINTS', 'density']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'density']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = densityPWF
clip1Display.ScalarOpacityUnitDistance = 0.9212026380832151
clip1Display.OpacityArrayName = ['POINTS', 'density']
clip1Display.SelectInputVectors = ['POINTS', 'Normals']
clip1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [-7.547744274139404, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [-7.547744274139404, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for densityLUT in view renderView1
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.Title = 'density'
densityLUTColorBar.ComponentTitle = ''

# set color bar visibility
densityLUTColorBar.Visibility = 1

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

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
pNG1.Writer.FileName = 'Density_Field_{timestep:06d}{camera}.png'
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
options.ExtractsOutputDirectory = 'datasets_density'
options.GenerateCinemaSpecification = 1
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
