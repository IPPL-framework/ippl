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
renderView1.ViewSize = [1199, 917]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [10.0, 10.0, 10.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-36.185896294276546, 67.84051131760981, 79.13441790882564]
renderView1.CameraFocalPoint = [10.0, 10.0, 10.0]
renderView1.CameraViewUp = [0.2886814022290065, 0.8204174596703359, -0.49353646255891515]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 26.738898830650854
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1199, 917)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
ippl_field = PVTrivialProducer(registrationName='ippl_field')

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=ippl_field)
cellDatatoPointData1.CellDataArraytoprocess = ['density']

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
contour1.ContourBy = ['POINTS', 'density']
contour1.Isosurfaces = [-2.953365099609886, -5.906730199219772, -5.305982399306464, -4.705234599393156, -4.104486799479848, -3.50373899956654, -2.902991199653232, -2.302243399739924, -1.7014955998266164, -1.1007477999133082, -0.5]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=contour1)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'density']
clip1.Value = -2.90299129486084

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [9.766819953918457, 9.9811110496521, 10.671213865280151]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [9.766819953918457, 9.9811110496521, 10.671213865280151]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from cellDatatoPointData1
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
cellDatatoPointData1Display.Representation = 'Outline'
cellDatatoPointData1Display.ColorArrayName = [None, '']
cellDatatoPointData1Display.SelectTCoordArray = 'None'
cellDatatoPointData1Display.SelectNormalArray = 'None'
cellDatatoPointData1Display.SelectTangentArray = 'None'
cellDatatoPointData1Display.OSPRayScaleArray = 'density'
cellDatatoPointData1Display.OSPRayScaleFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.SelectOrientationVectors = 'None'
cellDatatoPointData1Display.ScaleFactor = 2.0
cellDatatoPointData1Display.SelectScaleArray = 'None'
cellDatatoPointData1Display.GlyphType = 'Arrow'
cellDatatoPointData1Display.GlyphTableIndexArray = 'None'
cellDatatoPointData1Display.GaussianRadius = 0.1
cellDatatoPointData1Display.SetScaleArray = ['POINTS', 'density']
cellDatatoPointData1Display.ScaleTransferFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.OpacityArray = ['POINTS', 'density']
cellDatatoPointData1Display.OpacityTransferFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.DataAxesGrid = 'GridAxesRepresentation'
cellDatatoPointData1Display.PolarAxes = 'PolarAxesRepresentation'
cellDatatoPointData1Display.ScalarOpacityUnitDistance = 2.165063509461097
cellDatatoPointData1Display.OpacityArrayName = ['POINTS', 'density']
cellDatatoPointData1Display.ColorArray2Name = ['POINTS', 'density']
cellDatatoPointData1Display.SliceFunction = 'Plane'
cellDatatoPointData1Display.Slice = 8
cellDatatoPointData1Display.SelectInputVectors = [None, '']
cellDatatoPointData1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cellDatatoPointData1Display.ScaleTransferFunction.Points = [-5.906730199219772, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cellDatatoPointData1Display.OpacityTransferFunction.Points = [-5.906730199219772, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
cellDatatoPointData1Display.SliceFunction.Origin = [10.0, 10.0, 10.0]

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'density'
densityTF2D = GetTransferFunction2D('density')
densityTF2D.ScalarRangeInitialized = 1
densityTF2D.Range = [-7.111382323777126, 0.0, 0.0, 1.0]

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.TransferFunction2D = densityTF2D
densityLUT.RGBPoints = [-5.30598258972168, 0.231373, 0.298039, 0.752941, -2.90299129486084, 0.865003, 0.865003, 0.865003, -0.5, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [-5.30598258972168, 0.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0]
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
clip1Display.ScaleFactor = 1.5288142442703248
clip1Display.SelectScaleArray = 'density'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'density'
clip1Display.GaussianRadius = 0.07644071221351624
clip1Display.SetScaleArray = ['POINTS', 'density']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'density']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = densityPWF
clip1Display.ScalarOpacityUnitDistance = 1.5107155620204928
clip1Display.OpacityArrayName = ['POINTS', 'density']
clip1Display.SelectInputVectors = ['POINTS', 'Normals']
clip1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [-5.30598258972168, 0.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [-5.30598258972168, 0.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0]

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
vTPD1 = CreateExtractor('VTPD', ippl_field, registrationName='VTPD1')
# trace defaults for the extractor.
vTPD1.Trigger = 'TimeStep'

# init the 'VTPD' selected for 'Writer'
vTPD1.Writer.FileName = 'ippl_field_{timestep:06d}.vtpd'

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1199, 917]
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
