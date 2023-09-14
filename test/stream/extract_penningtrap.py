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
renderView1.ViewSize = [1877, 1171]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [10.0, 10.0, 10.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-29.281150563975967, 38.91994832117599, 65.77495370861955]
renderView1.CameraFocalPoint = [14.933355093656326, 7.465538329176106, 5.669943215819909]
renderView1.CameraViewUp = [0.22018111884477137, 0.9214003427041039, -0.3202213037401241]
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
layout1.SetSize(1877, 1171)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Partitioned Dataset Reader'
ippl_field = XMLPartitionedDatasetReader(registrationName='ippl_field', FileName=['/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000000.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000001.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000002.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000003.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000004.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000005.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000006.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000007.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000008.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000009.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000010.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000011.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000012.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000013.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000014.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000015.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000016.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000017.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000018.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000019.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000020.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000021.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000022.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000023.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000024.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000025.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000026.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000027.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000028.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000029.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000030.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000031.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000032.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000033.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000034.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000035.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000036.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000037.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000038.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000039.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000040.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000041.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000042.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000043.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000044.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000045.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000046.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000047.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000048.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000049.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000050.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000051.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000052.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000053.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000054.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000055.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000056.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000057.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000058.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000059.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000060.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000061.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000062.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000063.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000064.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000065.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000066.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000067.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000068.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000069.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000070.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000071.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000072.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000073.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000074.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000075.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000076.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000077.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000078.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000079.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000080.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000081.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000082.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000083.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000084.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000085.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000086.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000087.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000088.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000089.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000090.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000091.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000092.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000093.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000094.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000095.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000096.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000097.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000098.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000099.vtpd', '/p/project/ccstma/schurk1/in-situ/ippl_cuda/ippl/datasets-fine/ippl_field_000100.vtpd'])

# create a new 'Ghost Cells Generator'
ghostCellsGenerator1 = GhostCellsGenerator(registrationName='GhostCellsGenerator1', Input=ippl_field)

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=ghostCellsGenerator1)
cellDatatoPointData1.CellDataArraytoprocess = ['density']

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
contour1.ContourBy = ['POINTS', 'density']
contour1.Isosurfaces = [-10.925284203191916, -9.766919291726147, -8.60855438026038, -7.450189468794611, -6.291824557328843, -5.133459645863074, -3.975094734397306, -2.8167298229315367, -1.6583649114657693, -0.5000000000000018]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=contour1)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'density']
clip1.Value = -5.133459568023682

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [9.98001554608345, 10.002425193786621, 9.999979734420776]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [9.98001554608345, 10.002425193786621, 9.999979734420776]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from ippl_field
ippl_fieldDisplay = Show(ippl_field, renderView1, 'UniformGridRepresentation')

# get 2D transfer function for 'density'
densityTF2D = GetTransferFunction2D('density')
densityTF2D.ScalarRangeInitialized = 1
densityTF2D.Range = [-16.259595698935676, 0.0, 0.0, 1.0]

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.TransferFunction2D = densityTF2D
densityLUT.RGBPoints = [-16.259595698935676, 0.231373, 0.298039, 0.752941, -8.129797849467838, 0.865003, 0.865003, 0.865003, 0.0, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [-16.259595698935676, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
ippl_fieldDisplay.Representation = 'Outline'
ippl_fieldDisplay.ColorArrayName = ['CELLS', 'density']
ippl_fieldDisplay.LookupTable = densityLUT
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
ippl_fieldDisplay.ScalarOpacityUnitDistance = 0.5412658773652742
ippl_fieldDisplay.ScalarOpacityFunction = densityPWF
ippl_fieldDisplay.TransferFunction2D = densityTF2D
ippl_fieldDisplay.OpacityArrayName = ['CELLS', 'density']
ippl_fieldDisplay.ColorArray2Name = ['CELLS', 'density']
ippl_fieldDisplay.SliceFunction = 'Plane'
ippl_fieldDisplay.Slice = 32
ippl_fieldDisplay.SelectInputVectors = [None, '']
ippl_fieldDisplay.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
ippl_fieldDisplay.SliceFunction.Origin = [10.0, 10.0, 10.0]

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

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
clip1Display.ScaleFactor = 1.8501425802707674
clip1Display.SelectScaleArray = 'density'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'density'
clip1Display.GaussianRadius = 0.09250712901353836
clip1Display.SetScaleArray = ['POINTS', 'density']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'density']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = densityPWF
clip1Display.ScalarOpacityUnitDistance = 0.7274508700056125
clip1Display.OpacityArrayName = ['POINTS', 'density']
clip1Display.SelectInputVectors = ['POINTS', 'Normals']
clip1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [-9.766919136047363, 0.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [-9.766919136047363, 0.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for densityLUT in view renderView1
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.WindowLocation = 'Upper Right Corner'
densityLUTColorBar.Title = 'density'
densityLUTColorBar.ComponentTitle = ''

# set color bar visibility
densityLUTColorBar.Visibility = 1

# show color legend
ippl_fieldDisplay.SetScalarBarVisibility(renderView1, True)

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
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1877, 1171]
pNG1.Writer.Format = 'PNG'
pNG1.Writer.ResetDisplay = 1

# create extractor
vTPD1 = CreateExtractor('VTPD', clip1, registrationName='VTPD1')
# trace defaults for the extractor.
vTPD1.Trigger = 'TimeStep'

# init the 'VTPD' selected for 'Writer'
vTPD1.Writer.FileName = 'Clip1_{timestep:06d}.vtpd'

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
