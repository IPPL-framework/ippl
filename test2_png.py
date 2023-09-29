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
renderView1.ViewSize = [2169, 1139]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [10.0, 10.0, 10.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [47.29585541750492, 44.99428098019204, 53.161098647260886]
renderView1.CameraFocalPoint = [10.0, 10.0, 10.0]
renderView1.CameraViewUp = [-0.33836890260180724, 0.8523744266916093, -0.3987033013083779]
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
layout1.SetSize(2169, 1139)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
ippl_particle = PVTrivialProducer(registrationName='ippl_particle')

# create a new 'PVTrivialProducer'
ippl_field = PVTrivialProducer(registrationName='ippl_field')

# create a new 'Ghost Cells Generator'
ghostCellsGenerator1 = GhostCellsGenerator(registrationName='GhostCellsGenerator1', Input=ippl_field)

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=ghostCellsGenerator1)
cellDatatoPointData1.CellDataArraytoprocess = ['vtkGhostType', 'density', 'vtkGhostType']

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
contour1.ContourBy = ['POINTS', 'density']
contour1.Isosurfaces = [-2.598547365298835, -5.39240723059767, -4.771549482753485, -4.150691734909299, -3.5298339870651136, -2.908976239220928, -2.2881184913767423, -1.667260743532557, -1.0464029956883714, -0.4255452478441857, 0.1953125]
contour1.PointMergeMethod = 'Uniform Binning'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from ippl_particle
ippl_particleDisplay = Show(ippl_particle, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'velocity'
velocityTF2D = GetTransferFunction2D('velocity')

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.TransferFunction2D = velocityTF2D
velocityLUT.RGBPoints = [0.05045547731616287, 0.231373, 0.298039, 0.752941, 2.8507787375282088, 0.865003, 0.865003, 0.865003, 5.651101997740255, 0.705882, 0.0156863, 0.14902]
velocityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')
velocityPWF.Points = [0.05045547731616287, 0.0, 0.5, 0.0, 5.651101997740255, 1.0, 0.5, 0.0]
velocityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
ippl_particleDisplay.Representation = 'Point Gaussian'
ippl_particleDisplay.ColorArrayName = ['POINTS', 'velocity']
ippl_particleDisplay.LookupTable = velocityLUT
ippl_particleDisplay.SelectTCoordArray = 'None'
ippl_particleDisplay.SelectNormalArray = 'None'
ippl_particleDisplay.SelectTangentArray = 'None'
ippl_particleDisplay.OSPRayScaleArray = 'charge'
ippl_particleDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
ippl_particleDisplay.SelectOrientationVectors = 'None'
ippl_particleDisplay.ScaleFactor = 1.9956522745753908
ippl_particleDisplay.SelectScaleArray = 'None'
ippl_particleDisplay.GlyphType = 'Arrow'
ippl_particleDisplay.GlyphTableIndexArray = 'None'
ippl_particleDisplay.GaussianRadius = 0.15
ippl_particleDisplay.SetScaleArray = ['POINTS', 'charge']
ippl_particleDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ippl_particleDisplay.OpacityArray = ['POINTS', 'charge']
ippl_particleDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ippl_particleDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_particleDisplay.PolarAxes = 'PolarAxesRepresentation'
ippl_particleDisplay.ScalarOpacityFunction = velocityPWF
ippl_particleDisplay.ScalarOpacityUnitDistance = 1.3403262389439892
ippl_particleDisplay.OpacityArrayName = ['POINTS', 'charge']
ippl_particleDisplay.SelectInputVectors = ['POINTS', 'velocity']
ippl_particleDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
ippl_particleDisplay.ScaleTransferFunction.Points = [-0.15625, 0.0, 0.5, 0.0, -0.156219482421875, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
ippl_particleDisplay.OpacityTransferFunction.Points = [-0.15625, 0.0, 0.5, 0.0, -0.156219482421875, 1.0, 0.5, 0.0]

# show data from ghostCellsGenerator1
ghostCellsGenerator1Display = Show(ghostCellsGenerator1, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
ghostCellsGenerator1Display.Representation = 'Outline'
ghostCellsGenerator1Display.ColorArrayName = [None, '']
ghostCellsGenerator1Display.SelectTCoordArray = 'None'
ghostCellsGenerator1Display.SelectNormalArray = 'None'
ghostCellsGenerator1Display.SelectTangentArray = 'None'
ghostCellsGenerator1Display.OSPRayScaleArray = 'vtkGhostType'
ghostCellsGenerator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
ghostCellsGenerator1Display.SelectOrientationVectors = 'None'
ghostCellsGenerator1Display.ScaleFactor = 2.0
ghostCellsGenerator1Display.SelectScaleArray = 'None'
ghostCellsGenerator1Display.GlyphType = 'Arrow'
ghostCellsGenerator1Display.GlyphTableIndexArray = 'None'
ghostCellsGenerator1Display.GaussianRadius = 0.1
ghostCellsGenerator1Display.SetScaleArray = ['POINTS', 'vtkGhostType']
ghostCellsGenerator1Display.ScaleTransferFunction = 'PiecewiseFunction'
ghostCellsGenerator1Display.OpacityArray = ['POINTS', 'vtkGhostType']
ghostCellsGenerator1Display.OpacityTransferFunction = 'PiecewiseFunction'
ghostCellsGenerator1Display.DataAxesGrid = 'GridAxesRepresentation'
ghostCellsGenerator1Display.PolarAxes = 'PolarAxesRepresentation'
ghostCellsGenerator1Display.ScalarOpacityUnitDistance = 2.165063509461097
ghostCellsGenerator1Display.OpacityArrayName = ['POINTS', 'vtkGhostType']
ghostCellsGenerator1Display.ColorArray2Name = ['POINTS', 'vtkGhostType']
ghostCellsGenerator1Display.SliceFunction = 'Plane'
ghostCellsGenerator1Display.Slice = 8
ghostCellsGenerator1Display.SelectInputVectors = [None, '']
ghostCellsGenerator1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
ghostCellsGenerator1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
ghostCellsGenerator1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
ghostCellsGenerator1Display.SliceFunction.Origin = [10.0, 10.0, 10.0]

# show data from contour1
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# get 2D transfer function for 'density'
densityTF2D = GetTransferFunction2D('density')
densityTF2D.ScalarRangeInitialized = 1
densityTF2D.Range = [-5.39240723059767, 0.1953125, 0.0, 1.0]

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.TransferFunction2D = densityTF2D
densityLUT.RGBPoints = [-5.962676343529195, 0.231373, 0.298039, 0.752941, -2.8836819217645977, 0.865003, 0.865003, 0.865003, 0.1953125, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = ['POINTS', 'density']
contour1Display.LookupTable = densityLUT
contour1Display.Opacity = 0.5
contour1Display.SelectTCoordArray = 'None'
contour1Display.SelectNormalArray = 'Normals'
contour1Display.SelectTangentArray = 'None'
contour1Display.OSPRayScaleArray = 'density'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = 0.9665217399597168
contour1Display.SelectScaleArray = 'density'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'density'
contour1Display.GaussianRadius = 0.048326086997985844
contour1Display.SetScaleArray = ['POINTS', 'density']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'density']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'
contour1Display.SelectInputVectors = ['POINTS', 'Normals']
contour1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [-2.5985474586486816, 0.0, 0.5, 0.0, -2.5980591773986816, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [-2.5985474586486816, 0.0, 0.5, 0.0, -2.5980591773986816, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for densityLUT in view renderView1
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.Title = 'density'
densityLUTColorBar.ComponentTitle = ''

# set color bar visibility
densityLUTColorBar.Visibility = 1

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.WindowLocation = 'Upper Right Corner'
velocityLUTColorBar.Title = 'velocity'
velocityLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
velocityLUTColorBar.Visibility = 1

# show color legend
ippl_particleDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [-5.962676343529195, 0.0, 0.5, 0.0, 0.1953125, 1.0, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1920, 1080]
pNG1.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(contour1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
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
