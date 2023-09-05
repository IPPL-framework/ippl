# script-version: 2.0
# Catalyst state generated using paraview version 5.11.1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [862, 839]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.5, 0.5, 0.5]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-8.321539336674517, 5.118013528880978, 10.826590265546251]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.5]
renderView1.CameraViewUp = [0.24312654349512303, 0.9458058854041994, -0.21526892711883117]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.7320508075688772

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(862, 839)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Legacy VTK Reader'
scalar_0000vtk = LegacyVTKReader(registrationName='scalar_0000.vtk*', FileNames=['/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0000.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0001.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0002.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0003.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0004.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0005.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0006.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0007.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0008.vtk', '/home/felix/data/juelich/ippl_toolchain/ippl/data/scalar_0009.vtk'])

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=scalar_0000vtk)
cellDatatoPointData1.CellDataArraytoprocess = ['Rho']

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
contour1.ContourBy = ['POINTS', 'Rho']
contour1.Isosurfaces = [0.0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 0.6666666666666666, 0.7777777777777777, 0.8888888888888888, 1.0]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=contour1)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'Rho']
clip1.Value = 0.5555555559694767

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.5, 0.5, 0.5]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [0.5, 0.5, 0.5]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'Rho'
rhoTF2D = GetTransferFunction2D('Rho')

# get color transfer function/color map for 'Rho'
rhoLUT = GetColorTransferFunction('Rho')
rhoLUT.TransferFunction2D = rhoTF2D
rhoLUT.RGBPoints = [0.1111111119389534, 0.231373, 0.298039, 0.752941, 0.5555555559694767, 0.865003, 0.865003, 0.865003, 1.0, 0.705882, 0.0156863, 0.14902]
rhoLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Rho'
rhoPWF = GetOpacityTransferFunction('Rho')
rhoPWF.Points = [0.1111111119389534, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
rhoPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'Rho']
clip1Display.LookupTable = rhoLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'Normals'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleArray = 'Rho'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 0.2
clip1Display.SelectScaleArray = 'Rho'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'Rho'
clip1Display.GaussianRadius = 0.01
clip1Display.SetScaleArray = ['POINTS', 'Rho']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'Rho']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = rhoPWF
clip1Display.ScalarOpacityUnitDistance = 0.49106951261706133
clip1Display.OpacityArrayName = ['POINTS', 'Rho']
clip1Display.SelectInputVectors = ['POINTS', 'Normals']
clip1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [0.1111111119389534, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [0.1111111119389534, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for rhoLUT in view renderView1
rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)
rhoLUTColorBar.Title = 'Rho'
rhoLUTColorBar.ComponentTitle = ''

# set color bar visibility
rhoLUTColorBar.Visibility = 1

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
pNG1 = CreateExtractor('PNG', renderView1, registrationName='ippl_field')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [862, 839]
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
