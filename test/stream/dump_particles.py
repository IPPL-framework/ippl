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
renderView1.ViewSize = [877, 811]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [9.804888932121028, 10.012698468217557, 10.017046030145888]
renderView1.HiddenLineRemoval = 1
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [42.01243955103898, 42.22024908713551, 42.224596649063855]
renderView1.CameraFocalPoint = [9.804888932121028, 10.012698468217557, 10.017046030145888]
renderView1.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 14.438249951766423
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(877, 811)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Partitioned Dataset Reader'
ippl_particle = PVTrivialProducer(registrationName='ippl_particle')

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
velocityLUT.RGBPoints = [0.050641224585373915, 0.231373, 0.298039, 0.752941, 2.3924284143906274, 0.865003, 0.865003, 0.865003, 4.734215604195881, 0.705882, 0.0156863, 0.14902]
velocityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')
velocityPWF.Points = [0.050641224585373915, 0.0, 0.5, 0.0, 4.734215604195881, 1.0, 0.5, 0.0]
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
ippl_particleDisplay.ScaleFactor = 1.9956595386576226
ippl_particleDisplay.SelectScaleArray = 'None'
ippl_particleDisplay.GlyphType = 'Arrow'
ippl_particleDisplay.GlyphTableIndexArray = 'None'
ippl_particleDisplay.GaussianRadius = 0.09978297693288113
ippl_particleDisplay.SetScaleArray = ['POINTS', 'charge']
ippl_particleDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ippl_particleDisplay.OpacityArray = ['POINTS', 'charge']
ippl_particleDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ippl_particleDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_particleDisplay.PolarAxes = 'PolarAxesRepresentation'
ippl_particleDisplay.ScalarOpacityFunction = velocityPWF
ippl_particleDisplay.ScalarOpacityUnitDistance = 1.3403283950605853
ippl_particleDisplay.OpacityArrayName = ['POINTS', 'charge']
ippl_particleDisplay.SelectInputVectors = ['POINTS', 'position']
ippl_particleDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
ippl_particleDisplay.ScaleTransferFunction.Points = [-0.15625, 0.0, 0.5, 0.0, -0.156219482421875, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
ippl_particleDisplay.OpacityTransferFunction.Points = [-0.15625, 0.0, 0.5, 0.0, -0.156219482421875, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.Title = 'velocity'
velocityLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
velocityLUTColorBar.Visibility = 1

# show color legend
ippl_particleDisplay.SetScalarBarVisibility(renderView1, True)

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
pNG1.Writer.FileName = 'Particles_{timestep:06d}{camera}.png'
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
options.ExtractsOutputDirectory = 'datasets_particles'
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
