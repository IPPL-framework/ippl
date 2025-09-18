from paraview.simple import *
# script-version: 2.0
## Catalyst state generated using paraview version 5.12.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12
#### import the simple module from the paraview
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()



def particle_png():


    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [1550, 1176]
    renderView1.AxesGrid = 'Grid Axes 3D Actor'
    renderView1.CenterOfRotation = [9.503237689977233, 9.482050288256222, 9.861933138323092]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [38.70076378467479, 38.679576382953755, 39.05945923302062]
    renderView1.CameraFocalPoint = [9.503237689977233, 9.482050288256222, 9.861933138323092]
    renderView1.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 13.088892872246259
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
    layout1.SetSize(1550, 1176)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'PV Trivial Producer'
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
    velocityLUT.RGBPoints = [0.1752655363940977, 0.0, 0.0, 0.0, 4.7435611821988095, 0.901960784314, 0.0, 0.0, 9.311856828003522, 0.901960784314, 0.901960784314, 0.0, 11.596004650905877, 1.0, 1.0, 1.0]
    velocityLUT.ColorSpace = 'RGB'
    velocityLUT.NanColor = [0.0, 0.498039215686, 1.0]
    velocityLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'velocity'
    velocityPWF = GetOpacityTransferFunction('velocity')
    velocityPWF.Points = [0.1752655363940977, 0.0, 0.5, 0.0, 11.596004650905877, 1.0, 0.5, 0.0]
    velocityPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    ippl_particleDisplay.Representation = 'Surface'
    ippl_particleDisplay.ColorArrayName = ['POINTS', 'velocity']
    ippl_particleDisplay.LookupTable = velocityLUT
    ippl_particleDisplay.PointSize = 3.0
    ippl_particleDisplay.SelectTCoordArray = 'None'
    ippl_particleDisplay.SelectNormalArray = 'None'
    ippl_particleDisplay.SelectTangentArray = 'None'
    ippl_particleDisplay.OSPRayScaleArray = 'charge'
    ippl_particleDisplay.OSPRayScaleFunction = 'Piecewise Function'
    ippl_particleDisplay.Assembly = 'Hierarchy'
    ippl_particleDisplay.SelectOrientationVectors = 'None'
    ippl_particleDisplay.ScaleFactor = 1.938351422158595
    ippl_particleDisplay.SelectScaleArray = 'None'
    ippl_particleDisplay.GlyphType = 'Arrow'
    ippl_particleDisplay.GlyphTableIndexArray = 'None'
    ippl_particleDisplay.GaussianRadius = 0.09691757110792974
    ippl_particleDisplay.SetScaleArray = ['POINTS', 'charge']
    ippl_particleDisplay.ScaleTransferFunction = 'Piecewise Function'
    ippl_particleDisplay.OpacityArray = ['POINTS', 'charge']
    ippl_particleDisplay.OpacityTransferFunction = 'Piecewise Function'
    ippl_particleDisplay.DataAxesGrid = 'Grid Axes Representation'
    ippl_particleDisplay.PolarAxes = 'Polar Axes Representation'
    ippl_particleDisplay.ScalarOpacityFunction = velocityPWF
    ippl_particleDisplay.ScalarOpacityUnitDistance = 3.2722232180615647
    ippl_particleDisplay.OpacityArrayName = ['POINTS', 'charge']
    ippl_particleDisplay.SelectInputVectors = ['POINTS', 'position']
    ippl_particleDisplay.WriteLog = ''

    # init the 'Piecewise Function' selected for 'ScaleTransferFunction'
    ippl_particleDisplay.ScaleTransferFunction.Points = [-3.0517578125, 0.0, 0.5, 0.0, -3.05126953125, 1.0, 0.5, 0.0]

    # init the 'Piecewise Function' selected for 'OpacityTransferFunction'
    ippl_particleDisplay.OpacityTransferFunction.Points = [-3.0517578125, 0.0, 0.5, 0.0, -3.05126953125, 1.0, 0.5, 0.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for velocityLUT in view renderView1
    velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
    velocityLUTColorBar.WindowLocation = 'Any Location'
    velocityLUTColorBar.Position = [0.8516129032258065, 0.006802721088435368]
    velocityLUTColorBar.Title = 'velocity'
    velocityLUTColorBar.ComponentTitle = 'Magnitude'
    velocityLUTColorBar.ScalarBarLength = 0.3300000000000002

    # set color bar visibility
    velocityLUTColorBar.Visibility = 1

    # show color legend
    ippl_particleDisplay.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity maps used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup animation scene, tracks and keyframes
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # get the time-keeper
    timeKeeper1 = GetTimeKeeper()

    # initialize the timekeeper

    # get time animation track
    timeAnimationCue1 = GetTimeTrack()

    # initialize the animation track

    # get animation scene
    animationScene1 = GetAnimationScene()

    # initialize the animation scene
    animationScene1.ViewModules = renderView1
    animationScene1.Cues = timeAnimationCue1
    animationScene1.AnimationTime = 0.0

    # initialize the animation scene

    # ----------------------------------------------------------------
    # setup extractors
    # ----------------------------------------------------------------

    # create extractor
    pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
    # trace defaults for the extractor.
    pNG1.Trigger = 'Time Step'

    # init the 'Time Step' selected for 'Trigger'
    pNG1.Trigger.Frequency = 10

    # init the 'PNG' selected for 'Writer'
    pNG1.Writer.FileName = 'PTrap_Part_RenderView1_{timestep:06d}{camera}.png'
    pNG1.Writer.ImageResolution = [1500, 1000]
    pNG1.Writer.OverrideColorPalette = 'GradientBackground'
    pNG1.Writer.Format = 'PNG'

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(pNG1)
    # ----------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # Catalyst options
    from paraview import catalyst
    options = catalyst.Options()
    options.ExtractsOutputDirectory = 'datasets_png'
    options.GlobalTrigger = 'Time Step'
    options.EnableCatalystLive = 1
    options.CatalystLiveTrigger = 'Time Step'

    # ------------------------------------------------------------------------------
    if __name__ == '__main__':
        from paraview.simple import SaveExtractsUsingCatalystOptions
        # Code for non in-situ environments; if executing in post-processing
        # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
        SaveExtractsUsingCatalystOptions(options)



particle_png()