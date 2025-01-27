# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html
from paraview import print_info
from paraview.simple import *
from paraview import catalyst
import time

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# print start marker
print_info("begin '%s'", __name__)

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# registrationName must match the channel name used in the 'CatalystAdaptor'.
ippl_field = PVTrivialProducer(registrationName='ippl_E')
ippl_particle = PVTrivialProducer(registrationName='ippl_particle')


from paraview.simple import LoadPlugin, CreateSteerableParameters

# SteerableParameters 생성
try:
    steerable_parameters = CreateSteerableParameters("SteerableParameters")
    print("SteerableParameters loaded successfully.")
except Exception as e:
    print(f"Error loading SteerableParameters: {e}")

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
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0

# initialize the animation scene

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor (U = unstructured)
# vTI1 = CreateExtractor('VTI', ippl_field, registrationName='VTI1')
# vTI1.Trigger = 'Time Step'
# vTI1.Writer.FileName = 'ippl_field_{timestep:06d}.vti'

# create extractor (PD=point data)
#vTPD2 = CreateExtractor('VTPD', ippl_particle, registrationName='VTPD2')
#vTPD2.Trigger = 'Time Step'
#vTPD2.Writer.FileName = 'ippl_particle_{timestep:06d}.vtpd'

# ----------------------------------------------------------------
# restore active source
#SetActiveSource(ippl_particle)
SetActiveSource(ippl_field)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
#options.ExtractsOutputDirectory = 'data_vtpd'

# ------------------------------------------------------------------------------
def catalyst_initialize():
    print_info("in '%s::catalyst_initialize'", __name__)


# ------------------------------------------------------------------------------
def catalyst_execute(info):
    print_info("in '%s::catalyst_execute'", __name__)

    global ippl_field
    ippl_field.UpdatePipeline()
    global ippl_particle
    ippl_particle.UpdatePipeline()
    
    global steerable_parameters
    steerable_parameters.scaleFactor[0] = 31 + info.cycle

    print("-----------------------------------")
    print("executing (cycle={}, time={})".format(info.cycle, info.time))
    #print("field bounds   :", ippl_field.GetDataInformation().GetBounds())
    #print("particle bounds:", ippl_particle.GetDataInformation().GetBounds())

    # In a real simulation sleep is not needed. We use it here to slow down the
    # "simulation" and make sure ParaView client can catch up with the produced
    # results instead of having all of them flashing at once.
    if options.EnableCatalystLive:
        time.sleep(5)


# ------------------------------------------------------------------------------
def catalyst_finalize():
    print_info("in '%s::catalyst_finalize'", __name__)


# print end marker
print_info("end '%s'", __name__)