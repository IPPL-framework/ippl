# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html


# Add this block to the top of your file
import sys
import time
import os
sys.path.append(os.path.dirname(__file__))

from paraview import print_info
from paraview.simple import *
from paraview import catalyst
from paraview.simple import LoadPlugin, CreateSteerableParameters


# from catalyst_scripts import proxy_helpers
# from catalyst_scripts import data_extractor_helper

import proxy_helpers
import data_extractor_helper


#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


# print start marker
print("====================================>")
print("===EXECUTING CATALYST PIPELINE======>")
print("====================================>")
print_info("\nstart'%s'\n", __name__)



print("===CREATING STEERABLES======")
try:
    steerable_parameters = CreateSteerableParameters("SteerableParameters")
    # steerable_parameters = CreateSteerableParameters("STEERING_TYPE", "SteerableParameters")
    # steering_parameters = servermanager.ProxyManager().GetProxy("sources", "SteeringParameters")
    if steerable_parameters is None:
        print("Error: SteerableParameters proxy not found (CreateSteerableParameters returned None).")
    else:
        print("SteerableParameters loaded successfully.")
except Exception as e:
    print(f"Exception while loading SteerableParameters: {e}")

print("===CREATING STEERABLES======DONE")
proxy_helpers.print_proxy_overview()



print("===SETTING ACTIVE SOURCES=======")
# registrationName must match the channel name used in the 'CatalystAdaptor'.
ippl_field_v        = PVTrivialProducer(registrationName='ippl_E')
ippl_field_s        = PVTrivialProducer(registrationName='ippl_scalar')
ippl_particle       = PVTrivialProducer(registrationName='ippl_particle')
# ippl_field_phi = PVTrivialProducer(registrationName='ippl_phi')
print("===SETTING ACTIVE SOURCES=======DONE")



# ------------------------------------------------------------------------------
# -------------EXTRACTORS-----------------------------------------------
# ------------------------------------------------------------------------------
print("===CREATING EXTRACTORS==============")


""" these will directly reate extractors without further use
these files are catalst save files directly created from paraview -... """
from catalyst_extractors import png_ext_particle
from catalyst_extractors import png_ext_sfield
from catalyst_extractors import png_ext_vfield


# create extractor (PD=partitioned dataset...)
# vTPD_eg = CreateExtractor('VTPD', ippl_eg, registrationName='VTPD_eg')
# vTPD_eg.Trigger = 'TimeStep'  
# vTPD_eg.Trigger.Frequency = 10
# vTPD_eg.Writer.FileName = 'ippl_particle_{timestep:06d}.vtpd'

vTPD_particle = data_extractor_helper.create_VTPD_extractor("particle", ippl_particle)
vTPD_particle = data_extractor_helper.create_VTPD_extractor("field_v", ippl_field_v)
vTPD_particle = data_extractor_helper.create_VTPD_extractor("field_s", ippl_field_s)
print("===CREATING EXTRACTORS==============DONE")



print("===CHECKING IPPL DATA===================")
# Add detailed data type checking
field_output    = ippl_field_v.GetClientSideObject().GetOutput()
scalar_output   = ippl_field_s.GetClientSideObject().GetOutput()
particle_output = ippl_particle.GetClientSideObject().GetOutput()
particle_info   = ippl_particle.GetDataInformation()
field_info      = ippl_field_v.GetDataInformation()
scalar_info     = ippl_field_s.GetDataInformation()
# Debug: Check data availability (can be done at each cycle...)
print(f"Data types:")
print(f"  - ippl_field_v       : {type(field_output).__name__}")
print(f"                       {field_info.GetNumberOfPoints()} points, {field_info.GetNumberOfCells()} cells")
print(f"  - ippl_field_s:      {type(scalar_output).__name__}")
print(f"                       {scalar_info.GetNumberOfPoints()} points, {scalar_info.GetNumberOfCells()} cells") 
print(f"  - ippl_particle    : {type(particle_output).__name__}")
print(f"                       {particle_info.GetNumberOfPoints()} points, {particle_info.GetNumberOfCells()} cells")
print("===CHECKING IPPL DATA===================DONE")



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



print("===SETING CATALYST OPTIONS======================")
# Catalyst options
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'ippl_catalyst_output'
#options.ExtractsOutputDirectory = 'data_vtpd'
 # Set only a single output directory
# options.CatalystLiveTriggerFrequency = 1 ?????
print("===SETING CATALYST OPTIONS=======================DONE")




# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

print("===DEFINING CATALYST_ init, exe, fini======================")

def catalyst_initialize():
    print_info("in '%s::catalyst_initialize'", __name__)
    print("====================================>")
    print("===CALLING catalyst_initialize()====>")



    
    print("===CALLING catalyst_initialize()====>DONE")
    print("====================================>")



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------





def catalyst_execute(info):
    # print_info("in '%s::catalyst_execute'", __name__)
    print(f"---Cycle {info.cycle}:----catalyst_execute()---------------START")
    print("executing (cycle={}, time={})".format(info.cycle, info.time))
    print("field bounds   :", ippl_field_v.GetDataInformation().GetBounds())
    print("field bounds   :", ippl_field_s.GetDataInformation().GetBounds())
    print("particle bounds:", ippl_particle.GetDataInformation().GetBounds())


    
    

    ippl_field_v.UpdatePipeline()
    ippl_field_s.UpdatePipeline()
    ippl_particle.UpdatePipeline()



    
    global steerable_parameters
    steerable_parameters.scaleFactor[0] = 31 + info.cycle
    


    if options.EnableCatalystLive:
        time.sleep(0.5)

    print(f"---Cycle {info.cycle}:----catalyst_execute()---------------DONE")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



def catalyst_finalize():
    print_info("in '%s::catalyst_finalize'", __name__)
    print("==================================|")
    print("===CALLING catalyst_finalize()====|")


    
    print("===CALLING catalyst_finalize()====|DONE")
    print("==================================|")





print("===DEFINING CATALYST_ init, exe, fini======================DONE")



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
print("\n\n")
print_info("end '%s'", __name__)
print("====================================|")
print("===END OF CATALYST PIPELINE=========|")
print("====================================|")
print("\n\n")
# print end marker
