# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html


# Add this block to the top of your file
import sys
import time
import os
sys.path.append(os.path.dirname(__file__))

import paraview
from paraview import print_info
from paraview import catalyst
from paraview.simple import *
from paraview.simple import LoadPlugin, CreateSteerableParameters, PVTrivialProducer
import paraview.simple as pvs
# from paraview.simple import GetActive


from paraview import servermanager as sm
from paraview import servermanager

# Import Catalyst utility subroutines
from catalystSubroutines import (
    print_proxy_overview,
    create_VTPD_extractor,
    register_png_extractor,
    load_state_module,
    _fix_png_extractors_force,
    _debug_dump_state,
    get_keepalive_counts,
    set_log_level
)



#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


# print start marker
print("====================================>")
print("===EXECUTING CATALYST PIPELINE======>")
print("====================================>")
print_info("\nstart'%s'\n", __name__)





print_proxy_overview()
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

print_proxy_overview()




# print("===SETTING DATA EXTRAXCTION=======")
# registrationName must match the channel name used in the 'CatalystAdaptor'.
ippl_field_v        = PVTrivialProducer(registrationName='ippl_E')
ippl_field_s        = PVTrivialProducer(registrationName='ippl_scalar')
ippl_particle       = PVTrivialProducer(registrationName='ippl_particle')
# # ippl_field_phi = PVTrivialProducer(registrationName='ippl_phi')


# vTPD_particle = create_VTPD_extractor("particle", ippl_particle)
# vTPD_field_v  = create_VTPD_extractor("field_v",  ippl_field_v)
# vTPD_field_s  = create_VTPD_extractor("field_s",  ippl_field_s)

# print("===SETTING DATA EXTRACTION=======DONE")





# ----------------------------------------------------------------------
# -------------EXTRACTORS-----------------------------------------------
# ----------------------------------------------------------------------

# print("===CREATING PNG EXTRACTORS==============")
# """ these will directly create extractors without further use
# these files are catalyst save files directly created from paraview -... """
# from catalyst_extractors import png_ext_particle
# from catalyst_extractors import png_ext_sfield
# from catalyst_extractors import png_ext_vfield
# print("===CREATING PNG EXTRACTORS==============DONE")





# print("===CHECKING IPPL DATA===================")
# # Add detailed data type checking
# field_output    = ippl_field_v.GetClientSideObject().GetOutput()
# scalar_output   = ippl_field_s.GetClientSideObject().GetOutput()
# particle_output = ippl_particle.GetClientSideObject().GetOutput()
# particle_info   = ippl_particle.GetDataInformation()
# field_info      = ippl_field_v.GetDataInformation()
# scalar_info     = ippl_field_s.GetDataInformation()
# # Debug: Check data availability (can be done at each cycle...)
# print(f"Data types:")
# print(f"  - ippl_field_v       : {type(field_output).__name__}")
# print(f"                       {field_info.GetNumberOfPoints()} points, {field_info.GetNumberOfCells()} cells")
# print(f"  - ippl_field_s:      {type(scalar_output).__name__}")
# print(f"                       {scalar_info.GetNumberOfPoints()} points, {scalar_info.GetNumberOfCells()} cells") 
# print(f"  - ippl_particle    : {type(particle_output).__name__}")
# print(f"                       {particle_info.GetNumberOfPoints()} points, {particle_info.GetNumberOfCells()} cells")
# print("===CHECKING IPPL DATA===================DONE")



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

set_log_level("NONE")

# Capture VTK errors to file for post-mortem (vtk_errors.log in CWD)
try:
    from vtkmodules.vtkCommonCore import vtkFileOutputWindow, vtkOutputWindow
    _vtk_log = vtkFileOutputWindow()
    _vtk_log.SetFileName("vtk_errors.log")
    vtkOutputWindow.SetInstance(_vtk_log)
    print("[DEBUG] VTK error log redirected to vtk_errors.log")
except Exception as e:
    print(f"[WARN] VTK error log setup failed: {e}")











print("===SETING CATALYST OPTIONS======================")
# Catalyst options
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'ippl_catalyst_output'
 # Set only a single output directory
print("===SETING CATALYST OPTIONS=======================DONE")



try:
    from catalyst_extractors import png_ext_particle
    register_png_extractor(png_ext_particle)
except Exception as e:
    print(f"[WARN] particle registration failed: {e}")

try:
    from catalyst_extractors import png_ext_sfield
    register_png_extractor(png_ext_sfield)
except Exception as e:
    print(f"[WARN] sfield registration failed: {e}")

try:
    from catalyst_extractors import png_ext_vfield
    register_png_extractor(png_ext_vfield)
except Exception as e:
    print(f"[WARN] vfield registration failed: {e}")

# try:
#     from catalyst_extractors import png_ext_particle
#     register_png_extractor(png_ext_particle.renderView1, apng_ext_partcile.pNG1)
# except Exception as e:
#     print(f"[WARN]  registration failed: {e}")


# try:
#     from catalyst_extractors import aa
#     register_png_extractor(aa.renderView2, aa.pNG2)
# except Exception as e:
#     print(f"[WARN]  registration failed: {e}")

kc = get_keepalive_counts()
print(f"[DEBUG] Keepalive views count: {kc[0]}  extractors: {kc[1]}")





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

    # Force PNG extractors to have proper view associations each cycle
    _fix_png_extractors_force()
    # Ensure views are freshly rendered (helps when views were created but not yet rendered)
    try:
        pvs.RenderAllViews()
    except Exception:
        pass
    # _debug_dump_state(tag=f"pre-cycle {info.cycle}")

    # print("field bounds   :", ippl_field_v.GetDataInformation().GetBounds())
    # print("field bounds   :", ippl_field_s.GetDataInformation().GetBounds())
    # print("particle bounds:", ippl_particle.GetDataInformation().GetBounds())

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
