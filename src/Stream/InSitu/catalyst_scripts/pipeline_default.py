# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html
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
import paraview.catalyst
import argparse


from paraview import servermanager

# Import Catalyst utility subroutines
from catalystSubroutines import (
    print_proxy_overview,
    create_VTPD_extractor
)
#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


# print start marker
# print_info("'%s'=====EXECUTING CATALYST PIPELINE========>",__name__)


print("\n\n\n")
print_info("=================================================="[0:40]+"|")
print_info("'%s'===EXECUTING CATALYST PIPELINE================"[0:40]+"|", __name__)
print_info("=================================================="[0:40]+"|")




arg_list = paraview.catalyst.get_args()
# print_info(f"Arguments received: {arg_list}")
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="default_name", help="doesnt matter")
parser.add_argument("--VTKextract", type=bool, default=False, help="Enable the VTK extracts of all incoming channels")
parser.add_argument("--steer",      type=bool, default=False, help="Enable steering from catalyst python side")
parsed = parser.parse_args(arg_list)
print_info(f"Parsed VTK extract options:     {parsed.VTKextract}")
print_info(f"Parsed steering option:         {parsed.steer}")





# ----------------------------------------------------------------------
# -------------EXTRACTORS-----------------------------------------------
# ----------------------------------------------------------------------

# print("=== SETTING TRIVIAL PRODUCERS (LIVE) =======0")
# registrationName must match the channel name used in the 'CatalystAdaptor'.
ippl_parti_e        = PVTrivialProducer(registrationName='ippl_particles')
ippl_field_s        = PVTrivialProducer(registrationName='ippl_scalar')
ippl_field_v        = PVTrivialProducer(registrationName='ippl_E')
# # ippl_field_phi = PVTrivialProducer(registrationName='ippl_phi')
# print("=== SETTING TRIVIAL PRODUCERS (LIVE) =======1")



if parsed.VTKextract:
    print_info("===SETTING VTK DATA EXTRAXCTION================"[0:30]+"|0")
    vTPD_particle = create_VTPD_extractor("particle", ippl_parti_e, 1)
    vTPD_field_v  = create_VTPD_extractor("field_v",  ippl_field_v, 1)
    vTPD_field_s  = create_VTPD_extractor("field_s",  ippl_field_s, 1)
    print_info("===SETTING VTK DATA EXTRACTION==================="[0:30]+"|1")


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------




print_info("===SETTING CATALYST OPTIONS================"[0:30]+"|0")
# Catalyst options
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'data_vtk_extracts'
 # Set only a single output directory
print_info("===SETTING CATALYST OPTIONS==================="[0:30]+"|1")





if parsed.steer:
    print_info("===CREATING STEERABLES============="[0:30]+"|0")
    try:
        # steerable_parameters = CreateSteerableParameters("STEERING_TYPE", "SteerableParameters")
        # steering_parameters = servermanager.ProxyManager().GetProxy("sources", "SteeringParameters")
    
    
    
        # = CreateSteerableParameters("SteerableParameters")
        steerable_parameters_electric =  CreateSteerableParameters(
                                    steerable_proxy_type_name           = "SteerableParameters_electric",
                                    steerable_proxy_registration_name   = "SteeringParameters_electric",
                                    result_mesh_name                    = "steerable_channel_backward_electric"
                                )
        
        steerable_parameters_magnetic =  CreateSteerableParameters(
                                    steerable_proxy_type_name           = "SteerableParameters_magnetic",
                                    steerable_proxy_registration_name   = "SteeringParameters_magnetic",
                                    result_mesh_name                    = "steerable_channel_backward_magnetic"
                                    # result_mesh_name                    = "steerable_magnetic_mesh_backward"
                                )
    
    
    
        if steerable_parameters_electric is None:
            print_info("Error: SteerableParameters_electric proxy not found (CreateSteerableParameters returned None).")
        else:
            print_info("SteerableParameters_electric loaded successfully.")
        
        if steerable_parameters_magnetic is None:
            print_info("Error: SteerableParameters_magnetic proxy not found (CreateSteerableParameters returned None).")
        else:
            print_info("SteerableParameters_magnetic loaded successfully.")
    
    except Exception as e:
        print_info(f"Exception while loading SteerableParameters: {e}")
    
    print_info("===CREATING STEERABLES=============="[0:30]+"|1")





print_info("=== Printing Proxy Overview ============"[0:30]+"0")
print_proxy_overview()
print_info("=== Printing Proxy Overview ============"[0:30]+"1")





# from catalyst_extractors import png_ext_particle




# from catalyst_extractors import png_ext_sfield
# from catalyst_extractor import png_ext_sfield.ippl_scalar 
# from png_ext_sfield


# from catalyst_extractors import png_ext_vfield
""" if done like this view will detach after a couple time steps ... """

""" maybe need to test like this """
# from catalyst_extractors.png_ext_particle import pNG1
# from catalyst_extractors.png_ext_sfield import pNG2
# from catalyst_extractors.png_ext_vfield import pNG3




# from catalyst_extractors.png_ext_vfield import glyph1
""" this import while extract also active being listed in conduit will cause two separate extarcts  """


# glyph1 = Glyph( registrationName='Glyph1', 
#                 Input=ippl_field_v_,
#                 GlyphType='Arrow')

# glyph1.OrientationArray = ['CELLS', 'electrostatic']
# glyph1.ScaleArray = ['CELLS', 'electrostatic']
# glyph1.ScaleFactor = 1.32
# glyph1.GlyphTransform = 'Transform2'

# # init the 'Arrow' selected for 'GlyphType'
# glyph1.GlyphType.TipResolution = 20
# glyph1.GlyphType.TipLength = 0.29
# glyph1.GlyphType.ShaftResolution = 8
# glyph1.GlyphType.ShaftRadius = 0.02


# # show data from glyph1
# glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')


print_info("===DEFINING CATALYST_ init, exe, fini======================"[0:40]+"|0")

def catalyst_initialize():
    print_info("in '%s::catalyst_initialize'", __name__)
    print_info("===CALLING catalyst_initialize()===="[0:30]+">0")

    # arg_list = paraview.catalyst.get_args()


    print_info("===CALLING catalyst_initialize()===="[0:30]+">1")


# ------------------------------------------------------------------------------
def catalyst_execute(info):
    print_info("_________executing (cycle={}, time={})___________".format(info.cycle, info.time))
    print_info("'%s::catalyst_execute()'", __name__)

    global parsed
    global ippl_parti_e
    global ippl_field_s
    global ippl_field_v
    # global glyph1

    ippl_parti_e.UpdatePipeline()
    ippl_field_s.UpdatePipeline()
    ippl_field_v.UpdatePipeline()
    # glyph1.UpdatePipeline()

    if parsed.steer:

        global steerable_parameters_electric
        global steerable_parameters_magnetic
        steerable_parameters_electric.scaleFactor_e[0] = 31 + info.cycle
        steerable_parameters_magnetic.scaleFactor_m[0] = 31 + info.cycle
    


    if options.EnableCatalystLive:
        time.sleep(0.2)

# ------------------------------------------------------------------------------




def catalyst_finalize():
    print_info("in '%s::catalyst_finalize'", __name__)
    print_info("===CALLING catalyst_finalize()===="[0:30]+"|0")
    print_info("===CALLING catalyst_finalize()===="[0:30]+"|1")





print_info("===DEFINING CATALYST_ init, exe, fini======================"[0:40]+"|1")



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
print_info("================================================"[0:40]+"|")
print_info("'%s'===END OF CATALYST PIPELINE================="[0:40]+"|", __name__)
print_info("================================================"[0:40]+"|\n\n\n")
# print end marker









        # // // Pass Conduit node to Catalyst
        # // catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        # // if (err != catalyst_status_ok) {
        # //     std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        # // }

        # // Results(scaleFactor);

    # # In a real simulation sleep is not needed. We use it here to slow down the
    # # "simulation" and make sure ParaView client can catch up with the produced
    # # results instead of having all of them flashing at once.
    # if options.EnableCatalystLive:
    #     time.sleep(5)


    # https://docs.paraview.org/en/latest/Catalyst/blueprints.html#background


# https://github.com/jhgoebbert/ippl/commit/5b3012942849e939f56cf1c260ef2334e1565533#diff-96bf80a34c8923d60556bbae1b4d35791ea13aadc3d3e75d722fd801be609ee7




# def CreateSteerableParameters(
#                               steerable_proxy_type_name = ,
#                               steerable_proxy_registration_name   ="SteeringParameters",
#                               result_mesh_name                    ="steerable"
#         ):

#     pxm = servermanager.ProxyManager()
#     steerable_proxy = pxm.NewProxy("sources", steerable_proxy_type_name)
#     pxm.RegisterProxy("sources", steerable_proxy_registration_name,
#                       steerable_proxy)
#     steerable_proxy_wrapper = servermanager._getPyProxy(steerable_proxy)
#     UpdateSteerableParameters(steerable_proxy_wrapper, result_mesh_name)
#     return steerable_proxy_wrapper

