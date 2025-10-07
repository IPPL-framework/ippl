# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html

########################################################
######################################################## 
# Main paraview catalyst script. Includes VTK extractors,
# steering capabilities and updating pipelines of all channels 
# Visualizes 3D particles. (ParticleContainer/ParticleBase)
# 
# Currently hard coded to rely on attributes:
# - 
# 
# Currently coded to rely on hardcoded steering labels
#  - "electric"
#  - "magnetic"
# 
# 
# Possible TODO:
#  - Make Steering more Versatile
#  - Together with CPP don't rely on hard coded attributes
#  - additionally pass field string to have constistent bounds
#    and don't have to guess reference frame ...
#  - figure out how to also display glyphs inside the PV 
#    client GUI (worked when wasnt in a separate script ...)
#  - More
########################################################
########################################################

import paraview
from paraview.simple import *
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 12
import paraview.catalyst
from paraview import catalyst

import paraview.simple as pvs
from paraview.simple import (
    LoadPlugin,
    CreateSteerableParameters,
    PVTrivialProducer
)
from paraview import servermanager
from paraview import print_info

import argparse
import sys
import time
import os

sys.path.append(os.path.dirname(__file__))


from catalystSubroutines import (
    print_proxy_overview,
    create_VTPD_extractor
)
#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()



print_info("\n\n\n")
# ------------------------------------------------------------------------------
print_info("=================================================="[0:40]+"|")
print_info("'%s'===EXECUTING CATALYST PIPELINE================"[0:40]+"|", __name__)
print_info("=================================================="[0:40]+"|")
# ------------------------------------------------------------------------------




# ----------------------------------------------------------------
# Parse arguments received via conduit node
# ----------------------------------------------------------------
arg_list = paraview.catalyst.get_args()
# print_info(f"Arguments received: {arg_list}")
parser = argparse.ArgumentParser()
parser.add_argument("--channel_names", nargs="*",
                     help="Pass All Channel Names for which we need to update the privial producer each round")
parser.add_argument("--VTKextract", type=bool, default=False, help="Enable the VTK extracts of all incoming channels")
parser.add_argument("--steer",      default="OFF", help="Enable steering from catalyst python side")
parsed = parser.parse_args(arg_list)
print_info(f"Parsed channel_names:           {parsed.channel_names}")
print_info(f"Parsed VTK extract options:     {parsed.VTKextract}")
print_info(f"Parsed steering option:         {parsed.steer}")



# ----------------------------------------------------------------
# create a new 'XML Partitioned Dataset Reader'
# Dynamically create PVTrivialProducer objects for each channel name
# ----------------------------------------------------------------
print_info("=== SETTING TRIVIAL PRODUCERS (LIVE) =======0")
channel_readers = {}
if parsed.channel_names:
    for cname in parsed.channel_names:
        channel_readers[cname] = PVTrivialProducer(registrationName=cname)
else:
    print_info("No channel names provided in parsed.channel_names.")
print_info("=== SETTING TRIVIAL PRODUCERS (LIVE) =======1")



# ------------------------------------------------------------------------------
# Optionally create VTPD extractors for each channel
# ------------------------------------------------------------------------------
extractors = {}
if parsed.VTKextract and parsed.channel_names:
    print_info("===SETTING VTK DATA EXTRAXCTION================"[0:30]+"|0")
    for cname, reader in channel_readers.items():
        extractors[cname] = create_VTPD_extractor(cname, reader, 1)
    print_info("===SETTING VTK DATA EXTRACTION==================="[0:30]+"|1")



# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'data_vtk_extracts'



# ------------------------------------------------------------------------------
# Setup steering channels
# ------------------------------------------------------------------------------
if parsed.steer == "ON" :
    print_info("===CREATING STEERABLES============="[0:30]+"|0")
    
    # ------------------------------------------------------------------------------
    # forward / incoming steering channels
    # ------------------------------------------------------------------------------

    steerable_field_in_electric = PVTrivialProducer(registrationName='steerable_channel_forward_electric')
    steerable_field_in_magnetic = PVTrivialProducer(registrationName='steerable_channel_forward_magnetic')

    # ------------------------------------------------------------------------------
    # backward / outgoing steering channels
    # ------------------------------------------------------------------------------

    try:    
        steerable_parameters_electric =  CreateSteerableParameters(
                                    steerable_proxy_type_name           = "SteerableParameters_electric",
                                    steerable_proxy_registration_name   = "SteeringParameters_electric",
                                    result_mesh_name                    = "steerable_channel_backward_electric"
                                )
        
        steerable_parameters_magnetic =  CreateSteerableParameters(
                                    steerable_proxy_type_name           = "SteerableParameters_magnetic",
                                    steerable_proxy_registration_name   = "SteeringParameters_magnetic",
                                    result_mesh_name                    = "steerable_channel_backward_magnetic"
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
        print_info(f"Exception while loading (backward) SteerableParameters: {e}")
    
    print_info("===CREATING STEERABLES=============="[0:30]+"|1")




# ------------------------------------------------------------------------------
print_info("=== Printing Proxy Overview ============"[0:30]+"0")
print_proxy_overview()
print_info("=== Printing Proxy Overview ============"[0:30]+"1")
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
def catalyst_initialize():
    print_info("in '%s::catalyst_initialize'", __name__)
    # print_info("===CALLING catalyst_initialize()===="[0:30]+">0")

    # arg_list = paraview.catalyst.get_args()
    # print_info("===CALLING catalyst_initialize()===="[0:30]+">1")
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
def catalyst_execute(info):
    print_info("_________executing (cycle={}, time={})___________".format(info.cycle, info.time))
    print_info("'%s::catalyst_execute()'", __name__)

    global parsed
    global channel_readers
    # Update all channel readers
    for reader in channel_readers.values():
        reader.UpdatePipeline()

    if parsed.steer == "ON":
        print_info("setting backward steerables")
        global steerable_parameters_electric
        global steerable_parameters_magnetic
        steerable_parameters_electric.scaleFactor_e[0] = 31 + info.cycle
        steerable_parameters_magnetic.scaleFactor_m[0] = 31 + info.cycle

    if options.EnableCatalystLive:
        time.sleep(0.2)
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
def catalyst_finalize():
    print_info("in '%s::catalyst_finalize'", __name__)
    print_info("===CALLING catalyst_finalize()===="[0:30]+"|0")
    print_info("===CALLING catalyst_finalize()===="[0:30]+"|1")
# ------------------------------------------------------------------------------






# ------------------------------------------------------------------------------
print_info("================================================"[0:40]+"|")
print_info("'%s'===END OF CATALYST PIPELINE================="[0:40]+"|", __name__)
print_info("================================================"[0:40]+"|\n\n\n")
# ------------------------------------------------------------------------------









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
