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
#  - (alternative or additionally) write working macros for glyph filter   
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
    create_VTPD_extractor,
    _first_dataset_from_composite,
    _fetch_first_dataset,
    _get_first_scalar_from_any_assoc
)
#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()



def print_info_(s, level=0):
    global verbosity
    if verbosity>level:
        print_info(s)


# ----------------------------------------------------------------
# Parse arguments received via conduit node
# ----------------------------------------------------------------
arg_list = paraview.catalyst.get_args()
# print_info_(f"Arguments received: {arg_list}")
parser = argparse.ArgumentParser()
parser.add_argument("--channel_names", nargs="*",
                     help="Pass All Channel Names for which we need to update the privial producer each round")
parser.add_argument("--steer_channel_names", nargs="*",
                     help="Pass All Channel Names for Steering scalar parameters")

parser.add_argument("--verbosity", type=int, default="1", help="Communicate the catalyst Output Level from the simulation")
parser.add_argument("--VTKextract", default="OFF", help="Enable the VTK extracts of all incoming channels")
parser.add_argument("--steer",      default="OFF", help="Enable steering from catalyst python side")
parser.add_argument("--experiment_name", default="_", help="Needed to correctly for safe folder.")

parsed = parser.parse_args(arg_list)

exp_string = parsed.experiment_name
verbosity = parsed.verbosity


# ------------------------------------------------------------------------------
print_info_("==========================================================0"[0:55]+"|")
print_info_("======= EXECUTING catalyst_pipeline GLOBAL SCOPE =========0"[0:55]+"|")
print_info_("==========================================================0"[0:55]+"|")
# ------------------------------------------------------------------------------



print_info_(f"Parsed steer_channel_names:     {parsed.steer_channel_names}")
print_info_(f"Parsed channel_names:           {parsed.channel_names}")
print_info_(f"Parsed verbosity level:         {parsed.verbosity}")
print_info_(f"Parsed VTK extract options:     {parsed.VTKextract}")
print_info_(f"Parsed steering option:         {parsed.steer}")

# ----------------------------------------------------------------
# create a new 'XML Partitioned Dataset Reader'
# Dynamically create PVTrivialProducer objects for each channel name
# ----------------------------------------------------------------
print_info_("=== SETTING TRIVIAL PRODUCERS (LIVE) ======="[0:40]+"|0")
channel_readers = {}
if parsed.channel_names:
    for cname in parsed.channel_names:
        channel_readers[cname] = PVTrivialProducer(registrationName=cname)
else:
    print_info_("No channel names provided in parsed.channel_names.")
print_info_("=== SETTING TRIVIAL PRODUCERS (LIVE) ======="[0:40]+"|1")



# ------------------------------------------------------------------------------
# Optionally create VTPD extractors for each channel
# ------------------------------------------------------------------------------
extractors = {}
if parsed.VTKextract == "ON" and parsed.channel_names:
    print_info_("=== SETTING VTK DATA EXTRAXCTION================"[0:40]+"|0")
    for cname, reader in channel_readers.items():
        extractors[cname] = create_VTPD_extractor(cname, reader, 1)
    print_info_("=== SETTING VTK DATA EXTRACTION==================="[0:40]+"|1")



# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'data_vtk_extracts_' + exp_string



# ------------------------------------------------------------------------------
# Setup steering channels
# ------------------------------------------------------------------------------

steer_channel_readers = {}
steer_channel_senders = {}
steer_channels = {}

if parsed.steer == "ON":
    if parsed.steer_channel_names :
        print_info_("===CREATING STEERABLES============="[0:40]+"|0")

        # ------------------------------------------------------------------------------
        # forward / incoming steering channels
        # ------------------------------------------------------------------------------
        print_info_("FORWARD")
        for sname in parsed.steer_channel_names:

            print_info_(sname)
            steer_channel_readers[sname] = PVTrivialProducer(registrationName="steerable_channel_forward_"+sname)
    else:
        print_info_("No channel names provided in parsed.channel_names.")

    # EG:
    # steerable_field_in_magnetic = PVTrivialProducer(registrationName='steerable_channel_forward_magnetic')



    # ------------------------------------------------------------------------------
    # backward / outgoing steering channels
    # ------------------------------------------------------------------------------
    print_info_("BACKWARD")
    try:    
        for sname in parsed.steer_channel_names:
            steer_channel_senders[sname] = CreateSteerableParameters(
                                    steerable_proxy_type_name           = "SteerableParameters_"+sname,
                                    steerable_proxy_registration_name   = "SteeringParameters_"+sname,
                                    result_mesh_name                    = "steerable_channel_backward_"+sname
            )
            if steer_channel_senders[sname] is None:
                print_info_("Error: SteerableParameters_"+sname+" proxy not found (CreateSteerableParameters returned None).")
            else:
                print_info_("SteerableParameters_" + sname + " loaded successfully.")
        
        # EG:
        # steerable_parameters_magnetic =  CreateSteerableParameters(
        #                             steerable_proxy_type_name           = "SteerableParameters_magnetic",
        #                             steerable_proxy_registration_name   = "SteeringParameters_magnetic",
        #                             result_mesh_name                    = "steerable_channel_backward_magnetic"
        #                         )
        # if steerable_parameters_magnetic is None:
        #     print_info_("Error: SteerableParameters_magnetic proxy not found (CreateSteerableParameters returned None).")
        # else:
        #     print_info_("SteerableParameters_magnetic loaded successfully.")
    except Exception as e:
        print_info_(f"Exception while loading (backward) SteerableParameters: {e}")
    
    print_info_("===CREATING STEERABLES=============="[0:40]+"|1")



for sname in parsed.steer_channel_names:
    steer_channels[sname] = (steer_channel_readers[sname], steer_channel_senders[sname])



# ------------------------------------------------------------------------------
print_info_("=== Printing Proxy Overview ============"[0:40]+"0")
if verbosity > 0: 
    print_proxy_overview()
print_info_("=== Printing Proxy Overview ============"[0:40]+"1")
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
def catalyst_initialize():
    print_info_("catalyst_initialize()"+exp_string)
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
def catalyst_execute(info):
    print_info_("_________executing (cycle={}, time={})___________".format(info.cycle, info.time))
    print_info_("catalyst_execute()::"+exp_string)

    global parsed
    global channel_readers
    # Update all channel readers
    for reader in channel_readers.values():
        reader.UpdatePipeline()
        

    if parsed.steer == "ON":

        # global steer_channel_readers
        # global steer_channel_senders
        # global steerable_parameters_magnetic
        # for reader in steer_channel_readers.values():
        
        print_info_("setting backward steerables")

        
        global steer_channels
        for name, (reader, sender) in steer_channels.items():
            
            reader.UpdatePipeline()
            sender.UpdatePipeline()
             
            if sender is None:
                print("Error: SteerableParameters proxy not found (CreateSteerableParameters returned None).")
            else:
                            # # --- Set initial value from incoming channel if available ---
                            try:
                                # Access the incoming 'steerable' channel (PVTrivialProducer)
                                output = reader.GetClientSideObject().GetOutput()
                                # output = reader.GetClientSideObject().GetOutputDataObject(0)
                                partition = output.GetPartition(0)
                                # partition = output.GetBlock(0)
                                # partition = output  # fallback, may not work

                                if partition is not None and hasattr(partition, "GetPointData"):
                                    data_info = partition.GetPointData()
                                    # data_info = partition.GetFieldData()
                                    if data_info is not None and data_info.GetNumberOfArrays() > 0:
                                            initial_value = data_info.GetArray("steerable_field_f_"+name).GetTuple1(0)
                                            sender.scaleFactor[0] =  initial_value
                                else:
                                    print("Could not find a valid partition with point data for steerable channel.")

                            # try:
                            #     # Only set initial default once (don’t clobber later user edits)
                            #     # if info.cycle == 0 and name not in _initialized_steer_defaults:
                            #         # Preferred: fetch a VTK object to Python and extract the first dataset
                            #         ds = _fetch_first_dataset(reader)

                            #         # Fallback: access server-side object correctly
                            #         if ds is None:
                            #             obj = reader.GetClientSideObject().GetOutputDataObject(0)
                            #             ds = _first_dataset_from_composite(obj)

                            #         initial_value = _get_first_scalar_from_any_assoc(ds, "steerable_field_f_" + name)
                            #         if initial_value is not None:
                            #             sender.scaleFactor[0] = float(initial_value)
                            #             # _initialized_steer_defaults.add(name)
                            #             print_info_(f"Initialized steerable '{name}' from forward channel: {initial_value}")
                            #         else:
                            #             print_info_(f"Could not find array 'steerable_field_f_{name}' on forward channel.")


                            # try:
                            #         # Only set the initial default once (on first execute)
                            #         # if info.cycle == 0 and name not in _initialized_steer_defaults:
                            #     ds = _fetch_first_dataset(reader)
                            #     initial_value = _get_first_scalar_from_any_assoc(ds, "steerable_field_f_" + name)
                            #     if initial_value is not None:
                            #         sender.scaleFactor[0] = initial_value
                            #         # _initialized_steer_defaults.add(name)
                            #         print_info_(f"Initialized steerable '{name}' from forward channel: {initial_value}")
                            #     else:
                            #         print_info_(f"Could not find array 'steerable_field_f_{name}' on forward channel.")








                            except Exception as e:
                                print(f"Could not set initial steerable value from simulation for '{name}': {e}")

                            
                            """ something like this should also work...
                            but might reduce integration... """
                            # steer=servermanager.Fetch(FindSource("steerable")).GetPartition(0,0)
                            # steer_array = steer.GetPointData().GetArray("steerable")
                            # # steer_array = steer.GetFieldData().GetArray("steerable")
                            # steer_atm = 1
                            # if steer_array:
                            #       steer_atm = steer_array.GetTuple1(0)
                                    # steerable_parameters.scaleFactor[0] = steer_atm
                                                                    #  works...

                            # steerable_parameters_electric.scaleFactor[0] = 31 + info.cycle
                            # steerable_parameters_magnetic.scaleFactor[0] = 31 + info.cycle
                        
                            print_info_(f"SteerableParameter[{name}] intermediate value: {sender.scaleFactor[0]}")
















    if options.EnableCatalystLive:
        time.sleep(0.2)
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
def catalyst_finalize():
    print_info_("catalyst_finalize()::" + exp_string)
# ------------------------------------------------------------------------------






# ------------------------------------------------------------------------------
print_info_("==========================================================="[0:55]+"|")
print_info_("========== END OF catalyst_pipeline GLOBAL SCOPE =========0"[0:55]+"|")
print_info_("==========================================================="[0:55]+"|\n\n")
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
