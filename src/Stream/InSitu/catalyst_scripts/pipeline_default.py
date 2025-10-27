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
    # _first_dataset_from_composite,
    # _fetch_first_dataset,
    # _get_first_scalar_from_any_assoc
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
        # Unified sender: one proxy carrying one property per channel, single result mesh
        sender_all = CreateSteerableParameters(
                                steerable_proxy_type_name           = "SteerableParameters_ALL",
                                steerable_proxy_registration_name   = "SteeringParameters_ALL",
                                result_mesh_name                    = "steerable_channel_backward_all"
        )
        if sender_all is None:
            print_info_("Error: SteerableParameters_ALL proxy not found (CreateSteerableParameters returned None).")
        else:
            print_info_("SteerableParameters_ALL loaded successfully.")
        # keep dictionary for compatibility; all entries point to the same sender
        for sname in parsed.steer_channel_names:
            steer_channel_senders[sname] = sender_all
        
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


    if parsed.steer == "ON":
        print_info_("setting backward steerables")
        for name, (reader, sender) in steer_channels.items():
            if sender is None or reader is None:
                print_info_(f"Steer channel '{name}' not ready")
                continue

            reader.UpdatePipeline()

            # 1) Read current value from forward (simulation) channel
            sim_val = None
            # sim_val = 0
            try:
                output = reader.GetClientSideObject().GetOutput()
                part0 = output.GetPartition(0) if output else None
                pd = part0.GetPointData() if part0 and hasattr(part0, "GetPointData") else None
                arr = pd.GetArray(f"steerable_field_f_{name}") if pd else None
                if arr is not None and arr.GetNumberOfTuples() > 0:
                    sim_val = float(arr.GetTuple1(0))
            except Exception as e:
                print_info(f"Could not read sim value for '{name}': {e}")

            # --- Always set the checked value from simulation ---
            if sim_val is not None:
                # unified property name per label
                try:
                    prop_name = f"scaleFactor_{name}"
                    prop = sender.GetProperty(prop_name)
                    if prop is not None:
                        prop.SetElement(0, sim_val)
                        # Apply to server-side VTK objects and produce backward mesh with all fields
                        sender.UpdateVTKObjects()
                        sender.UpdatePipeline()
                    else:
                        # Fallback for per-channel proxy (legacy)
                        if hasattr(sender, "scaleFactor"):
                            sender.scaleFactor[0] = sim_val
                            sender.UpdateVTKObjects()
                            sender.UpdatePipeline()
                except Exception as e:
                    print_info_(f"Failed to set unified property for '{name}': {e}")

            # Log applied value (checked)
            try:

                try:
                    prop = sender.GetProperty(f"scaleFactor_{name}")
                    val = prop.GetElement(0) if prop is not None else (sender.scaleFactor[0] if hasattr(sender, "scaleFactor") else None)
                except Exception:
                    val = None
                print_info_(f"=================================>>> SteerableParameter[{name}] received: {sim_val}")
                print_info_(f"=================================>>> SteerableParameter[{name}]     sent: {val}")
            except Exception:
                pass


    if options.EnableCatalystLive:
        # time.sleep(0.2)
        time.sleep(0.5)
            
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







    # # In a real simulation sleep is not needed. We use it here to slow down the
    # # "simulation" and make sure ParaView client can catch up with the produced
    # # results instead of having all of them flashing at once.


    # https://docs.paraview.org/en/latest/Catalyst/blueprints.html#background
    # https://github.com/jhgoebbert/ippl/commit/5b3012942849e939f56cf1c260ef2334e1565533#diff-96bf80a34c8923d60556bbae1b4d35791ea13aadc3d3e75d722fd801be609ee7


