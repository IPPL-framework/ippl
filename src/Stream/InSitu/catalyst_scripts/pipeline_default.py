"""! \file pipeline_default.py
\brief Main ParaView Catalyst pipeline: live visualization, VTK extracts, and steering.
\details Discovers channel proxies, wires optional extractors, updates live views,
and forwards/fetches steerable parameters between the simulation and the GUI.
Designed to orchestrate the per-channel extractor scripts in catalyst_extractors/.
"""

# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html

########################################################
######################################################## 
# Main paraview catalyst script. Includes VTK extractors,
# steering capabilities and updating pipelines of all channels 
# Visualizes 3D particles. (ParticleContainer/ParticleBase)
# 
#  DONE:
#  - Make Steering more Versatile
#  - Together with CPP don't rely on hard coded attributes
#  - additionally pass field string to have constistent bounds
#    and don't have to guess reference frame ...
#  - figure out how to also display glyphs inside the PV 
#    client GUI (worked when wasnt in a separate script ...)
# 
# Possible TODO:
#  - (alternative or additionally) write working MACROS for inaccessible PV settings  
#  - Colouring in GUI
# - currently png extractors rely on velocity attribute
#  - currently extractor scriots rely on velocity attribute ...
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
    CreateSteerableParameters,
    PVTrivialProducer,
    Show,
    MergeBlocks, 
    ExtractBlock,
    CellDatatoPointData,
    Glyph,
    GetColorTransferFunction
    # LoadPlugin,
    # ExtractSubset,
    # ResampleToImage,
    # AdaptiveResampleToImage,
)
from paraview import servermanager
from paraview import servermanager as sm
from paraview import print_info

import argparse
import sys
import time
import os

import string

sys.path.append(os.path.dirname(__file__))


from catalystSubroutines import (
    print_proxy_overview,
    create_VTPD_extractor
)
#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


def _log(msg):
    """Logs a message from the Catalyst script."""

    controller = sm.vtkProcessModule.GetProcessModule().GetGlobalController()
    rank = 0
    if controller:
        rank = controller.GetLocalProcessId()
        msg =  "[rank " + str(rank) + "]:  " + msg
    

    print( msg  )

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
parser.add_argument("--show_forward_channels", default="OFF", help="Show forward steerable channels in the GUI (PVTrivialProducer)")
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




# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'data_vtk_extracts_' + exp_string




# ------------------------------------------------------------------------------
# Proactively remove the forward 0D mesh source from the GUI if it was auto-created
# ------------------------------------------------------------------------------
if parsed.steer == "ON" and parsed.show_forward_channels == "OFF":
    try:
        src = paraview.simple.FindSource("steerable_channel_0D_mesh")
        if src is not None:
            print_info_("Hiding auto-created steerable_channel_0D_mesh from GUI")
            paraview.simple.Delete(src)
    except Exception as e:
        print_info_(f"Could not hide steerable_channel_0D_mesh: {e}")



# Global dictionaries to store proxies, filters etc
_extractors = {}
_sources = {}
_filters = {}
# _shows={}

# # ----------------------------------------------------------------
# # create a new 'XML Partitioned Dataset Reader'
# # Dynamically create PVTrivialProducer objects for each channel name
# # ----------------------------------------------------------------
# print_info_("=== SETTING TRIVIAL PRODUCERS (LIVE) ======="[0:40]+"|0")
# if parsed.channel_names:
#     for cname in parsed.channel_names:
#         _sources[cname] = PVTrivialProducer(registrationName=cname)


# else:
#     print_info_("No channel names provided in parsed.channel_names.")
# print_info_("=== SETTING TRIVIAL PRODUCERS (LIVE) ======="[0:40]+"|1")

# # ------------------------------------------------------------------------------
# # Optionally create VTPD extractors for each channel
# # ------------------------------------------------------------------------------
# extractors = {}
# if parsed.VTKextract == "ON" and parsed.channel_names:
#     print_info_("=== SETTING VTK DATA EXTRAXCTION================"[0:40]+"|0")
#     for cname, reader in _sources.items():
#         extractors[cname] = create_VTPD_extractor(cname, reader, 1)
#     print_info_("=== SETTING VTK DATA EXTRACTION==================="[0:40]+"|1")



# NEW / ALT APPROAH:

# Find proxies and set up pipelines in the global scope
pm = servermanager.ProxyManager()
for cname in parsed.channel_names:
    # proxy = pm.GetProxy("sources", cname) 
    # if not proxy:
    #     _log(f"WARNING: Could not find auto-generated proxy for channel '{cname}'")
    #     continue
    # _log(f"Found auto-generated source proxy: {cname} ({proxy.GetXMLName()})")
    proxy = PVTrivialProducer(registrationName=cname)
    proxy.UpdatePipeline()
    _sources[cname] = proxy

    if options.EnableCatalystLive:
        # _log(f"Live-Vis on '{cname}'")
        pass

        

    # filters for particle proxies
    if "particles" in cname:
        _log(f"Creating ExtractBlock filter to select main particle block for '{cname}'")

        particles = ExtractBlock(
            registrationName=f"{cname[15:]}_bunch",
            Input=proxy,
            Selectors=['//main']
        )
        helper = ExtractBlock(
            registrationName=f"{cname[15:]}_box",
            Input=proxy,
            Selectors=['//help']
            # Assembly= not needed....'
        )
            
        # try:
        #     particles.MaintainStructure = 1
        # except Exception:
        #     pass
        # helper = proxy.GetSubsetDataInformation(0, "//block_helper", "Hierarchy")
        # main = proxy.GetSubsetDataInformation(0, "//block_main", "Hierarchy")
        
        particles.UpdatePipeline()
        helper.UpdatePipeline()

        _filters[cname+"main"] = particles
        _filters[cname+"help"] = helper

        if options.EnableCatalystLive:
            pass




    if "sField" in cname:
    # Assumes channels are structured grids
        if options.EnableCatalystLive:

            _log("   -> Using MergeBlocks for structured scalar field data.")
            merged = MergeBlocks(registrationName=cname[12:]+'_MergedBlocks',
                                 Input=proxy)
            merged.MergePartitionsOnly = 1
            Show(merged)

            _log("   -> Using CellDataToPointtData for structured scalar field data.")
            cell2point = CellDatatoPointData(registrationName=cname[12:]+'_Cell2Point', 
                                             Input=merged)
            # glyphShow = 
            Show(cell2point)

            # cell2point.CellDataArraytoprocess = ['RankID', 'density']
            _filters[cname[12:]+"_merge"] = merged
            _filters[cname[12:]+"_glyph"] = merged


            # glyphShow.ColorArrayName = ['CELLS', cname[12:]] 
            # lut = GetColorTransferFunction(cname[12:])
            # lut.ApplyPreset('Cool to Warm', True)
            # lut.RescaleTransferFunctionToDataRange(True, False)
            # glyphShow.LookupTable = lut


    if "vField" in cname:
        if options.EnableCatalystLive:
            
            # Assumes channels are structured grids
            _log("   -> Using MergeBlocks for structured vector field data.")
            merged = MergeBlocks(registrationName=cname[12:]+'_MergedBlocks',Input=proxy)
            merged.MergePartitionsOnly = 1
            Show(merged)

            _log("   -> Using Glyph for structured vector field data.")
            glyph = Glyph(registrationName=cname[12:]+'_Glyph', Input=merged, GlyphType='Arrow')
            glyph.OrientationArray = ['CELLS', cname[12:]]
            # glyphShow = 
            Show(glyph)

            
            _filters[cname + "_merged"] = merged
            _filters[cname + "_glyph"]  = glyph

            # glyphShow.ColorArrayName = ['CELLS', 'RankID']
            # ... colour enforcemen nt possible....



    if parsed.VTKextract == "ON":
        _log(f"Attaching VTPD extractor to raw proxy '{cname}'")
        _extractors[cname] = create_VTPD_extractor(cname, proxy, 1)






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
        if parsed.show_forward_channels == "ON":
            for sname in parsed.steer_channel_names:
                print_info_(sname)
                # All steerables share a single 0D mesh channel; reuse the same producer for each entry
                steer_channel_readers[sname] = PVTrivialProducer(registrationName="steerable_channel_0D_mesh")
        else:
            # Suppress creating pipeline objects so nothing appears in the GUI
            for sname in parsed.steer_channel_names:
                print_info_(f"(hidden) {sname}")
                steer_channel_readers[sname] = None
    else:
        print_info_("No channel names provided in parsed.channel_names.")

    # EG: but not needed...
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
    pm = servermanager.ProxyManager()

    global parsed
    global steer_channels
    global _extractors
    global _sources
    global _filters
    
    # for name, channel in vis_channel_readers.items():
    #     channel.UpdatePipeline()
    #     channel.UpdateVTKObjects()
        

    # Loop over all registered vis channels
    for name, proxy in _sources.items():
    # for name in _sources.keys():
        # proxy = pm.GetProxy("sources", name)
        # if not proxy:
            #  _log(f"WARNING: proxy '{name}' not found in step {info.cycle}")
            #  continue
        
        # Manually update the proxy's pipeline.# This is necessary to make the new data 
        # available for live and to downstream filters (like the extractors).
        proxy.UpdatePipeline()
        proxy.UpdateVTKObjects()


            
    # for name_, (filter, input) in _filters.items():
            # live_filter.Input = ?? not needed
    for name_, filter in _filters.items():
            filter.UpdatePipeline()
            filter.UpdateVTKObjects()



    if parsed.steer == "ON" and  parsed.show_forward_channels == "ON":
        for name, (reader, sender) in steer_channels.items():
            # not needed proxy handles all of steering...
            reader.UpdatePipeline()
            reader.UpdateVTKObjects()
            # sender.UpdatePipeline()
            # sender.UpdateVTKObjects()



    if options.EnableCatalystLive:
        time.sleep(0.1)
            
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