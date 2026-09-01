"""! \file pipeline_default.py
\brief Main ParaView Catalyst pipeline: live visualization, VTK extracts, and steering.
\details Discovers channel proxies, wires optional extractors, updates live views,
and forwards/fetches steerable parameters between the simulation an        
"""

# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html
# BUG: if pv client is opened after Simulation has completed first execute
# the created PNG extractor show up in the clients pipeline browser
# Iam not sure how to avoid this. This will break the View needed for
# the extractor to work properly and png extaction starrts failing ca
# 3 timesteps after the client was opened.
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
    GetColorTransferFunction,
    ResampleToImage
    # LoadPlugin,
    # ExtractSubset,
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
    get_global_spatial_bounds
    # ,
    # create_VTPD_extractor
)
#### disable automatic camera rest on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


from paraview.simple import CreateExtractor

# ------------------------------------------------------------------------------
# Extractor factory functions
# ------------------------------------------------------------------------------
# IMPORTANT: Match writer type to Conduit data structure:
#   - Conduit "mesh" (uniform/structured) → VTPD (homogeneous partitions)
#   - Conduit "multimesh" (heterogeneous blocks) → VTM (MultiBlock)
# Using wrong writer causes: "Can not execute ... without output ports" error
# ------------------------------------------------------------------------------

def create_VTPD_extractor(name, object, fr = 10):
    """Create a VTPD extractor for simple mesh data (uniform/structured grids).
    
    Use for: scalar fields, vector fields (Conduit type: "mesh")
    Output: .vtpd files (XML Partitioned Dataset)
    """
    vTPD = CreateExtractor('VTPD', object, registrationName='VTPD_'+ name)
    vTPD.Trigger.Frequency = fr
    vTPD.Writer.FileName = 'ippl_'+name+'_{timestep:06d}.vtpd'
    return vTPD

def create_VTM_extractor(name, object, fr = 10):
    """Create a VTPC extractor for multimesh particle data (Partitioned Dataset Collection).
    
    Use for: particle data with multiple block types (Conduit type: "multimesh")
    Output: .vtpc files (XML Partitioned Dataset Collection)
    Note: Conduit multimesh → vtkPartitionedDataSetCollection (not vtkMultiBlockDataSet)
    """
    # Use VTPC (Partitioned Dataset Collection) not VTM (MultiBlock)
    vTPC = CreateExtractor('VTPC', object, registrationName='VTPC_'+ name)
    vTPC.Trigger.Frequency = fr
    vTPC.Writer.FileName = 'ippl_'+name+'_{timestep:06d}.vtpc'
    return vTPC

# Option 2: Extract only specific blocks (e.g., just particles without helper)
        # Uncomment the following to extract only the main particle block:
        # if parsed.VTKextract == "ON":
        #     particles_block = ExtractBlock(
        #         registrationName=f"{cname}_main_extract",
        #         Input=proxy, 
        #         Selectors=['//main']
        #     )
        #     particles_block.UpdatePipeline()
        #     _log(f"Attaching VTPD extractor to MAIN particle block (via MergeBlocks) '{cname}'")
        #     _extractors[cname+"_main_only"] = create_extractor_from_single_block(cname+"_main", particles_block, 1)UI.
# Designed to orchestrate the per-channel extractor scripts in catalyst_extractors/.

# def create_extractor_from_single_block(name, extract_block_filter, fr = 10):
#     """Create an extractor for a single block from multimesh.
    
#     Use for: extracting just one block (e.g., only particles, no helper)
#     Strategy: ExtractBlock → MergeBlocks to flatten → appropriate writer
    
#     Args:
#         name: base name for output files
#         extract_block_filter: an ExtractBlock filter with Selectors set
#         fr: extraction frequency
    
#     Output: .vtpd or .vtu files depending on block content
#     """
#     # Merge blocks to flatten the extracted subset into a single partitioned dataset
#     merged = MergeBlocks(
#         registrationName=f"{name}_merged",
#         Input=extract_block_filter
#     )
#     merged.MergePartitionsOnly = 1  # Keep partitions, just flatten hierarchy
#     merged.UpdatePipeline()
    
#     # Now use VTPD which should work on the flattened structure
#     vTPD = CreateExtractor('VTPD', merged, registrationName='VTPD_'+ name)
#     vTPD.Trigger.Frequency = fr
#     vTPD.Writer.FileName = 'ippl_'+name+'_{timestep:06d}.vtpd'
#     return vTPD


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





#### disable automatic camera rest on 'Show'
# paraview.simple._DisableFirstRenderCameraReset()
colorPalette = GetSettingsProxy('ColorPalette')


colorPalette.BackgroundColorMode = 'Gradient'

colorPalette.Background = [0.0, 0.0, 0.0]
# colorPalette.Background = [0.0, 0.8, 1.0]
# colorPalette.Background = [0.3333333333333333, 1.0, 1.0]


colorPalette.Background2 = [0.9, 0.9, 0.9]
# colorPalette.Background2 = [0.97, 0.97, 0.97]

# # Properties modified on colorPalette
colorPalette.Foreground = [0.8, 0.0, 0.0]
colorPalette.Edges = [1.0, 0.0, 0.0]
colorPalette.Surface = [1.0, 0.0, 0.0]


# # 1. Set default representation to "Points"
# renderViewSettings.DefaultRepresentation = 'Points'
# colorPalette.Surface = [1.0, 0.0, 0.0]  # [R, G, B] for Red
# renderViewSettings.DefaultPointSize = 10
# renderViewSettings.DefaultRenderPointsAsSpheres = 1  # 1 means True


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
parser.add_argument("--live",       default="OFF", help="Enable options.CatalystLive")
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

if parsed.live == "ON":
    options.EnableCatalystLive = 1
else:
    options.EnableCatalystLive = 0


    

options.GlobalTrigger = 'Time Step'
options.CatalystLiveTrigger = 'Time Step'
options.CatalystLiveURL = 'localhost:22222' #is also default
options.ExtractsOutputDirectory = 'data_vtk_extracts_' + exp_string


steer_enabled = parsed.live == "ON" and parsed.steer == "ON"

# ------------------------------------------------------------------------------
# Proactively remove the forward 0D mesh source from the GUI if it was auto-created
# ------------------------------------------------------------------------------
if steer_enabled and parsed.show_forward_channels == "OFF":
    try:
        src = paraview.simple.FindSource("steerable_channel_0D_mesh")
        if src is not None:
            print_info_("Hiding auto-created steerable_channel_0D_mesh from GUI")
            paraview.simple.Delete(src)
    except Exception as e:
        print_info_(f"Could not hide steerable_channel_0D_mesh: {e}")

    # Additionally hide any per-array forward meshes inferred from steer_channel_names
    try:
        if parsed.steer_channel_names:
            for sname in parsed.steer_channel_names:
                # Deduce forward mesh registration name for std::vector-based steerables
                # Labels are normalized on the C++ side to start with 'array:' for vectors.
                reg_name = None
                if isinstance(sname, str) and sname.startswith("array:"):
                    # namespace prefix before first '.' (e.g., 'array:LinMaps.time' -> 'LinMaps')
                    # ns = sname[6:]
                    ns = sname

                    dot = ns.find('.')
                    if dot != -1:
                        ns = ns[:dot]
                    reg_name = f"steerable_channel_1D_mesh_{ns}"

                if reg_name:
                    src_arr = paraview.simple.FindSource(reg_name)
                    if src_arr is not None:
                                print_info_(f"Hiding per-array forward mesh '{reg_name}' from GUI")
                                paraview.simple.Delete(src_arr)
    except Exception as e:
        print_info_(f"Error while hiding inferred per-array forward meshes: {e}")



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


    if "particles" in cname:
        # For Live visualization: create ExtractBlock filters to show specific blocks
        if options.EnableCatalystLive:
            _log(f"Creating ExtractBlock filter(s) for particle channel '{cname}' (Live view)")
            particles = ExtractBlock(
                # registrationName=f"{cname[15:]}.bunch",
                registrationName=f"{cname}.bunch",
                Input=proxy,
                # // fails to enable visualisation by attributes exclusive to this block
                Selectors=['//main', '//block_main'] 
                # Selectors=[]
            )
            helper = ExtractBlock(
                # registrationName=f"{cname[15:]}_box",
                registrationName=f"{cname}.box",
                Input=proxy,
                Selectors=['//help', '//block_help']
            )
            particles.UpdatePipeline()
            helper.UpdatePipeline()

            _filters[cname+"_main"] = particles
            _filters[cname+"_help"] = helper
            
            Show(particles)
            Show(helper)


        # Particles come as multimesh with block_main and block_help
        # Conduit multimesh → vtkPartitionedDataSetCollection → use VTPC writer
        # Option 1: Extract entire multimesh (all blocks together)
        if parsed.VTKextract == "ON":
            _log(f"Attaching VTPC extractor to complete multimesh particle proxy '{cname}'")
            _extractors[cname] = create_VTM_extractor(cname, proxy, 1)
        


    if "sField" in cname:
        if options.EnableCatalystLive:

            _log("   -> Using MergeBlocks for structured scalar field data.")
            # merged = MergeBlocks(registrationName=cname[12:]+'_MergedBlocks',
            merged = MergeBlocks(registrationName=cname +'.MergedBlocks',
                                 Input=proxy)
            merged.MergePartitionsOnly = 1
            Show(merged)

            _log("   -> Using CellDataToPointtData for structured scalar field data.")
            # cell2point = CellDatatoPointData(registrationName=cname[12:]+'.Cell2Point', 
            cell2point = CellDatatoPointData(registrationName=cname+'.Cell2Point', 
                                             Input=merged)
            Show(cell2point)
            
            _log("   -> Using ResampleToImage for structured scalar field data.")
            
            # Calculate dimensions (logic from png_ext_sfield.py)
            info = proxy.GetDataInformation()
            local_bounds = info.GetBounds()
            local_extent = info.GetExtent()
            global_bounds = get_global_spatial_bounds(local_bounds)
            
            nx = (local_extent[1] - local_extent[0] + 1)
            ny = (local_extent[3] - local_extent[2] + 1)
            nz = (local_extent[5] - local_extent[4] + 1)
            
            lx = (local_bounds[1] - local_bounds[0])
            ly = (local_bounds[3] - local_bounds[2])
            lz = (local_bounds[5] - local_bounds[4])
            
            spacing_x = lx / max(nx - 1, 1)
            spacing_y = ly / max(ny - 1, 1)
            spacing_z = lz / max(nz - 1, 1)
            
            dx = max(spacing_x, 1e-12)
            dy = max(spacing_y, 1e-12)
            dz = max(spacing_z, 1e-12)


            ghost_x = 1
            ghost_y = 1
            ghost_z = 1

            dim_x = int(round((global_bounds[1] - global_bounds[0]) / dx)) - 2*ghost_x
            dim_y = int(round((global_bounds[3] - global_bounds[2]) / dy)) - 2*ghost_y
            dim_z = int(round((global_bounds[5] - global_bounds[4]) / dz)) - 2*ghost_z
            global_extent = [dim_x, dim_y, dim_z]
            print(global_extent)


            
            resample = ResampleToImage(registrationName=cname+'.ResampleToImage', Input=merged)
            resample.UseInputBounds = 1
            # resample.SamplingBounds = global_bounds
            resample.SamplingDimensions = global_extent
            Show(resample)

            # cell2point.CellDataArraytoprocess = ['RankID', 'density']
            _filters[cname[12:]+"_merge"] = merged
            _filters[cname[12:]+"_c2p"] = merged
            _filters[cname[12:]+"_resample"] = resample

        if parsed.VTKextract == "ON":
            # _log(f"Attaching VTPD extractor to proxy '{cname}'")
            _extractors[cname] = create_VTPD_extractor(cname, proxy, 1)

        


    if "vField" in cname:
        if options.EnableCatalystLive:
            
            _log("   -> Using MergeBlocks for structured vector field data.")
            merged = MergeBlocks(registrationName=cname+'.MergedBlocks',Input=proxy)
            merged.MergePartitionsOnly = 1
            Show(merged)

            _log("   -> Using Glyph for structured vector field data.")
            glyph = Glyph(registrationName=cname +'.Glyph', Input=merged, GlyphType='Arrow')
            glyph.OrientationArray = ['CELLS', cname[12:]]
            # glyphShow = 
            Show(glyph)
            
            _filters[cname + "_merged"] = merged
            _filters[cname + "_glyph"]  = glyph

        if parsed.VTKextract == "ON":
            # _log(f"Attaching VTPD extractor to proxy '{cname}'")
            _extractors[cname] = create_VTPD_extractor(cname, proxy, 1)

        



# ------------------------------------------------------------------------------
# Setup steering channels
# ------------------------------------------------------------------------------

steer_channel_readers = {}
steer_channel_senders = {}
steer_channels = {}

if steer_enabled:
    if parsed.steer_channel_names :
        print_info_("===CREATING STEERABLES============="[0:40]+"|0")

#         # ------------------------------------------------------------------------------
#         # forward / incoming steering channels
#         # ------------------------------------------------------------------------------
#         print_info_("FORWARD")
#         if parsed.show_forward_channels == "ON":
#             for sname in parsed.steer_channel_names:
#                 print_info_(sname)
#                 # Default forward mesh for scalar steerables
#                 reg = "steerable_channel_0D_mesh"
#                 if "_" in sname:
#                     pref = sname.split('_',1)[0]
#                     # Only treat as dynamic array if prefix contains no '.' (exclude struct scalar members like simp.temperature)
#                     if '.' not in pref:
#                         reg = f"steerable_channel_1D_mesh_{pref}"
#                 steer_channel_readers[sname] = PVTrivialProducer(registrationName=reg)
#         else:
#             # Suppress creating readers entirely so channels stay hidden
#             for sname in parsed.steer_channel_names:
#                 print_info_(f"(hidden) {sname}")
#                 steer_channel_readers[sname] = None
    else:
        print_info_("No channel names provided in parsed.channel_names.")


    # ------------------------------------------------------------------------------
    # backward / outgoing steering channels
    # ------------------------------------------------------------------------------
    print_info_("BACKWARD")
    try:
        # Unified sender: one proxy carrying one property per channel, single result mesh
        sender_all = CreateSteerableParameters(
            steerable_proxy_type_name           = "SteerableParameters_SCALARS",
            steerable_proxy_registration_name   = "SteeringParameters_SCALARS",
            result_mesh_name                    = "steerable_channel_backward_all"
        )
        # print_info_("[DEBUG] Created sender_all (LinMaps + generic) proxy object", level=2 )
      
        # print_info_("[DEBUG] Created sender_all2 (SimParams scalar struct) proxy object", level=2 )



        # if sender3 is None:
            # print_info_("[DEBUG][WARN] Failed to create hard-coded struct array sender 'simpVec'", level=2)
        # else:
            # print_info_("[DEBUG] Created hard-coded struct array sender 'simpVec'", level=2)
        if sender_all is None:
            print_info_("Error: SteerableParameters_ALL proxy not found (CreateSteerableParameters returned None).")
        else:
            print_info_("SteerableParameters_ALL loaded successfully.")
        



        # Dynamically detect struct-array namespaces from steer_channel_names with prefix 'array:'
        # Example entries: 'array:LinMaps.time', 'array:simpVec.temperature'
        struct_array_senders = {}
        detected_namespaces = set()
        try:
            for entry in (parsed.steer_channel_names or []):
                if not isinstance(entry, str):
                    continue
                if not entry.startswith("array:"):
                    continue
                # Extract namespace between ':' and first '.'; fallback to full tail if no '.'
                tail = entry.split(":", 1)[1]
                ns = tail.split(".", 1)[0] if "." in tail else tail
                if ns:
                    detected_namespaces.add(ns)
        except Exception as e:
            print_info_(f"[DEBUG][WARN] Failed to parse struct-array namespaces: {e}", level=2)

        # Create one sender per detected namespace (e.g., 'LinMaps', 'simpVec')
        for ns in sorted(detected_namespaces):
            sender = CreateSteerableParameters(
                steerable_proxy_type_name         = f"SteerableParameters_{ns}",
                steerable_proxy_registration_name = f"Steering_{ns}",
                result_mesh_name                  = "steerable_channel_backward_all"
            )
            struct_array_senders[ns] = sender

            if sender is None:
                print_info_(f"[DEBUG][WARN] Failed to create struct array sender '{ns}'", level=2)
            else:
                print_info_(f"[DEBUG] Created struct array sender '{ns}'", level=2)

    except Exception as e:
        print_info_(f"Exception while loading (backward) SteerableParameters: {e}")
    
    print_info_("===CREATING STEERABLES=============="[0:40]+"|1")



# for sname in parsed.steer_channel_names:
#     steer_channels[sname] = (steer_channel_readers[sname], steer_channel_senders[sname])



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
# Manually update the proxy's pipeline.# This is necessary to make the new data 
# available for live and to downstream filters (like the extractors). Not sure when why vtk
#  update is needed. Extractors objects themselves don't need to be updated i think.
# Inputs also don't need to be reset. Backwards channels are entirely managed by the
# proxy file so? no update calls needed?...
def catalyst_execute(info):
    print_info_("_________executing (cycle={}, time={})___________".format(info.cycle, info.time))
    print_info_("catalyst_execute()::"+exp_string)
    pm = servermanager.ProxyManager()

    global parsed
    global steer_channels
    global _extractors
    global _sources
    global _filters
        

    # Loop over all registered vis channels
    for name, proxy in _sources.items():
    # for name in _sources.keys():
        # proxy = pm.GetProxy("sources", name)
        # if not proxy:
            #  _log(f"WARNING: proxy '{name}' not found in step {info.cycle}")
            #  continue
        
        proxy.UpdatePipeline()
        proxy.UpdateVTKObjects()


    for name_, filter in _filters.items():
            filter.UpdatePipeline()
            filter.UpdateVTKObjects()



    if steer_enabled and  parsed.show_forward_channels == "ON":
        for name, reader in steer_channel_readers.items():
        # for name, (reader, sender) in steer_channels.items():
            reader.UpdatePipeline()
            reader.UpdateVTKObjects()

    # Not needed anymore ... 
    # # Always update backward senders to ensure result mesh generation
    # if steer_enabled:
    #     # Unique sender proxies
    #     updated = set()
    #     for sender in steer_channel_senders.values():
    #         if sender is None: continue
    #         if sender in updated: continue
    #         try:
    #             sender.UpdatePipeline()
    #             sender.UpdateVTKObjects()
    #             pname = getattr(sender, 'GetXMLName', lambda : 'unknown')()
    #             print_info_(f"[DEBUG] Updated backward sender proxy '{pname}'", level=2)
    #         except Exception as e:
    #             print_info_(f"[DEBUG][WARN] Failed to update sender proxy: {e}", level=2)
    #         updated.add(sender)

    #     # Debug: list properties present on each sender
    #     for sender in updated:
    #         try:
    #             pname = getattr(sender, 'GetXMLName', lambda : 'unknown')()
    #             props = [p for p in dir(sender) if not p.startswith('_') and hasattr(sender, p)]
    #             print_info_(f"[DEBUG] Sender '{pname}' has properties: {props[:25]} ...", level=2)
    #         except Exception as e:
    #             print_info_(f"[DEBUG][WARN] Could not enumerate properties for sender: {e}", level=2)



    if options.EnableCatalystLive:
        # time.sleep(1)
        pass
            
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