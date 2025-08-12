# script-version: 2.0
# for more details check https://www.paraview.org/paraview-docs/latest/cxx/CatalystPythonScriptsV2.html
from paraview import print_info
from paraview.simple import *
from paraview import catalyst
import time

import os
from paraview.simple import LoadPlugin, CreateSteerableParameters
from paraview import servermanager

# import catalyst_subroutines as cr

from paraview.simple import LoadPlugin

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


# print start marker




# not needed with running pvserver i think?
# from paraview.catalyst import bridge
# Connect to ParaView Live.
# The host should be the hostname of the login node where you created the tunnel.
# 'localhost' often works if the compute node can resolve it as the login node.
# The port must match the one used in the -R tunnel command.
# bridge.connect(host='localhost', port=22222)
# bridge.connect(host='jrc0193i', port=22222) no module named connect??->hallucinatio  maybe or deprecated ...



print_info("start'%s'", __name__)
print("==========================='%s'")
print("EXECUTING CATALYST PIPELINE'%s'")
print("==========================='%s'")


""" 
ERROR: In vtkSIProxyDefinitionManager.cxx, line 517
vtkSIProxyDefinitionManager (0x542cc80): No proxy that matches: group=sources and proxy=STEERING were found.

ERROR: In vtkSMDeserializerXML.cxx, line 41
vtkSMStateLoader (0x2bd1e00): Could not create a proxy of group: sources type: STEERING 
"""

    
# ======================================================
# ======================================================
# ======================================================





# print("\n\n--- CATALYST SCRIPT DEBUGGING ---")
# proxy_xml_path = "/p/home/jusers/klapproth1/jureca/sshfs/insituvis209/proxy.xml" 

# print(f"Attempting to load plugin from path: '{proxy_xml_path}'")
# print(f"Checking if file exists at that path: {os.path.exists(proxy_xml_path)}")

# # Attempt to load the plugin and catch any errors
# try:
#     LoadPlugin(proxy_xml_path, ns=globals())
#     print("LoadPlugin() command executed without raising an exception.")
# except Exception as e:
#     print(f"CRITICAL ERROR: LoadPlugin() FAILED with an exception: {e}")

# # Now, let's list all known source proxies to see if ours is in the list.
# print("\n--> Listing all 'sources' proxies known to ParaView after the load attempt:")
# pm = servermanager.ProxyManager()
# available_sources = []
# for (proxy_name, _), proxy_id in pm.GetProxiesInGroup("sources").items():
#     available_sources.append(proxy_name)

# # Sort and print the list for clarity
# available_sources.sort()
# for source_name in available_sources:
#     print(f"    - {source_name}")

# # Check specifically for our proxy
# if "SteerableParameters" in available_sources:
#     print("\nSUCCESS: 'SteerableParameters' was found in the list of available sources.")
# else:
#     print("\nFAILURE: 'SteerableParameters' was NOT found in the list of available sources.")

# print("--- END OF DEBUGGING BLOCK ---\n\n")
# # =================== END DEBUGGING BLOCK ===================
# print_proxy_overview()





# steerable_parameters = None
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


def print_proxy_overview():
    
    print("\n\n\n\n ========Printing Proxy Overview  ========================")
    pm = servermanager.ProxyManager()
    print("Available 'sources' proxies:")
    for (proxy_name, _), proxy_id in pm.GetProxiesInGroup("sources").items():
        proxy = pm.GetProxy("sources", proxy_name)
        print(f" - ProxyPrint: {proxy})")
        print(f" - Proxy Name: {proxy_name}")
        print(f"   - XML Label: {proxy.GetXMLLabel()}")
        print(f"   - XML Group: {proxy.GetXMLGroup()}")
        print(f"   - Class Name: {proxy.GetXMLName()}")
        print(f"   - Properties:")
        for prop_name in proxy.ListProperties():
            print(f"     - {prop_name}")
    print("========DONe========================\n\n\n\n")
# cr.
print_proxy_overview()
# 




# pm = servermanager.ProxyManager()
""" steering_parameters = pm.GetProxy("sources", "SteeringParameters")
if steering_parameters is None:
    print("\n[ERROR] 'SteeringParameters' proxy not found.")
else:
    print("✅ 'SteeringParameters' proxy loaded successfully.")
    print("Available properties:")
    for prop in steerable_parameters.ListProperties():
        print(f"  - {prop}")
 """




print("====SETTING ACTIVE SOURCES========")
# registrationName must match the channel name used in the 'CatalystAdaptor'.
ippl_field     = PVTrivialProducer(registrationName='ippl_E')
ippl_field_scalar = PVTrivialProducer(registrationName='ippl_scalar')
# ippl_field_phi = PVTrivialProducer(registrationName='ippl_phi')
ippl_particle  = PVTrivialProducer(registrationName='ippl_particle')



# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------


def create_data_extractors():
    # create extractor (U = unstructured)
    vTI1 = CreateExtractor('VTI', ippl_field, registrationName='VTI1')
    vTI1.Trigger = 'Time Step'
    vTI1.Writer.FileName = 'ippl_field_{timestep:06d}.vti'

    # create extractor (U = unstructured)
    vTI2 = CreateExtractor('VTI', ippl_field_scalar, registrationName='VTI2')
    vTI2.Trigger = 'Time Step'
    vTI2.Writer.FileName = 'ippl_field_scalar_{timestep:06d}.vti'

    # create extractor (PD=point data)
    vTPD2 = CreateExtractor('VTPD', ippl_particle, registrationName='VTPD2')
    vTPD2.Trigger = 'Time Step'
    vTPD2.Writer.FileName = 'ippl_particle_{timestep:06d}.vtpd'

create_data_extractors()




# ----------------------------------------------------------------
# restore active source
SetActiveSource(ippl_field)
SetActiveSource(ippl_field_scalar)
SetActiveSource(ippl_particle)
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
    print("-----------------------------------")
    print_info("in '%s::catalyst_execute'", __name__)

    global ippl_field
    ippl_field.UpdatePipeline()
    global ippl_field_scalar
    ippl_field_scalar.UpdatePipeline()
    global ippl_particle
    ippl_particle.UpdatePipeline()
    
    global steerable_parameters
    steerable_parameters.scaleFactor[0] = 31 + info.cycle
    print(info.cycle)

    # print("executing (cycle={}, time={})".format(info.cycle, info.time))
    #print("field bounds   :", ippl_field.GetDataInformation().GetBounds())
    #print("particle bounds:", ippl_particle.GetDataInformation().GetBounds())

    if options.EnableCatalystLive:
        time.sleep(0.5)


# ------------------------------------------------------------------------------
def catalyst_finalize():
    print_info("in '%s::catalyst_finalize'", __name__)


# print end marker
print_info("end '%s'", __name__)




































# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# # get the time-keeper
# timeKeeper1 = GetTimeKeeper()

# # initialize the timekeeper

# # get time animation track
# timeAnimationCue1 = GetTimeTrack()

# # initialize the animation track

# # get animation scene
# animationScene1 = GetAnimationScene()

# # initialize the animation scene
# animationScene1.Cues = timeAnimationCue1
# animationScene1.AnimationTime = 0.0

# initialize the animation scene








""" def list_registered_proxies():
    print("=== Registered Proxies ===")
    proxy_manager = servermanager.ProxyManager()

    for group in proxy_manager.GetGroups():
        proxies = proxy_manager.GetProxiesInGroup(group)
        if proxies:
            print(f"\nGroup: '{group}'")
            for key, proxy_info in proxies.items():
                print(f"  - Proxy name: '{key}'")

list_registered_proxies()
 """
    # In a real simulation sleep is not needed. We use it here to slow down the
    # "simulation" and make sure ParaView client can catch up with the produced
    # results instead of having all of them flashing at once.
