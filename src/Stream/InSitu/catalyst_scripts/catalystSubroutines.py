"""! \file catalystSubroutines.py
\brief Helper utilities for Catalyst extractors and diagnostics.
\details Provides logging helpers, proxy inspection, and convenience creators
for extractors (e.g., VTPD). Intended for use by Catalyst pipeline/extractor
scripts in this package.
"""

# Utilities for Catalyst PNG extractor/view management and diagnostics


from paraview import servermanager
from paraview import simple
from paraview.simple import CreateExtractor
import sys
import math
from vtkmodules.vtkParallelCore import vtkCommunicator, vtkMultiProcessController
import array



from paraview import print_info

def hide_source_from_gui(proxy):
    """
    Removes the proxy from the 'sources' registration group.
    It remains valid in the script but disappears from the GUI Pipeline Browser.
    """
    pxm = servermanager.ProxyManager()
    # Find the name this proxy is registered under
    name = pxm.GetProxyName("sources", proxy.SMProxy)
    if name:
        pxm.UnRegisterProxy("sources", name, proxy.SMProxy)
    # pass

def hide_extractor_from_gui(proxy):
    """
    Extractors live in the 'extractors' group, not 'sources'.
    We must unregister them from there to hide them in the GUI.
    """
    # pxm = servermanager.ProxyManager()
    # name = pxm.GetProxyName("extractors", proxy.SMProxy)
    # if name:
    #     pxm.UnRegisterProxy("extractors", name, proxy.SMProxy)
    pass

# create extractor (PD=partitioned dataset...)
def create_VTPD_extractor(name, object, fr = 10):

    # create extractor (PD=partitioned dataset...)
    vTPD = CreateExtractor('VTPD', object, registrationName='VTPD_'+ name)
    # vTPD2.Trigger = 'TimeStep'  """ not needed"""
    vTPD.Trigger.Frequency = fr
    vTPD.Writer.FileName = name+'_{timestep:06d}.vtpd'
    return vTPD
    

# --- Common visualization helpers (shared by PNG extractor scripts) ---

def nice_bounds(vmin, vmax):
    """Return 'nice' min/max covering [vmin, vmax]."""
    if vmin == vmax:
        return vmin, vmax
    order = math.floor(math.log10(max(abs(vmin), abs(vmax), 1e-10)))
    scale = 10 ** order
    nice_min = math.floor(vmin / scale) * scale
    nice_max = math.ceil(vmax / scale) * scale
    return nice_min, nice_max


def nice_bounds_sym(vmin, vmax):
    """Symmetric 'nice' range around zero that covers [vmin, vmax]."""
    if vmin == vmax:
        return vmin, vmax
    order = math.floor(math.log10(max(abs(vmin), abs(vmax), 1e-10)))
    scale = 10 ** order
    nice_min = math.floor(vmin / scale) * scale
    nice_max = math.ceil(vmax / scale) * scale
    if -nice_min > nice_max:
        nice_max = -nice_min
    else:
        nice_min = -nice_max
    return nice_min, nice_max


def set_camera(view, position=None, focal_point=None, view_up=None, parallel_scale=None):
    """Convenience to set camera properties if provided."""
    if position is not None:
        view.CameraPosition = position
    if focal_point is not None:
        view.CameraFocalPoint = focal_point
        # Keep rotation centered on focal point
        try:
            view.CenterOfRotation = focal_point
        except Exception:
            pass
    if view_up is not None:
        view.CameraViewUp = view_up
    if parallel_scale is not None:
        view.CameraParallelScale = parallel_scale


def auto_camera_from_bounds(view, bounds, distance_factor=1.5, parallel_factor=0.6):
    """Position camera looking from a diagonal direction based on bounds."""
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)

    direction = [1.0, 1.3, 0.6]
    # direction = [1.0, 0,0]
    # direction = [0,1,0]


    norm = math.sqrt(sum(d*d for d in direction))
    direction = [d / norm for d in direction]
    distance = distance_factor * diagonal

    cam_pos = [
        cx + direction[0] * distance,
        cy + direction[1] * distance,
        cz + direction[2] * distance,
    ]
    # cam_pos = [
    #     10,
    #     cy + direction[1] * distance,
    #     0,
    # ]
    # cam_pos = [
    #     cx + direction[0] * distance,
    #     0,
    #     0
    # ]


    set_camera(
        view,
        position=cam_pos,
        focal_point=[cx, cy, cz],
        view_up=[0, 0, 1],
        parallel_scale=parallel_factor * diagonal,
    )


def compute_bounding_box_scale(bounds):
    """Return diagonal length of the bounds box."""
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def get_global_range(local_min, local_max):
    """Reduce local min/max across ranks; falls back to local in serial."""
    controller = vtkMultiProcessController.GetGlobalController()
    if not controller or controller.GetNumberOfProcesses() == 1:
        return local_min, local_max

    g_min = [0.0]
    g_max = [0.0]
    controller.AllReduce([local_min], g_min, 1, vtkCommunicator.MIN_OP)
    controller.AllReduce([local_max], g_max, 1, vtkCommunicator.MAX_OP)
    return g_min[0], g_max[0]


def get_global_spatial_bounds(local_bounds):
    """Reduce [xmin, xmax, ymin, ymax, zmin, zmax] across ranks; serial-safe."""
    controller = vtkMultiProcessController.GetGlobalController()
    if not controller or controller.GetNumberOfProcesses() == 1:
        return local_bounds

    # Use typed double arrays to avoid ambiguous overload selection in wrappers
    gb = [0.0] * 6
    for i in (0, 2, 4):
        send = array.array('d', [float(local_bounds[i])])
        recv = array.array('d', [0.0])
        controller.AllReduce(send, recv, 1, vtkCommunicator.MIN_OP)
        gb[i] = float(recv[0])
    for i in (1, 3, 5):
        send = array.array('d', [float(local_bounds[i])])
        recv = array.array('d', [0.0])
        controller.AllReduce(send, recv, 1, vtkCommunicator.MAX_OP)
        gb[i] = float(recv[0])
    return tuple(gb)

# not meaningful for extent definitions ...
# def get_global_extent(local_extent):
#     """Reduce integer extent [imin, imax, jmin, jmax, kmin, kmax] across ranks."""
#     controller = vtkMultiProcessController.GetGlobalController()
#     # Ensure integer tuple in serial
#     if not controller or controller.GetNumberOfProcesses() == 1:
#         return tuple(int(x) for x in local_extent)

#     ge = [0] * 6
#     # Use typed int arrays to avoid ambiguous overload selection in wrappers
#     for i in (0, 2, 4):
#         send = array.array('i', [int(local_extent[i])])
#         recv = array.array('i', [0])
#         controller.AllReduce(send, recv, 1, vtkCommunicator.MIN_OP)
#         ge[i] = int(recv[0])
#     for i in (1, 3, 5):
#         send = array.array('i', [int(local_extent[i])])
#         recv = array.array('i', [0])
#         controller.AllReduce(send, recv, 1, vtkCommunicator.MAX_OP)
#         ge[i] = int(recv[0])
#     return tuple(ge)



# --- lightweight logging with levels ---
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "NONE": 100}
_LOG_LEVEL = _LEVELS["INFO"]


def set_log_level(level: str):
    """Set global log level: one of 'DEBUG' | 'INFO' | 'WARN' | 'ERROR' | 'NONE'"""
    global _LOG_LEVEL
    if not isinstance(level, str):
        return
    lvl = _LEVELS.get(level.upper())
    if lvl is not None:
        _LOG_LEVEL = lvl


def get_log_level() -> str:
    inv = {v: k for k, v in _LEVELS.items()}
    return inv.get(_LOG_LEVEL, "INFO")


def _log(msg: str, level: str = "INFO"):
    # respect log level threshold
    lvl = _LEVELS.get(level.upper(), _LEVELS["INFO"])
    if lvl < _LOG_LEVEL:
        return
    try:
        s = str(msg)
    except Exception:
        s = msg
    # Collapse multiple trailing newlines into exactly one
    s = s.rstrip("\r\n")
    sys.stdout.write(s + "\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass


def print_proxy_overview():
    # _log("====Printing Proxy Overview  ===========", "INFO")
    pm = servermanager.ProxyManager()
    _log("Available 'sources' proxies:", "INFO")
    for (proxy_name, _), proxy_id in pm.GetProxiesInGroup("sources").items():
        proxy = pm.GetProxy("sources", proxy_name)
        _log(f" - ProxyPrint: {proxy})", "INFO")
        _log(f" - Proxy Name: {proxy_name}", "INFO")
        _log(f"   - XML Label: {proxy.GetXMLLabel()}", "INFO")
        _log(f"   - XML Group: {proxy.GetXMLGroup()}", "INFO")
        _log(f"   - Class Name: {proxy.GetXMLName()}", "INFO")
        _log(f"   - Properties:", "INFO")
        for prop_name in proxy.ListProperties():
            _log(f"     - {prop_name}", "INFO")


    # _log("Available 'extractors' proxies:", "INFO")
    # for (proxy_name, _), proxy_id in pm.GetProxiesInGroup("extractors").items():
    #     proxy = pm.GetProxy("extractors", proxy_name)
    #     _log(f" - ProxyPrint: {proxy})", "INFO")
    #     _log(f" - Proxy Name: {proxy_name}", "INFO")
    #     _log(f"   - XML Label: {proxy.GetXMLLabel()}", "INFO")
    #     _log(f"   - XML Group: {proxy.GetXMLGroup()}", "INFO")
    #     _log(f"   - Class Name: {proxy.GetXMLName()}", "INFO")
    #     _log(f"   - Properties:", "INFO")
    #     for prop_name in proxy.ListProperties():
    #         _log(f"     - {prop_name}", "INFO")

def find_source_by_name(base_name):
    """
    Robustly finds a source in the pipeline.
    """
    s = simple.FindSource(base_name)
    if s: return s
    
    sources = simple.GetSources()
    for (group, name), proxy in sources.items():
        if name.startswith(base_name):
            return proxy
    return None
    
def get_available_extract_names(catalyst_link):
    """
    Queries the Catalyst In-Situ Proxy Manager for available channels.
    """
    if not catalyst_link:
        return []

    try:
        # Access the raw C++ object to get the InSitu Manager
        if hasattr(catalyst_link, "GetInsituProxyManager"):
            pm = catalyst_link.GetInsituProxyManager()
        else:
            pm = catalyst_link.GetClientSideObject().GetInsituProxyManager()

        if pm is None:
            return []

        # The 'sources' group in the Insitu Manager contains the available extracts
        num_proxies = pm.GetNumberOfProxies("sources")
        
        all_names = []
        for idx in range(num_proxies):
            name = pm.GetProxyName("sources", idx)
            if name:
                all_names.append(name)
        
        print(f"DEBUG: All found sources: {sorted(list(set(all_names)))}")

        names = []
        for name in all_names:
            # Filter out the generated filters (ending in .bunch, .box, .MergedBlocks, .Cell2Point, .Glyph)
            if any(name.endswith(suffix) for suffix in [".bunch", ".box", ".MergedBlocks", ".Cell2Point", ".Glyph", ".ResampleToImage"]):
                continue
            names.append(name)

        return sorted(list(set(names)))

    except Exception as e:
        print(f"[Error] Failed to query InsituProxyManager: {e}")
        return []

