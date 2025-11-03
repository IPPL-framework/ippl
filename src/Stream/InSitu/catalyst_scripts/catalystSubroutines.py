"""! \file catalystSubroutines.py
\brief Helper utilities for Catalyst extractors and diagnostics.
\details Provides logging helpers, proxy inspection, and convenience creators
for extractors (e.g., VTPD). Intended for use by Catalyst pipeline/extractor
scripts in this package.
"""

# Utilities for Catalyst PNG extractor/view management and diagnostics


from paraview import servermanager
from paraview.simple import CreateExtractor
import sys
# create extractor (PD=partitioned dataset...)
def create_VTPD_extractor(name, object, fr = 10):

    # create extractor (PD=partitioned dataset...)
    vTPD = CreateExtractor('VTPD', object, registrationName='VTPD_'+ name)
    # vTPD2.Trigger = 'TimeStep'  """ not needed"""
    vTPD.Trigger.Frequency = fr
    vTPD.Writer.FileName = 'ippl_'+name+'_{timestep:06d}.vtpd'
    return vTPD
    



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







def load_state_module(module_path_or_name: str):
    """Import a Catalyst state module and auto-register its PNG extractors."""
    import importlib, sys, os
    mod = None
    try:
        if module_path_or_name.endswith('.py') and os.path.exists(module_path_or_name):
            # import by path
            dirname = os.path.dirname(module_path_or_name)
            fname = os.path.splitext(os.path.basename(module_path_or_name))[0]
            if dirname not in sys.path:
                sys.path.append(dirname)
            mod = importlib.import_module(fname)
        else:
            mod = importlib.import_module(module_path_or_name)
        register_png_extractor(mod)
        return mod
    except Exception as e:
        _log(f"[REGISTER][ERROR] Failed to load/register module '{module_path_or_name}': {e}", "ERROR")
        return None
