# Utilities for Catalyst PNG extractor/view management and diagnostics


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
    

    # Alternative: If you want to extract individual partitions as separate files,
    # you could use PVD format:
    # vPVD_field = CreateExtractor('PVD', ippl_field, registrationName='PVD_field')
    # vPVD_field.Trigger = 'Time Step'
    # vPVD_field.Writer.FileName = 'ippl_field_{timestep:06d}.pvd'

    # not working:
    # create extractor (VTU, U = unstructured)
    # create extractor (VTI, I = Image Data for regular Grids ...) 

# create extractor (PD=partitioned dataset...)
# vTPD_eg = CreateExtractor('VTPD', ippl_eg, registrationName='VTPD_eg')
# vTPD_eg.Trigger = 'TimeStep'  
# vTPD_eg.Trigger.Frequency = 10
# vTPD_eg.Writer.FileName = 'ippl_particle_{timestep:06d}.vtpd'











from paraview import servermanager

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

# print_proxy_overview()






















from paraview import servermanager

# Public-ish shared state
_PNG_REGISTRY_BY_PROXY = {}   # key: extractor proxy -> view proxy
_PNG_REGISTRY_BY_NAME = {}    # key: extractor name (str, e.g. 'PNG1') -> view proxy
_keepalive_views = []         # Keep strong refs to views so GC can’t drop them
_keepalive_extractors = []


def get_keepalive_counts():
    return (len(_keepalive_views), len(_keepalive_extractors))


def _proxy_name_in_group(group: str, proxy):
    try:
        return servermanager.ProxyManager().GetProxyName(group, proxy)
    except Exception:
        return None


def register_png_extractor(module_or_view, render_view=None, extractor=None):
    """
    Register mapping between a PNG extractor and its RenderView.

    Usage:
    - Auto-scan a state module:
        register_png_extractor(png_ext_module)
      Finds module variables like renderView* and pNG* and maps pNG<num> -> renderView<num>,
      else falls back to the first renderView in that module. Also keeps proxies alive.

    - Explicit registration:
        register_png_extractor(RenderView1, pNG1)
      or
        register_png_extractor(png_ext_module, RenderView1, pNG1)
    """
    import types, re

    def _is_proxy(obj):
        return hasattr(obj, 'GetProperty') and hasattr(obj, 'UpdateVTKObjects')

    # Explicit registration forms
    if extractor is not None and render_view is not None and _is_proxy(render_view) and _is_proxy(extractor):
        ex_name = _proxy_name_in_group('extractors', extractor)
        _PNG_REGISTRY_BY_PROXY[extractor] = render_view
        if ex_name:
            _PNG_REGISTRY_BY_NAME[ex_name if isinstance(ex_name, str) else ex_name[0]] = render_view
        vname = _proxy_name_in_group('views', render_view)
        _log(f"[REGISTER] Extractor {ex_name} -> view {vname}", "INFO")
        # keepalive
        try:
            if render_view not in _keepalive_views:
                _keepalive_views.append(render_view)
            if extractor not in _keepalive_extractors:
                _keepalive_extractors.append(extractor)
        except Exception:
            pass
        return

    # If two arguments were provided and first is view and second is extractor
    if extractor is None and render_view is not None and _is_proxy(module_or_view) and _is_proxy(render_view):
        # called as register_png_extractor(RenderView1, pNG1)
        view_proxy = module_or_view
        extractor_proxy = render_view
        ex_name = _proxy_name_in_group('extractors', extractor_proxy)
        _PNG_REGISTRY_BY_PROXY[extractor_proxy] = view_proxy
        if ex_name:
            _PNG_REGISTRY_BY_NAME[ex_name if isinstance(ex_name, str) else ex_name[0]] = view_proxy
        vname = _proxy_name_in_group('views', view_proxy)
        _log(f"[REGISTER] Extractor {ex_name} -> view {vname}", "INFO")
        # keepalive
        try:
            if view_proxy not in _keepalive_views:
                _keepalive_views.append(view_proxy)
            if extractor_proxy not in _keepalive_extractors:
                _keepalive_extractors.append(extractor_proxy)
        except Exception:
            pass
        return

    # Auto-scan module variant
    if isinstance(module_or_view, types.ModuleType):
        mod = module_or_view
        # find views and png extractors in module, keep attribute names for numeric matching
        views = []               # list of proxies
        views_by_num = {}        # num -> proxy
        png_extractors = []      # list of proxies
        png_by_num = {}          # num -> proxy

        for attr in dir(mod):
            try:
                obj = getattr(mod, attr)
            except Exception:
                continue
            if not _is_proxy(obj):
                continue
            # Detect RenderView proxies
            try:
                if hasattr(obj, 'GetXMLName') and 'RenderView' in obj.GetXMLName():
                    views.append(obj)
                    m = re.search(r'renderView(\d+)$', attr, re.IGNORECASE)
                    if m:
                        views_by_num[int(m.group(1))] = obj
                    continue
            except Exception:
                pass
            # Detect PNG extractors by writer type
            try:
                wprop = obj.GetProperty('Writer')
                if wprop and wprop.GetNumberOfProxies() > 0:
                    w = wprop.GetProxy(0)
                    wx = w.GetXMLName() if hasattr(w, 'GetXMLName') else ''
                    if 'PNG' in wx or 'Image' in wx:
                        png_extractors.append(obj)
                        m = re.search(r'pNG(\d+)$', attr)
                        if m:
                            png_by_num[int(m.group(1))] = obj
            except Exception:
                pass

        target_default_view = views[0] if views else None

        # keepalive: add discovered views/extractors
        try:
            for v in views:
                if v not in _keepalive_views:
                    _keepalive_views.append(v)
            for ex in png_extractors:
                if ex not in _keepalive_extractors:
                    _keepalive_extractors.append(ex)
        except Exception:
            pass

        # First, map by matching numbers between pNG<num> and renderView<num>
        matched = set()
        for num, ex in png_by_num.items():
            view = views_by_num.get(num, target_default_view)
            _PNG_REGISTRY_BY_PROXY[ex] = view
            ex_name = _proxy_name_in_group('extractors', ex)
            if ex_name:
                _PNG_REGISTRY_BY_NAME[ex_name if isinstance(ex_name, str) else ex_name[0]] = view
            vname = _proxy_name_in_group('views', view) if view else None
            _log(f"[REGISTER] Extractor {ex_name} -> view {vname}", "INFO")
            matched.add(ex)

        # For any remaining PNG extractors, map to default view
        for ex in png_extractors:
            if ex in matched:
                continue
            _PNG_REGISTRY_BY_PROXY[ex] = target_default_view
            ex_name = _proxy_name_in_group('extractors', ex)
            if ex_name:
                _PNG_REGISTRY_BY_NAME[ex_name if isinstance(ex_name, str) else ex_name[0]] = target_default_view
            vname = _proxy_name_in_group('views', target_default_view) if target_default_view else None
            _log(f"[REGISTER] Extractor {ex_name} -> view {vname}", "INFO")
        return

    _log("[REGISTER][WARN] Unsupported registration call; provide a module or explicit proxies", "WARN")


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


def _debug_dump_state(tag=''):
    try:
        pm = servermanager.ProxyManager()
        views = pm.GetProxiesInGroup("views")
        exts  = pm.GetProxiesInGroup("extractors")

        def _name_for_view(v):
            try:
                return pm.GetProxyName("views", v)
            except Exception:
                return None

        _log(f"[DEBUG] {tag} Views ({len(views)}):", "DEBUG")
        for name, v in views.items():
            try:
                alive = bool(v.GetClientSideObject())
            except Exception:
                alive = False
            _log(f"  - {name}: alive={alive} id={v.GetGlobalIDAsString()}", "DEBUG")

        _log(f"[DEBUG] {tag} Extractors ({len(exts)}):", "DEBUG")
        for name, ex in exts.items():
            try:
                ex_xml = ex.GetXMLName()
                # extractor-level View
                ex_vp = ex.GetProperty("View")
                ex_view = ex_vp.GetProxy(0) if ex_vp and ex_vp.GetNumberOfProxies() > 0 else None
                ex_vname = _name_for_view(ex_view) if ex_view else None

                # writer-level View
                writer = None
                writer_vname = None
                w = ex.GetProperty("Writer")
                writer = w.GetProxy(0) if w and w.GetNumberOfProxies() > 0 else None
                if writer is not None:
                    wv = writer.GetProperty("View")
                    wview = wv.GetProxy(0) if wv and wv.GetNumberOfProxies() > 0 else None
                    writer_vname = _name_for_view(wview) if wview else None

                _log(f"  - {name}: xml={ex_xml} ex.View={ex_vname} writer.View={writer_vname} status_ex={'OK' if ex_view else 'MISSING'} status_w={'OK' if writer_vname else 'MISSING'}", "DEBUG")
            except Exception as e:
                _log(f"  - {name}: error inspecting: {e}", "DEBUG")
    except Exception as e:
        _log(f"[ERROR] _debug_dump_state failed: {e}", "ERROR")


def _fix_png_extractors_force(only_on_change: bool = True, log_level: str = None):
    """Forcibly reestablish PNG extractor-view connections each cycle.

    Parameters:
    - only_on_change: when True (default), log only when a binding was missing or was changed;
      suppress routine DEBUG chatter.
    - log_level: optional override for this call (e.g., 'DEBUG'|'INFO'|'WARN'|'ERROR'|'NONE').
      If provided, the global log level will be temporarily set during this call.
    """
    prev_level = None
    if log_level is not None:
        prev_level = get_log_level()
        set_log_level(log_level)
    _log("[DEBUG] _fix_png_extractors_force() called", "DEBUG")
    try:
        pm = servermanager.ProxyManager()

        # Get available views - use the first available view as fallback
        views_map = pm.GetProxiesInGroup("views")
        if not views_map:
            _log("[ERROR] No views available for PNG extractors", "ERROR")
            return

        fallback_view = list(views_map.values())[0]
        fallback_name = list(views_map.keys())[0]
        _log(f"[DEBUG] Using fallback view: {fallback_name}", "DEBUG")

        # Helper to resolve intended view via registry, else fallback
        def _intended_view_for(ex_proxy):
            # by proxy
            if ex_proxy in _PNG_REGISTRY_BY_PROXY:
                return _PNG_REGISTRY_BY_PROXY[ex_proxy]
            # by name
            try:
                ex_name = servermanager.ProxyManager().GetProxyName('extractors', ex_proxy)
                if isinstance(ex_name, tuple):
                    ex_name = ex_name[0]
                if ex_name in _PNG_REGISTRY_BY_NAME:
                    return _PNG_REGISTRY_BY_NAME[ex_name]
            except Exception:
                pass
            return fallback_view
            
        # Fix each PNG extractor
        extractors = pm.GetProxiesInGroup("extractors")
        for name_tuple, ex in extractors.items():
            # name_tuple is like ('PNG1', '4066') - extract the actual name
            actual_name = name_tuple[0] if isinstance(name_tuple, tuple) else str(name_tuple)
            if actual_name.startswith('PNG'):
                try:
                    # Check if this is a PNG extractor by looking at its writer
                    writer_prop = ex.GetProperty("Writer")
                    if writer_prop and writer_prop.GetNumberOfProxies() > 0:
                        writer = writer_prop.GetProxy(0)
                        writer_xml = writer.GetXMLName() if hasattr(writer, 'GetXMLName') else 'Unknown'
                        
                        if 'PNG' in writer_xml or 'Image' in writer_xml:
                            target_view = _intended_view_for(ex)
                            # Resolve a pretty view name for logs
                            try:
                                view_name_for_log = servermanager.ProxyManager().GetProxyName('views', target_view)
                            except Exception:
                                view_name_for_log = 'unknown'
                            # init previous state trackers for change detection
                            prev_ok_e = False
                            prev_ex_proxy = None
                            prev_ok_w = False
                            prev_w_proxy = None
                            # Set extractor-level View using vtkSMProxy
                            try:
                                ex_view_prop = ex.GetProperty("View")
                                if ex_view_prop and hasattr(target_view, 'SMProxy'):
                                    # detect prior assignment for change logging
                                    prev_ok_e = bool(ex_view_prop and ex_view_prop.GetNumberOfProxies() > 0 and ex_view_prop.GetProxy(0))
                                    prev_ex_proxy = ex_view_prop.GetProxy(0) if prev_ok_e else None
                                    ex_view_prop.SetProxy(0, target_view.SMProxy)
                                    ex.UpdateVTKObjects()
                            except Exception as prop_ex:
                                _log(f"[DEBUG] Extractor view property setting failed: {prop_ex}", "DEBUG")

                            # Set writer-level View using vtkSMProxy
                            try:
                                writer_view_prop = writer.GetProperty("View")
                                if writer_view_prop and hasattr(target_view, 'SMProxy'):
                                    prev_ok_w = bool(writer_view_prop and writer_view_prop.GetNumberOfProxies() > 0 and writer_view_prop.GetProxy(0))
                                    prev_w_proxy = writer_view_prop.GetProxy(0) if prev_ok_w else None
                                    writer_view_prop.SetProxy(0, target_view.SMProxy)
                                    writer.UpdateVTKObjects()
                            except Exception as writer_ex:
                                _log(f"[DEBUG] Writer view property setting failed: {writer_ex}", "DEBUG")

                            # Verify assignment
                            try:
                                wv = writer.GetProperty("View")
                                ok_w = (wv and wv.GetNumberOfProxies() > 0 and wv.GetProxy(0) is not None)
                                ev = ex.GetProperty("View")
                                ok_e = (ev and ev.GetNumberOfProxies() > 0 and ev.GetProxy(0) is not None)
                                changed = (not prev_ok_w or not prev_ok_e or (prev_w_proxy is not None and wv.GetProxy(0) != prev_w_proxy) or (prev_ex_proxy is not None and ev.GetProxy(0) != prev_ex_proxy))
                                if only_on_change:
                                    if changed or not ok_w or not ok_e:
                                        _log(f"[FIX] Rebound {actual_name} -> {view_name_for_log} | writer_ok={ok_w} extractor_ok={ok_e}", "INFO")
                                else:
                                    _log(f"[FIX] Rebound {actual_name} -> {view_name_for_log} | writer_ok={ok_w} extractor_ok={ok_e}", "INFO")
                            except Exception:
                                pass
                        else:
                            _log(f"[DEBUG] Skipping {actual_name} - not PNG extractor (writer: {writer_xml})", "DEBUG")
                        
                except Exception as e:
                    _log(f"[ERROR] Failed to fix {actual_name}: {e}", "ERROR")
                        
    except Exception as e:
        _log(f"[ERROR] _fix_png_extractors_force failed: {e}", "ERROR")
    finally:
        if log_level is not None and prev_level is not None:
            set_log_level(prev_level)
