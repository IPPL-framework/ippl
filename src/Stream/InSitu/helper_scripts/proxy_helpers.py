

from paraview import servermanager

def print_proxy_overview():
    
    print("====Printing Proxy Overview  ===========")
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
    print("===Printing Proxy Overview==============DONE")
# cr.
# print_proxy_overview()
# 
