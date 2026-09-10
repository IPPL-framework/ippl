# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12

import argparse
from paraview import catalyst, print_info

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

parser = argparse.ArgumentParser()
parser.add_argument("--verbosity", type=int, default=0, help="Catalyst output level")
parsed, _ = parser.parse_known_args(catalyst.get_args())

if parsed.verbosity >= 4:
    print_info("====================================>")
    print_info("===EXECUTING EMPTY EXTRACTOR======>")
    print_info("====================================>")
