"""! \file png_ext_sfield.py
\brief Catalyst PNG extractor for 3D scalar fields.
\details Performs volume rendering with adaptive camera, ghost cutting (Threshold), and smoothing (C2P).
"""

# script-version: 2.2
# Catalyst state generated using paraview version 5.13.3

import paraview
from paraview.simple import *
# We use 'import *' so we don't need to guess which specific names to import.

from paraview import print_info
import argparse
from paraview import servermanager as sm
from vtkmodules.vtkParallelCore import vtkCommunicator, vtkMultiProcessController
import math
from paraview.simple import (
    ResampleToImage,
    SetActiveView,
    PVTrivialProducer,
    Threshold,
    Hide
)
# ----------------------------------------------------------------
# helpers used for adaptive visualization
# ----------------------------------------------------------------
from catalystSubroutines import (
    nice_bounds_sym,
    auto_camera_from_bounds,
    get_global_spatial_bounds,
    get_global_range,
    hide_source_from_gui,
    hide_extractor_from_gui,
    # get_global_extent
)

def print_info_(s, level=0):
    global verbosity
    if verbosity > level:
        print_info(s)

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------
paraview.simple._DisableFirstRenderCameraReset()
SetActiveView(None)

arg_list = paraview.catalyst.get_args()
parser = argparse.ArgumentParser()
parser.add_argument("--label", default="AAAAAA", help="Label.")
parser.add_argument("--channel_name", default="DEFAULT_CHANNEL", help="Channel.")
parser.add_argument("--experiment_name", default="_", help="Exp Name.")
parser.add_argument("--verbosity", type=int, default="1", help="Verbosity")
parsed = parser.parse_args(arg_list)

label = parsed.label
exp_chann = parsed.channel_name
exp_string = parsed.experiment_name
verbosity = parsed.verbosity
print_info_("_global__scope__()::" + parsed.channel_name)



cname = parsed.channel_name
ippl_scalar_p = PVTrivialProducer(registrationName=cname)
associate = "CELLS"
scalar_info = ippl_scalar_p.GetDataInformation()


local_bounds = scalar_info.GetBounds()
print_info(local_bounds)
local_extent = scalar_info.GetExtent()
print_info(local_extent)
global_bounds = get_global_spatial_bounds(local_bounds)
print_info(global_bounds)

# # Check for ghosts
ippl_scalar_p.UpdatePipeline()
cinfo = scalar_info.GetCellDataInformation()
ghost_info = cinfo.GetArrayInformation("vtkGhostType") if cinfo else None


# # ----------------------------------------------------------------
# # 1. Reader
# # ----------------------------------------------------------------
"""
Derive uniform spacing from local bounds and local extent.
For ImageData, bounds length ≈ (num_points - 1) * spacing.
Assume uniform spacing across ranks; compute it from local segment.
"""
nx = (local_extent[1] - local_extent[0] + 1)
ny = (local_extent[3] - local_extent[2] + 1)
nz = (local_extent[5] - local_extent[4] + 1)

lx = (local_bounds[1] - local_bounds[0])
ly = (local_bounds[3] - local_bounds[2])
lz = (local_bounds[5] - local_bounds[4])

# Avoid division by zero for degenerate dims; fall back to 1.0
spacing_x = lx / max(nx - 1, 1)
spacing_y = ly / max(ny - 1, 1)
spacing_z = lz / max(nz - 1, 1)

gx = global_bounds[1] - global_bounds[0]  # X range
gy = global_bounds[3] - global_bounds[2]  # Y range
gz = global_bounds[5] - global_bounds[4]  # Z range
dx = max(spacing_x, 1e-12)
dy = max(spacing_y, 1e-12)
dz = max(spacing_z, 1e-12)

# Keep a copy of the original global bounds for camera framing to avoid visual "zoom"
original_global_bounds = tuple(global_bounds)

# Optionally trim one layer if you want to exclude outer ghost ring from resampling
if ghost_info:
    cut_layers = 1  # set to 0 to keep full bounds
else:
    cut_layers = 0  # set to 0 to keep full bounds

# positive will distrub border rendering with bad extrapolation ....
extra_dims = 0
global_bounds = (
    global_bounds[0] + cut_layers * dx,
    global_bounds[1] - cut_layers * dx,
    global_bounds[2] + cut_layers * dy,
    global_bounds[3] - cut_layers * dy,
    global_bounds[4] + cut_layers * dz,
    global_bounds[5] - cut_layers * dz,
)
# Points count = round(range / spacing) + 1 to preserve voxel size exactly
dim_x = int(round((global_bounds[1] - global_bounds[0]) / dx)) + extra_dims
dim_y = int(round((global_bounds[3] - global_bounds[2]) / dy)) + extra_dims
dim_z = int(round((global_bounds[5] - global_bounds[4]) / dz)) + extra_dims
global_extent = [dim_x, dim_y, dim_z]

print_info_(f"Local spacing: dx={spacing_x}, dy={spacing_y}, dz={spacing_z}")
print_info_(f"Global bounds (after modification)): {global_bounds}")
print_info_(f"Global extent (after modification)): {global_extent}")

# ----------------------------------------------------------------
# 2. Ghost Handling & Restore Grid (Threshold -> Resample)
# ----------------------------------------------------------------
if ghost_info:
    # A. Use Threshold to physically remove ghost cells
    # This converts ImageData -> UnstructuredGrid
    print_info_("vtkGhostType found; processing ghosts...", level=1)
    ippl_scalar_t = Threshold(registrationName='RemoveGhosts', Input=ippl_scalar_p)
    hide_source_from_gui(ippl_scalar_t)
    ippl_scalar_t.Scalars = ['CELLS', 'vtkGhostType']
    ippl_scalar_t.AllScalars = 0 
    ippl_scalar_t.ThresholdMethod = 'Between'
    ippl_scalar_t.LowerThreshold = 0.0
    ippl_scalar_t.UpperThreshold = 0.0

    # B. Convert BACK to Uniform Grid using ResampleToImage
    # This restores the high-quality GPU Volume Mapper.
    print_info_("Resampling back to ImageData...", level=1)
    ippl_resampled = ResampleToImage(registrationName='ResampleToGrid', Input=ippl_scalar_t)
    hide_source_from_gui(ippl_resampled)
    ippl_resampled.UseInputBounds = 1
    ippl_resampled.SamplingDimensions = global_extent
    """ does same thing for cut_layers=1 """
    # ippl_resampled = ResampleToImage(registrationName='ResampleToGrid', Input=ippl_scalar_t)
    # ippl_resampled.UseInputBounds = 0
    # ippl_resampled.SamplingBounds = global_bounds
    # ippl_resampled.SamplingDimensions = global_extent
    ippl_scalar = ippl_resampled
    associate = "POINTS" # ResampleToImage produces Point Data by default

    # ALTERNATIVES  

    # No modifaction: 
    # ippl_scalar = ippl_scalar_p
    # associate = "CELLS"
    # cuts in the middleinner ghost
    
    # CellDataToPointData
    # For Rendering  ResampleToImage Data works better works better than 
    
    # ippl_scalar = ippl_scalar_t
    # associate = "CELLS"
    ## good block strucuture rendering but not pretty., probaby good for machine learning

    #  Calculator filter ...
    # A0 calculator filter didn't manage to make them work ... -> would be ideal to only cut internal ghost, 
    # and not the rely on the proper resample cut ...
else:
    print_info_("vtkGhostType not found; skipping ghost filtering", level=1)

    print_info_("Resampling back to ImageData...", level=1)
    ippl_resampled = ResampleToImage(registrationName='ResampleToGrid', Input=ippl_scalar_p)
    hide_source_from_gui(ippl_resampled)
    ippl_resampled.UseInputBounds = 0
    ippl_resampled.SamplingBounds = global_bounds
    ippl_resampled.SamplingDimensions = global_extent

    ippl_scalar = ippl_scalar_p
    associate = "CELLS"
    ippl_scalar = ippl_resampled
    associate = "POINTS" # ResampleToImage produces Point Data by default

# ----------------------------------------------------------------
# Setup View
# ----------------------------------------------------------------
view_name = f"View_{cname}"
# renderView1 = CreateView('RenderView', registrationName=view_name)
renderView1 = CreateView('RenderView')
materialLibrary1 = GetMaterialLibrary()
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1
renderView1.ViewSize = [2000, 1500]
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.AxesGrid.Visibility = 1
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Gradient'
SetActiveView(renderView1)

# ----------------------------------------------------------------
# Initial Camera
# ----------------------------------------------------------------
ippl_scalar.UpdatePipeline()
# Use original (pre-trim) bounds to keep camera framing stable and avoid apparent zoom.
auto_camera_from_bounds(renderView1, original_global_bounds)
# ----------------------------------------------------------------
# Show Data
# ----------------------------------------------------------------
ippl_scalarDisplay = Show(ippl_scalar, renderView1)
# ----------------------------------------------------------------
# Transfer Functions
# ----------------------------------------------------------------
densityLUT = GetColorTransferFunction(label)
densityLUT.RGBPoints = [-2.00, 0.231373, 0.298039, 0.752941, 
                         0.00, 0.865003, 0.865003, 0.865003, 
                         2.00, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

densityPWF = GetOpacityTransferFunction(label)

densityPWF.Points = [-2.00, 1.00, 0.5, 0.0, 
                      0.00, 0.00, 0.5, 0.0, 
                      2.00, 1.00, 0.5, 0.0]

densityPWF.Points = [-2.00, 1.00, 0.5, 0.0, 
                     -1.20, 0.75, 0.5, 0.0, 
                     -0.80, 0.25, 0.5, 0.0, 
#     # --- The Zero Wall ---
                     -0.01, 0.00, 0.5, 0.0, 
                      0.00, 1.00, 0.5, 0.0, 
                      0.01, 0.00, 0.5, 0.0, 
#     # ---------------------
                      0.80, 0.25, 0.5, 0.0, 
                      1.20, 0.75, 0.5, 0.0, 
                      2.00, 1.00, 0.5, 0.0]

densityPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# Configure Display 
# ----------------------------------------------------------------
ippl_scalarDisplay.Representation = 'Volume'
ippl_scalarDisplay.LookupTable = densityLUT
ippl_scalarDisplay.ScaleTransferFunction = 'Piecewise Function'
ippl_scalarDisplay.ScaleFactor = 2.0
ippl_scalarDisplay.GaussianRadius = 0.1

# Dynamic Association (Points or Cells depending on what we found above)
ippl_scalarDisplay.ColorArrayName = [associate, label]

# Explicit Opacity Mapping 
ippl_scalarDisplay.OpacityArrayName = [associate, label]
ippl_scalarDisplay.OpacityTransferFunction = 'Piecewise Function'
ippl_scalarDisplay.ScalarOpacityUnitDistance = 4.00
ippl_scalarDisplay.ScalarOpacityFunction = densityPWF

# Scalar Bar
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.Title = label
densityLUTColorBar.ComponentTitle = 'Magnitude'
densityLUTColorBar.Visibility = 1
ippl_scalarDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# Extractors
# ----------------------------------------------------------------
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG_'+cname)
# hide_extractor_from_gui(pNG1)
pNG1.Trigger = 'Time Step'
pNG1.Trigger.Frequency = 1
pNG1.Writer.FileName = label+'_ScalarField_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [2000, 1500]
pNG1.Writer.Format = 'PNG'
# SetActiveSource(pNG1)

# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.EnableCatalystLive = 0
options.CatalystLiveTrigger = 'Time Step'
options.ExtractsOutputDirectory = 'data_png_extracts_' + exp_string 

if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    SaveExtractsUsingCatalystOptions(options)

# ------------------------------------------------------------------------------
def catalyst_execute(info):
    print_info_("catalyst_execute()::"+parsed.channel_name)
    
    
    global ghost_info
    # if ghost_info:
    global ippl_scalar_p
    global ippl_scalar_t
    global ippl_resampled
    global ippl_scalar
    global densityLUT
    global densityPWF
    global associate
    global renderView1

    ippl_scalar_p.UpdatePipeline()
    # ippl_scalar_t
    # ippl_resampled
    # ippl_scalar


    # SetActiveView(renderView1)
    # renderView1.UddatePipeline()



    if info.cycle % 10 == 0:
        ippl_scalar.UpdatePipeline()
        scalar_info = ippl_scalar.GetDataInformation()
        local_bounds = scalar_info.GetBounds()
        
        # === ADAPTIVE CHECK BASED ON ASSOCIATION ===
        data_array_info = None
        if associate == "POINTS":
            point_data_info = scalar_info.GetPointDataInformation()
            data_array_info = point_data_info.GetArrayInformation(label)
        else:
            cell_data_info = scalar_info.GetCellDataInformation()
            data_array_info = cell_data_info.GetArrayInformation(label)

        if data_array_info:
            local_vmin, local_vmax = data_array_info.GetComponentRange(-1)
        else:
            print_info_(f"Warning: Array '{label}' not found in {associate}.", level=1)
            local_vmin, local_vmax = 0.0, 1.0
        # ============================================

        global_bounds = get_global_spatial_bounds(local_bounds)
        # Keep camera framing using the original domain size to avoid visual zoom on trims/resampling
        global_bounds_for_camera = original_global_bounds if 'original_global_bounds' in globals() else global_bounds
        global_vmin, global_vmax = get_global_range(local_vmin, local_vmax)
        nice_min, nice_max = nice_bounds_sym(global_vmin, global_vmax)
        auto_camera_from_bounds(renderView1, global_bounds_for_camera)
        densityLUT.RescaleTransferFunction(nice_min, nice_max)
        densityPWF.RescaleTransferFunction(nice_min, nice_max)









# # ----------------------------------------------------------------
# # B. Smart Ghost Handling & Resampling
# #    (Stitch internal seams, keep outer borders, restore Grid)
# # ----------------------------------------------------------------

# # 1. Calculate Global Bounds & Spacing
# # ------------------------------------
# ippl_scalar_p.UpdatePipeline()
# data_info = ippl_scalar_p.GetDataInformation()
# local_bounds = data_info.GetBounds()
# local_extent = data_info.GetExtent()

# # Global Bounds (The size of the Red Box)
# global_bounds = get_global_spatial_bounds(local_bounds)

# # Calculate Spacing from local data (Robust)
# nx = (local_extent[1] - local_extent[0] + 1)
# ny = (local_extent[3] - local_extent[2] + 1)
# nz = (local_extent[5] - local_extent[4] + 1)

# lx = (local_bounds[1] - local_bounds[0])
# ly = (local_bounds[3] - local_bounds[2])
# lz = (local_bounds[5] - local_bounds[4])

# spacing_x = lx / max(nx - 1, 1)
# spacing_y = ly / max(ny - 1, 1)
# spacing_z = lz / max(nz - 1, 1)

# # 2. Calculate Exact Output Dimensions (Points)
# # ---------------------------------------------
# # ResampleToImage needs Dimensions = (Length / Spacing) + 1
# gx = global_bounds[1] - global_bounds[0]
# gy = global_bounds[3] - global_bounds[2]
# gz = global_bounds[5] - global_bounds[4]

# dim_x = int(round(gx / spacing_x)) + 1
# dim_y = int(round(gy / spacing_y)) + 1
# dim_z = int(round(gz / spacing_z)) + 1
# global_dims = [dim_x, dim_y, dim_z]

# print_info_(f"Target Grid: Bounds={global_bounds}, Dims={global_dims}")

# if ghost_info:
#     print_info_("Processing Ghosts (Smart Stitching)...", level=1)

#     # 3. Un-flag Global Border Ghosts
#     # -------------------------------
#     # We create a mask: If a cell is at the global edge, treat it as VALID (0).
#     # Only if it is strictly inside the domain do we treat it as a GHOST.
    
#     # Epsilon tolerance for floating point comparisons
#     eps_x = spacing_x * 0.1
#     eps_y = spacing_y * 0.1
#     eps_z = spacing_z * 0.1

#     # Logic: "Is this cell strictly inside the red box?"
#     # coordsX refers to the cell center.
#     is_internal_x = f"(coordsX > {global_bounds[0] + eps_x}) & (coordsX < {global_bounds[1] - eps_x})"
#     is_internal_y = f"(coordsY > {global_bounds[2] + eps_y}) & (coordsY < {global_bounds[3] - eps_y})"
#     is_internal_z = f"(coordsZ > {global_bounds[4] + eps_z}) & (coordsZ < {global_bounds[5] - eps_z})"
    
#     is_strictly_internal = f"({is_internal_x} & {is_internal_y} & {is_internal_z})"

#     # Calculator: IF(StrictlyInternal, keep GhostType, else 0)
#     # This keeps internal overlaps (to be cut) but saves the outer border.
#     smart_ghosts = Calculator(registrationName='SmartGhostFix', Input=ippl_scalar_p)
#     smart_ghosts.AttributeType = 'Cell Data'
#     smart_ghosts.ResultArrayName = 'vtkGhostType' 
#     smart_ghosts.Function = f"if({is_strictly_internal}, vtkGhostType, 0)"

#     # 4. Threshold (The Cut)
#     # ----------------------
#     # Now this only removes the internal overlaps.
#     ippl_scalar_t = Threshold(registrationName='RemoveInternalGhosts', Input=smart_ghosts)
#     ippl_scalar_t.Scalars = ['CELLS', 'vtkGhostType']
#     ippl_scalar_t.AllScalars = 0 
#     ippl_scalar_t.ThresholdMethod = 'Between'
#     ippl_scalar_t.LowerThreshold = 0.0
#     ippl_scalar_t.UpperThreshold = 0.0

#     # 5. Resample (Restore the Grid)
#     # ------------------------------
#     # We map the "stitched" unstructured data back onto the perfect global grid.
#     print_info_("Resampling back to ImageData...", level=1)
    
#     ippl_resampled = ResampleToImage(registrationName='ResampleToGrid', Input=ippl_scalar_t)
#     ippl_resampled.UseInputBounds = 0
#     ippl_resampled.SamplingBounds = global_bounds
#     ippl_resampled.SamplingDimensions = global_dims
    
#     ippl_scalar = ippl_resampled
#     associate = "POINTS" 

# else:
#     print_info_("No ghosts found. Using original data.", level=1)
#     ippl_scalar = ippl_scalar_p
#     associate = "CELLS"

