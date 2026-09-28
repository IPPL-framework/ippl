#!/usr/bin/env python3
## @file verify_mesh_output.py
# @brief Standard-library integration checks for exported mesh-1 diagnostics.
# @ingroup chdr_tests
"""Validate mesh-1 diagnostics and compare material masks across MPI runs.

Uses only the Python standard library. The shipped compact prism has an
independent Cartesian reference, x in [-min(z, 50-z), 0], y in [-20, 20],
z in [0, 50] mm. Its slices and quick-grid voxel count are checked against
that reference rather than against the C++ prism implementation. Bricks
are checked against their Cartesian intervals, including an independent
cell-centre count on every grid. The self-contained material viewer must
embed the same metadata and slice CSV as the separately exported files.

From the repository root, after equivalent serial and two-rank mesh runs:
    python demos/chdr/src/verify_mesh_output.py PATH_TO_SERIAL PATH_TO_MPI2

This integration checker reads existing output; it does not launch MPI or
exercise the geometry classes directly. It compares sampled slice values,
global occupied-cell counts and the exported checksum, not every 3D cell.
A checksum match is evidence of consistency, not a collision-free proof.
Other prism shapes receive structural checks but no independent shape oracle.
No electromagnetic evolution, PDF rendering or interactive display is tested.
Validation errors return status 1; success returns 0. CLI usage errors use
argparse's status 2. Library helpers raise exceptions instead of exiting.
"""

import argparse
import ast
import csv
import itertools
import json
import math
from pathlib import Path
import sys


## Slice name to (first plotted axis, second plotted axis, normal axis).
# Axis indices 0, 1 and 2 mean physical x, y and z respectively.
PLANES = {"xz": (0, 2, 1), "xy": (0, 1, 2), "yz": (1, 2, 0)}


## @brief Raise ValueError with the supplied diagnostic when a check fails.
def require(condition, message):
    if not condition:
        raise ValueError(message)


## @brief Compare serialized numbers using fixed roundoff tolerances.
# These tolerances check diagnostic consistency, not mesh convergence or physics.
def close(first, second):
    return math.isclose(first, second, rel_tol=1.0e-12, abs_tol=1.0e-10)


## @brief Return a finite numeric value, rejecting booleans and non-numbers.
# @param value Candidate JSON number.
# @param name Field name included in validation errors.
# @throws ValueError If the value is not a finite integer or float.
def number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{name}: expected a number")
    require(math.isfinite(value), f"{name}: expected a finite number")
    return value


## @brief Validate a numeric integer at least minimum and return a Python int.
# @param value Candidate numeric integer.
# @param name Field name included in validation errors.
# @param minimum Inclusive lower bound for the integer.
# @throws ValueError If the value is nonfinite, fractional or below minimum.
def integer(value, name, minimum=0):
    number(value, name)
    require(value == int(value) and value >= minimum,
            f"{name}: expected an integer >= {minimum}")
    return int(value)


## @brief Validate and return a three-element list of finite numbers.
# @param value Candidate three-element list.
# @param positive Require every component to be strictly positive when true.
# @param name Field name included in validation errors.
# @throws ValueError If shape, numeric type or a requested sign check fails.
def triple(value, name, positive=False):
    require(isinstance(value, list) and len(value) == 3,
            f"{name}: expected a three-element array")
    result = [number(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if positive:
        require(all(item > 0 for item in result), f"{name}: entries must be positive")
    return result


## @brief Recursively compare JSON-like metadata from two equivalent mesh runs.
# @param first Reference scalar, list or mapping.
# @param second Corresponding value from the run being compared.
# @param name Metadata path used to identify a mismatch.
# Numeric values use close(); array lengths, mapping keys and other values must
# match. MPI-specific rank boxes are excluded by the caller, not by this helper.
# @throws ValueError If the inputs differ beyond serialization tolerances.
def compareValues(first, second, name):
    """Compare metadata recursively, permitting only serialization roundoff."""
    if isinstance(first, dict):
        require(isinstance(second, dict) and first.keys() == second.keys(),
                f"{name}: keys differ")
        for key in first:
            compareValues(first[key], second[key], f"{name}.{key}")
    elif isinstance(first, list):
        require(isinstance(second, list) and len(first) == len(second),
                f"{name}: array lengths differ")
        for index, (left, right) in enumerate(zip(first, second)):
            compareValues(left, right, f"{name}[{index}]")
    elif isinstance(first, (int, float)) and not isinstance(first, bool):
        require(isinstance(second, (int, float)) and close(first, second),
                f"{name}: {first!r} != {second!r}")
    else:
        require(first == second, f"{name}: {first!r} != {second!r}")


## @brief Recognize the shipped 25 x 40 x 50 mm compact-prism reference geometry.
# Accepts any vertex ordering for the same triangle and the positive y extrusion.
# A false result disables the independent prism oracle; it does not declare that
# the supplied geometry is invalid.
def isCompactPrism(prism):
    expected = sorted([[0.0, -20.0, 0.0], [-25.0, -20.0, 25.0], [0.0, -20.0, 50.0]])
    actual = sorted(prism["vertices"])
    return (all(close(a, b) for row, reference in zip(actual, expected)
                for a, b in zip(row, reference))
            and all(close(a, b) for a, b in zip(prism["axis"], [0.0, 1.0, 0.0]))
            and close(prism["height"], 40.0))


## @brief Classify a point against the compact prism's explicit half-plane oracle.
# @param x,y,z Physical point coordinates in millimetres.
# The reference is x in [-min(z, 50-z), 0], y in [-20, 20], z in [0, 50].
# This independent formula includes faces within the fixed comparison tolerance;
# it does not reuse the C++ prism implementation.
def insideCompact(x, y, z):
    # This reference uses the explicit half-plane equations, not barycentric
    # coordinates or a copy of PrismGeometry::contains. Lengths are in mm.
    tolerance = 1.0e-10
    return (-20.0 - tolerance <= y <= 20.0 + tolerance
            and -tolerance <= z <= 50.0 + tolerance
            and -min(z, 50.0 - z) - tolerance <= x <= tolerance)


## @brief Classify a point against three independent closed brick intervals.
# @param point Three physical coordinates in millimetres.
# @param lower Brick lower corner in millimetres.
# @param size Positive brick dimensions in millimetres, already validated.
# @return True when all coordinates lie within the interval comparison tolerance.
def insideBrick(point, lower, size):
    """Independent axis-aligned box reference in millimetres."""
    tolerance = 1.0e-10
    return all(origin - tolerance <= coordinate <= origin + extent + tolerance
               for coordinate, origin, extent in zip(point, lower, size))


## @brief Check that a generated standalone viewer embeds its source diagnostics.
# @param directory Output directory containing mesh-1_Materials.py.
# @param metadata Parsed mesh.json expected in the embedded JSON string.
# @param sliceText Exact slices.csv text expected in the embedded CSV string.
# AST parsing checks Python syntax and literal assignments without importing the
# viewer or any plotting library. It does not validate rendering or GUI behavior.
# @throws ValueError If embedded data or their assignment structure differ.
# @throws SyntaxError If the generated Python source cannot be parsed.
def checkEmbeddedDiagnostics(directory, metadata, sliceText):
    """Inspect literal data without importing or executing the generated viewer."""
    script = directory / "mesh-1_Materials.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    assignments = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ("MESH_METADATA", "SLICE_CSV"):
                require(target.id not in assignments,
                        f"viewer defines {target.id} more than once")
                assignments[target.id] = node.value
    require(assignments.keys() == {"MESH_METADATA", "SLICE_CSV"},
            "viewer must embed MESH_METADATA and SLICE_CSV")
    metadataNode = assignments["MESH_METADATA"]
    require(isinstance(metadataNode, ast.Call)
            and isinstance(metadataNode.func, ast.Attribute)
            and isinstance(metadataNode.func.value, ast.Name)
            and metadataNode.func.value.id == "json" and metadataNode.func.attr == "loads"
            and len(metadataNode.args) == 1 and not metadataNode.keywords,
            "viewer MESH_METADATA must be json.loads of a literal string")
    embeddedMetadataText = ast.literal_eval(metadataNode.args[0])
    require(isinstance(embeddedMetadataText, str), "viewer metadata must be a literal string")
    require(json.loads(embeddedMetadataText) == metadata,
            "viewer embedded metadata differs from mesh.json")
    require(ast.literal_eval(assignments["SLICE_CSV"]) == sliceText,
            "viewer embedded slices differ from slices.csv")


## @brief Recover a global owned-cell index from an exported centre coordinate.
# @param coordinate Physical coordinate in millimetres.
# @param axis Coordinate direction: 0=x, 1=y or 2=z.
# @param domain Metadata containing lower corner, spacing and global cell counts.
# @param name Description included in a failed centre/range check.
# @return Zero-based global index; ghost-cell coordinates are rejected.
# @throws ValueError If the coordinate is not a valid owned global cell centre.
def cellIndex(coordinate, axis, domain, name):
    position = (coordinate - domain["lower"][axis]) / domain["spacing"][axis] - 0.5
    index = round(position)
    require(abs(position - index) <= 1.0e-8 and 0 <= index < domain["cells"][axis],
            f"{name}: {coordinate} mm is not an owned global cell centre")
    return index


## @brief Validate one mesh-1 output directory and return metadata plus samples.
# @ingroup chdr_tests
# @param directory pathlib.Path containing mesh.json, slices.csv and the viewer.
# @return Pair (metadata, rows), with rows keyed by (plane, global_u, global_v).
#
# Checks schema, dimensions, rank-box volume sum, reported halo status, preview
# completeness and embedded diagnostics. Compact-prism slice values use the
# independent oracle; its occupied-cell count is checked on the 40 x 40 x 64 grid.
# Bricks use independent interval checks and a count for every grid size. Other
# prisms receive no independent volume, shape or occupied-cell-count calculation.
# Rank-box volume sums do not alone prove non-overlap. Reported halo diagnostics
# are checked, but this script cannot re-read distributed halo values.
# @throws ValueError If a consistency or reference check fails.
# @throws OSError If a required output file cannot be read.
def readAndValidate(directory):
    with (directory / "mesh.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    require(metadata["schema_version"] == 2 and metadata["units"] == "mm",
            "expected schema_version=2 and units='mm'")
    domain = metadata["domain"]
    lower = triple(domain["lower"], "domain.lower")
    size = triple(domain["size"], "domain.size", positive=True)
    cells = [integer(value, "domain.cells", minimum=1)
             for value in triple(domain["cells"], "domain.cells")]
    spacing = triple(domain["spacing"], "domain.spacing", positive=True)
    for axis in range(3):
        require(close(spacing[axis], size[axis] / cells[axis]),
                f"domain.spacing[{axis}] disagrees with size/cells")

    radiator = metadata["radiator"]
    require(radiator["type"] in ("prism", "brick"), "unsupported radiator type")
    for key in ("epsilon_r", "analytic_volume_mm3"):
        require(number(radiator[key], f"radiator.{key}") > 0,
                f"radiator.{key} must be positive")
    compact = False
    insideReference = None
    if radiator["type"] == "prism":
        require(len(radiator["vertices"]) == 3, "prism.vertices must contain three vertices")
        for index, vertex in enumerate(radiator["vertices"]):
            triple(vertex, f"prism.vertices[{index}]")
        axis = triple(radiator["axis"], "prism.axis")
        require(close(sum(value * value for value in axis), 1.0), "prism.axis must be a unit vector")
        require(number(radiator["height"], "prism.height") > 0, "prism.height must be positive")
        compact = isCompactPrism(radiator)
        if compact:
            require(close(radiator["analytic_volume_mm3"], 25000.0),
                    "compact prism analytic volume must be 25000 mm^3")
            insideReference = lambda point: insideCompact(*point)
    else:
        brickLower = triple(radiator["lower"], "brick.lower")
        brickSize = triple(radiator["size"], "brick.size", positive=True)
        require(close(radiator["analytic_volume_mm3"], math.prod(brickSize)),
                "brick analytic volume must equal the product of its dimensions")
        insideReference = lambda point: insideBrick(point, brickLower, brickSize)
    background = number(metadata["background_epsilon_r"], "background_epsilon_r")
    require(background > 0, "background_epsilon_r must be positive")

    diagnostics = metadata["diagnostics"]
    total = integer(diagnostics["total_cells"], "diagnostics.total_cells", minimum=1)
    occupied = integer(diagnostics["dielectric_cells"], "diagnostics.dielectric_cells")
    require(total == math.prod(cells) and occupied <= total,
            "total/dielectric cell counts are inconsistent with domain")
    require(close(number(diagnostics["voxel_volume_mm3"], "voxel_volume_mm3"),
                  occupied * math.prod(spacing)), "voxel volume disagrees with dielectric cell count")
    checksumText = str(diagnostics["material_index_checksum"])
    require(checksumText.isdecimal() and 0 <= int(checksumText) < 2**64,
            "material_index_checksum must be an unsigned 64-bit decimal integer")
    rankCount = integer(diagnostics["rank_count"], "rank_count", minimum=1)
    require(integer(diagnostics["halo_mismatches"], "halo_mismatches") == 0,
            "material halo mismatches were reported")
    ranks = metadata["ranks"]
    require(len(ranks) == rankCount, "rank_count disagrees with rank-box count")
    require(sorted(integer(rank["rank"], "rank index") for rank in ranks) == list(range(rankCount)),
            "rank indices must be unique and contiguous")
    rankVolume = 0.0
    for rank in ranks:
        rankLower = triple(rank["lower"], "rank.lower")
        rankSize = triple(rank["size"], "rank.size", positive=True)
        for axis in range(3):
            require(rankLower[axis] >= lower[axis] - 1.0e-10
                    and rankLower[axis] + rankSize[axis] <= lower[axis] + size[axis] + 1.0e-10,
                    "rank box extends beyond the domain")
        rankVolume += math.prod(rankSize)
    require(close(rankVolume, math.prod(size)), "rank-box volumes do not sum to the domain volume")

    preview = metadata["preview"]
    maxPoints = integer(preview["max_points_per_axis"], "preview.max_points_per_axis", minimum=1)
    cuts = preview["slice_coordinates_mm"]
    for plane, (_, _, normal) in PLANES.items():
        cellIndex(number(cuts[plane], f"slice coordinate {plane}"), normal, domain, plane)
    rows = {}
    with (directory / "slices.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["plane", "u_mm", "v_mm", "epsilon_r"],
                "unexpected slices.csv columns")
        for line, row in enumerate(reader, start=2):
            plane = row["plane"]
            require(plane in PLANES, f"CSV line {line}: unknown slice {plane!r}")
            u, v, epsilon = [number(float(row[key]), f"CSV line {line} {key}")
                             for key in ("u_mm", "v_mm", "epsilon_r")]
            uAxis, vAxis, normal = PLANES[plane]
            uIndex = cellIndex(u, uAxis, domain, f"CSV line {line} u")
            vIndex = cellIndex(v, vAxis, domain, f"CSV line {line} v")
            key = (plane, uIndex, vIndex)
            require(key not in rows, f"CSV line {line}: duplicate sampled cell {key}")
            require(close(epsilon, background) or close(epsilon, radiator["epsilon_r"]),
                    f"CSV line {line}: unexpected material epsilon {epsilon}")
            if insideReference is not None:
                point = [0.0, 0.0, 0.0]
                point[uAxis], point[vAxis], point[normal] = u, v, cuts[plane]
                expected = radiator["epsilon_r"] if insideReference(point) else background
                require(close(epsilon, expected),
                        f"CSV line {line}: independent {radiator['type']} reference disagrees at {point}")
            rows[key] = (u, v, epsilon)
    for plane, (uAxis, vAxis, _) in PLANES.items():
        du = 1 + (cells[uAxis] - 1) // maxPoints
        dv = 1 + (cells[vAxis] - 1) // maxPoints
        expected = {(plane, i, j) for i, j in itertools.product(
            range(0, cells[uAxis], du), range(0, cells[vAxis], dv))}
        actual = {key for key in rows if key[0] == plane}
        require(actual == expected, f"{plane} slice has missing or unexpected sampled cells")

    if compact and cells == [40, 40, 64]:
        centres = [[lower[axis] + (index + 0.5) * spacing[axis]
                    for index in range(cells[axis])] for axis in range(3)]
        yCount = sum(insideCompact(0.0, y, 25.0) for y in centres[1])
        xzCount = sum(insideCompact(x, 0.0, z) for x, z in itertools.product(centres[0], centres[2]))
        require(occupied == yCount * xzCount,
                f"independent quick-grid count {yCount * xzCount} != exported count {occupied}")
    if radiator["type"] == "brick":
        # A Cartesian box factorizes into independent intervals. This checks all
        # owned centres using O(Nx + Ny + Nz) work, even for large 3D meshes.
        tolerance = 1.0e-10
        axisCounts = [sum(brickLower[axis] - tolerance
                          <= lower[axis] + (index + 0.5) * spacing[axis]
                          <= brickLower[axis] + brickSize[axis] + tolerance
                          for index in range(cells[axis])) for axis in range(3)]
        expectedCount = math.prod(axisCounts)
        require(occupied == expectedCount,
                f"independent brick count {expectedCount} != exported count {occupied}")
    checkEmbeddedDiagnostics(directory, metadata,
                             (directory / "slices.csv").read_text(encoding="utf-8"))
    print(f"PASS {directory}: {total:,} cells, {occupied:,} dielectric, {rankCount} MPI rank(s)")
    return metadata, rows


## @brief Validate CLI output directories and compare each with the first run.
# @ingroup chdr_tests
# All directories must describe the same physical mesh and radiator; rank counts
# and rank boxes may differ. The global checksum is compared between runs, not
# independently reconstructed from all material cells.
# @return Zero on success or one after printing a caught validation/read failure.
# Argument parsing can exit with status two for invalid CLI usage.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+", help="mesh-1 output directories")
    args = parser.parse_args()
    reference = None
    try:
        for directory in args.directories:
            metadata, rows = readAndValidate(directory)
            if reference is None:
                reference = metadata, rows
                continue
            first, firstRows = reference
            for key in ("domain", "radiator", "background_epsilon_r", "preview"):
                compareValues(first[key], metadata[key], key)
            for key in ("total_cells", "dielectric_cells", "material_index_checksum"):
                require(first["diagnostics"][key] == metadata["diagnostics"][key],
                        f"diagnostics.{key} differs between runs")
            require(firstRows.keys() == rows.keys(), "sampled slice cells differ between runs")
            for key in firstRows:
                compareValues(list(firstRows[key]), list(rows[key]), f"slice {key}")
        if len(args.directories) > 1:
            print(f"PASS identical mesh and material diagnostics across {len(args.directories)} runs")
    except (OSError, KeyError, TypeError, ValueError, OverflowError, SyntaxError) as error:
        print(f"FAIL {directory}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
