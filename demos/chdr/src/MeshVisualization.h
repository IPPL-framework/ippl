/** @file MeshVisualization.h
 * @brief Python report helpers embedded by MaterialViewer.h in the generated viewer.
 * @ingroup chdr_output
 * @details The emitted Matplotlib code renders the analytic geometry and actual
 * IPPL field samples. It never reconstructs missing samples or performs a solve.
 * The raw Python string is source code, not a separate handwritten Python module.
 */
#ifndef CHDR_MESH_VISUALIZATION_H
#define CHDR_MESH_VISUALIZATION_H

#include <ostream>
#include <stdexcept>

namespace chdr {

    /**
     * @brief Emit reusable Python functions for the four-panel mesh PDF report.
     * @ingroup chdr_output
     *
     * The generated saveMeshReport(metadata, slices, outputPath, dpi=180,
     * maxRankBoxes=64) function uses the supplied mesh metadata and actual material
     * samples, without reading diagnostic files or requiring PyVista. NumPy and
     * Matplotlib are imported only when generating the report, with the Agg
     * backend selected for headless PDF output.
     * Metadata schema version 2 describes either a triangular prism or an
     * axis-aligned brick in the radiator object.
     *
     * Geometry and slice coordinates in the diagnostic files are in millimetres.
     * Slice values come from the actual IPPL material field. For scalability the
     * diagnostic may subsample those values; the viewer labels this explicitly
     * and must not be used to infer that an unsampled thin feature is absent.
     * Rank bounding boxes are display-thinned when many ranks are present.
     *
     * @param script Stream receiving Python helper and report function definitions.
     * @pre The caller's generated Python preamble is already written.
     * @post Python helper definitions are appended; the stream remains open.
     * @note There are no C++ mesh operations here. The supplied metadata and slices
     * are consumed only when the generated Python saveMeshReport() is called.
     * @throws std::runtime_error If the Python definitions cannot be written.
     */
    inline void writeMeshReportFunctions(std::ostream& script) {
        script << R"CHDRPY(
def boxEdges(lower, size, np):
    import itertools
    corners = np.array([lower + np.array(bits) * size
                        for bits in itertools.product((0, 1), repeat=3)])
    return [corners[[i, j]] for i in range(8) for j in range(i + 1, 8)
            if np.count_nonzero(corners[i] != corners[j]) == 1]


def radiatorMesh(radiator, np):
    """Return vertices, outward polygon faces and edges of either radiator."""
    if radiator["type"] == "prism":
        bottom = np.array(radiator["vertices"], dtype=float, copy=True)
        axis = np.asarray(radiator["axis"], dtype=float)
        # Canonical winding gives outward faces for either input triangle order.
        if np.dot(np.cross(bottom[1] - bottom[0], bottom[2] - bottom[0]), axis) < 0:
            bottom[[1, 2]] = bottom[[2, 1]]
        top = bottom + float(radiator["height"]) * axis
        vertices = np.concatenate((bottom, top))
        faces = [vertices[[0, 2, 1]], vertices[[3, 4, 5]]]
        faces += [vertices[[i, (i + 1) % 3, (i + 1) % 3 + 3, i + 3]]
                  for i in range(3)]
        edges = [(i, (i + 1) % 3) for i in range(3)]
        edges += [(i + 3, (i + 1) % 3 + 3) for i in range(3)]
        edges += [(i, i + 3) for i in range(3)]
    elif radiator["type"] == "brick":
        lower = np.asarray(radiator["lower"], dtype=float)
        size = np.asarray(radiator["size"], dtype=float)
        corners = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
        vertices = lower + corners * size
        faceIndices = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                       (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
        faces = [vertices[list(indices)] for indices in faceIndices]
        edges = [(i, (i + 1) % 4) for i in range(4)]
        edges += [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
        edges += [(i, i + 4) for i in range(4)]
    else:
        raise ValueError("Unsupported radiator type: " + str(radiator["type"]))
    return vertices, faces, edges


def analyticCut(vertices, edges, normalAxis, coordinate, np):
    """Intersect the convex radiator edges with a Cartesian slice plane."""
    tolerance = 1.0e-10 * max(1.0, float(np.ptp(vertices, axis=0).max()))
    points = []
    for first, second in edges:
        a, b = vertices[first], vertices[second]
        da, db = a[normalAxis] - coordinate, b[normalAxis] - coordinate
        if abs(da) <= tolerance:
            points.append(a)
        if abs(db) <= tolerance:
            points.append(b)
        if (da < -tolerance and db > tolerance) or (db < -tolerance and da > tolerance):
            points.append(a + da / (da - db) * (b - a))
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - other) <= tolerance for other in unique):
            unique.append(point)
    if len(unique) < 3:
        return None
    inPlane = [axis for axis in range(3) if axis != normalAxis]
    projected = np.asarray(unique)[:, inPlane]
    centre = projected.mean(axis=0)
    angles = np.arctan2(projected[:, 1] - centre[1], projected[:, 0] - centre[0])
    polygon = projected[np.argsort(angles)]
    return np.concatenate((polygon, polygon[:1]))


def displayEdges(centres, lower, upper, np):
    # For a full slice these are the real cell edges. For a sampled slice they
    # are display bins halfway between exported centres, not subgrid geometry.
    return np.concatenate(([lower], 0.5 * (centres[:-1] + centres[1:]), [upper]))


def saveMeshReport(metadata, slices, outputPath, dpi=180, maxRankBoxes=64):
    """Save a PDF with the analytic geometry and actual IPPL material samples.

    Coordinates are in millimetres. For sampled slices, the nearest-sample
    display bins do not resolve intervening staircase cells or thin features.
    This report does not rerasterize the analytic radiator.
    """
    from pathlib import Path
    from numbers import Integral
    if not isinstance(dpi, Integral) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    if not isinstance(maxRankBoxes, Integral) or maxRankBoxes < 0:
        raise ValueError("maxRankBoxes must be a nonnegative integer")
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, Normalize
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    except ImportError as error:
        raise ImportError("The mesh report requires NumPy and Matplotlib: " + str(error)) from error

    if metadata.get("schema_version") != 2 or metadata.get("units") != "mm":
        raise ValueError("Expected mesh metadata schema_version=2 and units='mm'")

    domain = metadata["domain"]
    lower = np.asarray(domain["lower"], dtype=float)
    size = np.asarray(domain["size"], dtype=float)
    upper = lower + size
    cells = np.asarray(domain["cells"], dtype=int)
    spacing = np.asarray(domain["spacing"], dtype=float)
    radiator = metadata["radiator"]
    vertices, faces, edges = radiatorMesh(radiator, np)
    radiatorLabel = f"Analytic {radiator['type']}"
    preview = metadata.get("preview", {})
    sliceCoordinates = preview.get("slice_coordinates_mm", {})
    diagnostics = metadata["diagnostics"]
    epsilonBackground = float(metadata["background_epsilon_r"])
    epsilonRadiator = float(radiator["epsilon_r"])
    epsilonMin, epsilonMax = sorted((epsilonBackground, epsilonRadiator))
    if epsilonMin == epsilonMax:
        epsilonMin -= 0.05
        epsilonMax += 0.05
    colors = ["#eef3f5", "#277da8"] if epsilonRadiator >= epsilonBackground else ["#277da8", "#eef3f5"]
    cmap = ListedColormap(colors)
    norm = Normalize(epsilonMin, epsilonMax)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "semibold", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax3d.add_collection3d(Line3DCollection(boxEdges(lower, size, np),
                                         colors="#243746", linewidths=1.1))
    ax3d.add_collection3d(Poly3DCollection(faces, facecolors="#3a91b5", alpha=0.32,
                                          edgecolors="#176182", linewidths=1.1))
    ranks = metadata.get("ranks", [])
    shown = min(maxRankBoxes, len(ranks))
    if shown > 0:
        selected = np.unique(np.linspace(0, len(ranks) - 1, shown, dtype=int))
        rankEdges = []
        for index in selected:
            rank = ranks[index]
            rankEdges.extend(boxEdges(np.asarray(rank["lower"]), np.asarray(rank["size"]), np))
        ax3d.add_collection3d(Line3DCollection(rankEdges, colors="#a47b41", alpha=0.38,
                                              linewidths=0.65, linestyles="dashed"))
    ax3d.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
             zlim=(lower[2], upper[2]), xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
    ax3d.set_box_aspect(size)
    ax3d.view_init(elev=23, azim=-55)
    ax3d.set_title(f"{radiatorLabel} and computational domain", pad=15)
    handles = [Line2D([], [], color="#176182", lw=3, label=radiatorLabel),
               Line2D([], [], color="#243746", label="Domain")]
    if shown:
        handles.append(Line2D([], [], color="#a47b41", ls="--",
                              label=f"Rank boxes ({shown}/{len(ranks)})"))
    ax3d.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    for plane, position in zip(("xz", "xy", "yz"), ((0, 1), (1, 0), (1, 1))):
        ax = fig.add_subplot(grid[position])
        selectedRows = [row for row in slices if row["plane"] == plane]
        if not selectedRows:
            ax.text(0.5, 0.5, f"No {plane} slice exported", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        u = np.unique([float(row["u_mm"]) for row in selectedRows])
        v = np.unique([float(row["v_mm"]) for row in selectedRows])
        values = np.full((len(v), len(u)), np.nan)
        for row in selectedRows:
            values[np.searchsorted(v, float(row["v_mm"])),
                   np.searchsorted(u, float(row["u_mm"]))] = float(row["epsilon_r"])
        if not np.isfinite(values).all() or len(selectedRows) != values.size:
            raise ValueError(f"Slice {plane} must contain exactly one finite value per sampled centre")
        uAxis, vAxis = ("xyz".index(letter) for letter in plane)
        normalAxis = next(axis for axis in range(3) if axis not in (uAxis, vAxis))
        uEdges = displayEdges(u, lower[uAxis], upper[uAxis], np)
        vEdges = displayEdges(v, lower[vAxis], upper[vAxis], np)
        artist = ax.pcolormesh(uEdges, vEdges, values, cmap=cmap, norm=norm,
                              shading="flat", rasterized=True)
        coordinate = sliceCoordinates.get(plane)
        subtitle = ""
        if coordinate is not None:
            outline = analyticCut(vertices, edges, normalAxis, float(coordinate), np)
            if outline is not None:
                ax.plot(outline[:, 0], outline[:, 1], color="#b84924", lw=1.2,
                        ls="--", label="Analytic interface")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            subtitle = f" at {'xyz'[normalAxis]}={float(coordinate):.4g} mm"
        thinned = len(u) < cells[uAxis] or len(v) < cells[vAxis]
        samplingLabel = "sampled cells" if thinned else "all cells in slice"
        ax.set_title(f"{plane[0]}–{plane[1]} slice{subtitle}\n"
                     f"{len(u)} × {len(v)} {samplingLabel}", fontsize=11)
        ax.set(xlabel=f"{plane[0]} [mm]", ylabel=f"{plane[1]} [mm]",
               xlim=(lower[uAxis], upper[uAxis]), ylim=(lower[vAxis], upper[vAxis]))
        ax.set_aspect("equal")
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.78, pad=0.025)
        colorbar.set_label(r"$\epsilon_r$")
        colorbar.set_ticks(sorted(set((epsilonBackground, epsilonRadiator))))

    cellText = " × ".join(str(value) for value in cells)
    spacingText = " × ".join(f"{value:.4g}" for value in spacing)
    rankCount = diagnostics.get("rank_count", len(ranks))
    dielectricCells = int(diagnostics["dielectric_cells"])
    voxelVolume = float(diagnostics["voxel_volume_mm3"])
    analyticVolume = float(radiator["analytic_volume_mm3"])
    volumeError = 100.0 * (voxelVolume / analyticVolume - 1.0)
    fig.suptitle("ChDR mesh inspection\n"
                 f"{cellText} cells · h = {spacingText} mm · {rankCount} MPI rank(s)",
                 fontsize=15, fontweight="semibold")
    footer = (f"Dielectric: {dielectricCells:,} cells · volume {voxelVolume:,.5g} mm³ "
              f"(analytic {analyticVolume:,.5g} mm³, difference {volumeError:+.3g}%)\n"
              "Slices contain actual material-field samples. Decimated previews can miss thin features; "
              "display bins between samples are not additional voxels.")
    fig.supxlabel(footer, fontsize=9)
    outputPath = Path(outputPath)
    try:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, format="pdf", dpi=dpi)
    finally:
        plt.close(fig)
)CHDRPY";

        if (!script) {
            throw std::runtime_error("Cannot write Python mesh report functions");
        }
    }

}  // namespace chdr

#endif  // CHDR_MESH_VISUALIZATION_H
