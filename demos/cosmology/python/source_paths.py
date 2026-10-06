## @file source_paths.py
# @brief Resolve cosmology source paths after Python scripts were grouped together.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Resolve cosmology source paths after Python scripts were grouped together."""
from pathlib import Path

## @brief Accept a cosmology root or its python directory; return an absolute path.
# @see cosmology_tools
#
# @param name Stable artifact/run/check identifier as defined by the caller.
# @param source_root Cosmology source root or its Python directory for the mixed-source resolver.
# @return Resolved pathlib.Path to the corresponding maintained source.
def source_path(name, source_root=None):
    """Accept a cosmology root or its python directory; return an absolute path."""
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parent.parent
    if root.name == "python":
        root = root.parent
    relative = Path(name)
    if source_root is not None and (root / relative).is_file():
        return root / relative
    return root / (Path("python") / relative if relative.suffix == ".py" else relative)
