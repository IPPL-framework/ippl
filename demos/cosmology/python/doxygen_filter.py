#!/usr/bin/env python3
## @file doxygen_filter.py
# @brief Documentation-only filter; emitted text is never imported or executed.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
## @file doxygen_filter.py
# @brief Normalize required Python dataclass annotations for Doxygen's parser only.
# @ingroup cosmology_python
# @see cosmology_documentation_quality
"""Documentation-only filter; emitted text is never imported or executed.

Doxygen 1.14 does not recognize every bare annotated dataclass field. Replace
such a line with a parseable placeholder assignment while retaining line
count. The documentation configuration hides initializers and displays the
original unfiltered source, so placeholders are not claimed as runtime defaults.
"""
from __future__ import annotations
import ast
from pathlib import Path
import sys


## @brief Normalize only bare annotated fields without changing line positions.
# @param path Python input file path supplied by Doxygen.
# @return Documentation-parser text; the original file is never modified.
def filter_source(path):
    source=Path(path).read_text()
    tree=ast.parse(source)
    lines=source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node,ast.AnnAssign) and node.value is None and isinstance(node.target,ast.Name):
            if node.lineno!=node.end_lineno:
                raise ValueError('Multiline required annotation needs an explicit documentation filter case')
            lines[node.lineno-1]=' '*node.col_offset+node.target.id+' = None\n'
    return ''.join(lines)


## @brief Write the documentation-only transformed source to standard output.
# @return Zero after writing one input file; raises on invalid input.
def main():
    if len(sys.argv)!=2:
        raise ValueError('Doxygen filter expects exactly one source filename')
    sys.stdout.write(filter_source(sys.argv[1]))
    return 0


## @cond CLI_DISPATCH
## @cond CLI_DISPATCH
if __name__=='__main__':
    sys.exit(main())
## @endcond
## @endcond
