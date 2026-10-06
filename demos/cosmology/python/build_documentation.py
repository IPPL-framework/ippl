#!/usr/bin/env python3
## @file build_documentation.py
# @brief Build HTML/XML documentation and reject warnings or missing API coverage.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
## @file build_documentation.py
# @brief Build and audit the strict cosmology Doxygen manual and source APIs.
# @ingroup cosmology_python
# @see cosmology_documentation_quality
"""Build HTML/XML documentation and reject warnings or missing API coverage.

This host-only workflow never launches cosmology, downloads/builds references,
rewrites retained evidence or imports the Python source files it documents.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

## @brief Maintained cosmology source root containing the documentation and Python tools.
## @var SourceRoot
# @brief Named SourceRoot protocol/schema value; the source initializer records its exact contents.
SourceRoot = Path(__file__).resolve().parents[1]
## @brief Code/evidence directories intentionally excluded from maintained-source extraction.
## @var ExcludedDirectories
# @brief Named ExcludedDirectories protocol/schema value; the source initializer records its exact contents.
ExcludedDirectories = {'results','ippl-cosmology-paper','__pycache__','docs'}


## @brief Enumerate maintained compiled and Python source files, excluding evidence/upstream inputs.
# @param root Cosmology source root to audit; may be a comment-only staging tree.
# @return Sorted source paths with suffixes .h, .hpp, .c, .cpp or .py.
def source_files(root):
    root=Path(root)
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and p.suffix in {'.h','.hpp','.c','.cpp','.py'}
                  and not ExcludedDirectories.intersection(p.relative_to(root).parts))


## @brief Enumerate module/class-level Python APIs without counting nested local callbacks.
# @param body AST statement list of a module or class.
# @param prefix Lexical class-name prefix for qualified API identity.
# @return Iterator of class/function nodes and their qualified local names.
def python_apis(body,prefix=''):
    for node in body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            yield node,prefix+node.name
            if isinstance(node,ast.ClassDef):
                yield from python_apis(node.body,prefix+node.name+'.')


## @brief Recover semantic text from a Doxygen XML documentation element.
# @param node XML element, or None for absent documentation.
# @return Whitespace-trimmed text; empty string means no documented content.
def xml_text(node):
    return ''.join(node.itertext()).strip() if node is not None else ''


## @brief Check source-file/API presence, documentation and parameter contracts in Doxygen XML.
# @param source Cosmology source root corresponding exactly to the generated XML.
# @param xml_dir Doxygen XML output directory.
# @return JSON-safe coverage counts and explicit missing-contract failures.
def audit_coverage(source,xml_dir):
    source=Path(source).resolve()
    compounds=[]
    for path in sorted(Path(xml_dir).glob('*.xml')):
        if path.name in {'index.xml','Doxyfile.xml'}:continue
        tree=ET.parse(path)
        compounds.extend(tree.findall('compounddef'))
    documented_files=set();python_symbols=set();members={};failures=[]
    for compound in compounds:
        kind=compound.attrib.get('kind')
        location=compound.find('location')
        if kind=='file':
            location_file=location.attrib.get('file') if location is not None else compound.findtext('compoundname')
            if location_file:
                path=Path(location_file)
                if not path.is_absolute():path=source/path
                documented_files.add(path.resolve())
        if kind=='class':
            name=compound.findtext('compoundname','')
            if compound.attrib.get('language')=='Python':name=name.replace('::','.')
            if xml_text(compound.find('briefdescription')) or xml_text(compound.find('detaileddescription')):
                python_symbols.add(name)
        for member in compound.findall('.//memberdef'):
            identifier=member.attrib.get('id')
            if identifier in members:continue
            members[identifier]=member
            name=member.findtext('qualifiedname') or member.findtext('definition','').removeprefix('def ')
            documented=bool(xml_text(member.find('briefdescription')) or xml_text(member.find('detaileddescription')))
            if documented:python_symbols.add(name)
            if not documented:
                failures.append('Undocumented extracted member: '+name)
            if member.attrib.get('kind')=='function':
                declared=[p.findtext('declname') for p in member.findall('param')]
                declared=[name.lstrip('*') for name in declared if name and name not in {'self','cls'}]
                parameter_names={xml_text(p).lstrip('*') for p in member.findall(
                    ".//parameterlist[@kind='param']/parameteritem/parameternamelist/parametername")}
                for arg in declared:
                    if arg not in parameter_names:failures.append(f'Missing parameter contract: {name}({arg})')
    files=source_files(source);python_count=0
    for path in files:
        if path.resolve() not in documented_files:
            failures.append('Missing source file: '+str(path.relative_to(source)))
        if path.suffix=='.py':
            tree=ast.parse(path.read_text())
            for node,name in python_apis(tree.body):
                python_count+=1
                symbol=path.stem+'.'+name
                if symbol not in python_symbols:
                    failures.append('Missing documented Python API: '+symbol)
    return {'source_files':len(files),'python_api_declarations':python_count,
            'extracted_members':len(members),'documented_python_symbols':len(python_symbols),
            'failures':sorted(set(failures)),
            'scope':'Maintained C/C++/Python files and module/class APIs; local callbacks and CLI dispatch are implementation detail.'}


## @brief Generate Doxygen configuration, build the site and fail on warnings or coverage errors.
# @param source Cosmology source root containing the exact input version.
# @param manual Root of authored .dox chapters, bibliography and Doxyfile.in.
# @param output Dedicated generated-documentation output directory.
# @param doxygen Doxygen executable path; version 1.14 or newer is required.
# @param formulas Formula backend: auto chooses offline SVG when TeX tools are present, otherwise MathJax; svg/mathjax selects explicitly.
# @return Validated machine-readable coverage report, including the generated HTML index path.
# @throws RuntimeError If a required tool, Doxygen version, documentation or API contract is invalid.
def build(source,manual,output,doxygen='doxygen',formulas='auto'):
    source=Path(source).resolve();manual=Path(manual).resolve();output=Path(output).resolve()
    if output==source or output==manual:
        raise ValueError('Documentation output must be separate from authored source/manual')
    binary=shutil.which(str(doxygen))
    if binary is None or shutil.which('bibtex') is None:
        raise RuntimeError('Doxygen >=1.14 and BibTeX must be available in PATH')
    version=subprocess.check_output([binary,'--version'],text=True).strip()
    match=re.match(r'(\d+)\.(\d+)',version)
    if not match or tuple(map(int,match.groups()))<(1,14):
        raise RuntimeError('This checked documentation configuration requires Doxygen >=1.14')
    svg_tools=all(shutil.which(tool) for tool in ('latex','dvips','gs')) and bool(shutil.which('pdf2svg') or shutil.which('inkscape'))
    if formulas=='auto':formulas='svg' if svg_tools else 'mathjax'
    if formulas not in ('svg','mathjax') or (formulas=='svg' and not svg_tools):
        raise RuntimeError('Offline SVG formulas require latex, dvips, Ghostscript and pdf2svg/inkscape; select --formulas mathjax otherwise')
    output.mkdir(parents=True,exist_ok=True)
    template=(manual/'Doxyfile.in').read_text()
    substitutions={'SOURCE_ROOT':str(source),'MANUAL_ROOT':str(manual),
                   'OUTPUT_ROOT':str(output),
                   'FILTER_COMMAND':shlex.join([sys.executable,str(SourceRoot/'python/doxygen_filter.py')]),
                   'HAVE_DOT':'YES' if shutil.which('dot') else 'NO',
                   'USE_MATHJAX':'YES' if formulas=='mathjax' else 'NO'}
    for key,value in substitutions.items():template=template.replace('@'+key+'@',value)
    if re.search(r'@[A-Z_]+@',template):raise RuntimeError('Unresolved Doxygen configuration placeholder')
    config=output/'Doxyfile';config.write_text(template)
    with (output/'build.log').open('w') as log:
        result=subprocess.run([binary,str(config)],stdout=log,stderr=subprocess.STDOUT)
    warnings=(output/'warnings.log').read_text() if (output/'warnings.log').exists() else ''
    if result.returncode or warnings.strip():
        raise RuntimeError(f'Doxygen failed ({result.returncode}); inspect {output}/warnings.log and build.log')
    coverage=audit_coverage(source,output/'xml')
    coverage.update(doxygen_version=version,source_root=str(source),manual_root=str(manual),
                    html_index=str(output/'html/index.html'),formula_backend=formulas,warning_count=0)
    (output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    if coverage['failures']:
        raise RuntimeError('Documentation coverage failed:\n'+'\n'.join(coverage['failures'][:40]))
    print(f"Doxygen {version}: {coverage['source_files']} files, "
          f"{coverage['python_api_declarations']} Python declarations; no warnings or missing contracts.")
    print('HTML: '+coverage['html_index'])
    return coverage


## @brief Parse the documentation-build interface and perform the strict generation/audit.
# @return Zero after a warning-free, coverage-complete build; exceptions report failure.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=SourceRoot)
    parser.add_argument('--manual-root',type=Path,default=SourceRoot/'docs')
    parser.add_argument('--output',type=Path,default=SourceRoot.parents[1]/'build_openmp/docs/cosmology')
    parser.add_argument('--doxygen',default='doxygen')
    parser.add_argument('--formulas',choices=('auto','svg','mathjax'),default='auto')
    args=parser.parse_args()
    build(args.source_root,args.manual_root,args.output,args.doxygen,args.formulas)
    return 0


## @cond CLI_DISPATCH
## @cond CLI_DISPATCH
if __name__=='__main__':
    try:sys.exit(main())
    except (RuntimeError,ValueError,OSError) as error:
        print('Documentation error: '+str(error),file=sys.stderr)
        sys.exit(1)
## @endcond
## @endcond
