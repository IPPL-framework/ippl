#!/usr/bin/env python3
## @file test_documentation.py
# @brief Check that documentation normalization is non-executing and coverage is strict.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
## @file test_documentation.py
# @brief Regressions for documentation-only parsing and strict API coverage failures.
# @ingroup cosmology_python
# @see cosmology_documentation_quality
"""Check that documentation normalization is non-executing and coverage is strict."""
from pathlib import Path
import ast
import sys
import tempfile
import unittest

## @cond RUNTIME_SETTINGS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
## @endcond
from build_documentation import audit_coverage
from doxygen_filter import filter_source


## @brief Verify source-preserving parser normalization and missing-contract rejection.
class DocumentationTests(unittest.TestCase):
    ## @brief Ensure the filter changes only emitted parser text and preserves original bytes/line count.
    # @return None; unittest assertions report a violation.
    def test_required_annotations_are_documentation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.py'
            original='class Model:\n    seed: int\n    flag: bool = True\n'
            path.write_text(original)
            filtered=filter_source(path)
            self.assertEqual(path.read_text(),original)
            self.assertEqual(len(filtered.splitlines()),len(original.splitlines()))
            self.assertIn('seed = None',filtered)
            self.assertIn('flag: bool = True',filtered)
            ast.parse(filtered)

    ## @brief Verify that an absent required source/API yields an explicit coverage failure.
    # @return None; unittest assertions report a violation.
    def test_missing_api_is_not_hidden_by_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';xml=root/'xml'
            source.mkdir();xml.mkdir()
            (source/'sample.py').write_text('def needed(value):\n    return value\n')
            report=audit_coverage(source,xml)
            self.assertIn('Missing documented Python API: sample.needed',report['failures'])
            self.assertIn('Missing source file: sample.py',report['failures'])

    ## @brief Exercise Python class namespace spelling and function-parameter coverage separately.
    # @return None; unittest assertions report a violation.
    def test_python_class_scope_and_parameter_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';xml=root/'xml'
            source.mkdir();xml.mkdir()
            path=source/'sample.py'
            path.write_text('class Model:\n    def evaluate(self, value):\n        return value\n')
            (xml/'file.xml').write_text(f'<doxygen><compounddef kind="file"><compoundname>sample.py</compoundname><location file="{path}"/></compounddef></doxygen>')
            text='''<doxygen><compounddef kind="class" language="Python"><compoundname>sample::Model</compoundname><briefdescription><para>Model contract.</para></briefdescription><sectiondef><memberdef id="evaluate" kind="function"><name>evaluate</name><qualifiedname>sample.Model.evaluate</qualifiedname><briefdescription><para>Evaluate a scalar.</para></briefdescription><param><declname>self</declname></param><param><declname>value</declname></param><detaileddescription>{parameters}</detaileddescription></memberdef></sectiondef></compounddef></doxygen>'''
            parameters='<parameterlist kind="param"><parameteritem><parameternamelist><parametername>value</parametername></parameternamelist><parameterdescription><para>Scalar input.</para></parameterdescription></parameteritem></parameterlist>'
            (xml/'class.xml').write_text(text.replace('{parameters}',parameters))
            self.assertEqual(audit_coverage(source,xml)['failures'],[])
            (xml/'class.xml').write_text(text.replace('{parameters}',''))
            self.assertIn('Missing parameter contract: sample.Model.evaluate(value)',audit_coverage(source,xml)['failures'])


## @cond CLI_DISPATCH
## @cond CLI_DISPATCH
if __name__=='__main__':unittest.main()
## @endcond
## @endcond
