from typing import List
from .model import TestSuite, TestResult, PipelineResult

def group_tests_by_suite(test_results: List[TestResult]) -> List[TestSuite]:
    """
    Group TestResult objects into TestSuites based on their 'suite' label.
    If a test has no suite label, it goes into a 'default' suite.
    """
    suites = {}
    for test in test_results:
        # Default suite name
        suite_name = "default"

        # Try to find a 'suite' label in the test labels
        if test.labels:
            for label in test.labels:
                if label.name == "suite":
                    suite_name = label.value
                    break

        suites.setdefault(suite_name, []).append(test)

    # Create TestSuite objects
    return [TestSuite(name=name, tests=tests) for name, tests in suites.items()]

def aggregate_pipeline(pipeline_id: str, test_results: List[TestResult]) -> PipelineResult:
    """
    Aggregate a list of TestResults into a PipelineResult.
    Group tests into suites and wrap everything into PipelineResult.
    """
    suites = group_tests_by_suite(test_results)
    return PipelineResult(pipeline_id=pipeline_id, suites=suites)

