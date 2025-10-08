from pathlib import Path
from typing import List
import hashlib
import re
from xml.etree import ElementTree as ET
from uuid import uuid4
from .model import TestResult, Attachment, StatusDetails, Label
import json


def read_gcc_xml_folder(folder: Path, test_run_id: str) -> List[TestResult]:
    """Parse all gcc-xml files in a folder and convert them to TestResult objects."""
    test_results = []

    job_name = ""
    context_file = folder / "context.json"
    if context_file.exists():
        with open(context_file, "r") as f:
            context_data = json.load(f)
            job_name = context_data.get("name", "")

    for xml_file in folder.glob("*.xml"):
        test_results.extend(parse_ctest_xml_file(xml_file, test_run_id, job_name))
    return test_results

def parse_ctest_xml_file(xml_file: Path, test_run_id: str, job_name: str) -> List[TestResult]:
    """Parse a single gcc-xml file (CTest XML format) into a list of TestResult."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    results = []

    timestamp = root.attrib.get("timestamp", None)

    for testcase in root.findall("testcase"):
        base_name = testcase.attrib.get("name")
        time_sec = float(testcase.attrib.get("time", "0"))
        classname = testcase.attrib.get("classname", "")
        status = testcase.attrib.get("status", "run")

        if testcase.find("failure") is not None:
            result_status = "failed"
        elif testcase.find("skipped") is not None:
            result_status = "skipped"
        else:
            result_status = "passed"

        system_out = testcase.findtext("system-out", default="")

        feature, story = extract_feature_and_story(classname, base_name)

        final_test_name = f"{job_name}: {base_name}" if job_name else base_name

        if job_name:
            final_suite_name = job_name
        else:
            final_suite_name = root.attrib.get("name", "gcc-xml-suite")

        labels = [
            Label(name="suite", value=final_suite_name),
            Label(name="feature", value=feature),
            Label(name="story", value=story)
        ]

        test_result = TestResult(
            name=final_test_name,
            status=result_status,
            uuid=str(uuid4()),
            testRunId=test_run_id,  
            historyId=generate_history_id(final_test_name),
            duration=int(time_sec * 1000),  # convert to milliseconds
            attachments=[],
            statusDetails=StatusDetails(
                message=None,
                trace=system_out
            ),
            labels=labels
        )

        results.append(test_result)

    return results 

def generate_history_id(name: str) -> str:
    """Generate a stable hash-based historyId from test name and classname."""
    combined = f"{name}"
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()

def extract_feature_and_story(classname: str, name: str) -> (str, str):
    feature = classname.split(".")[0] if "." in classname else classname

    if "." in classname:
        raw_story = classname.split(".")[-1]
    else:
        raw_story = name.split(".")[-1] if "." in name else name

    story_clean = re.split(r"<", raw_story)[0]

    return feature, story_clean
