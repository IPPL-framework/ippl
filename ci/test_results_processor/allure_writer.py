from pathlib import Path
from typing import List
from .model import PipelineResult, TestResult
import json
import uuid
import shutil

def write_allure_results(pipeline_result: PipelineResult, output_folder: Path):
    """
    Write Allure result files from a PipelineResult.
    Each TestResult becomes one <uuid>-result.json file in the output folder.
    Also copies all attachment files into the output folder with renamed source names.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for suite in pipeline_result.suites:
        for test in suite.tests:
            allure_result = convert_testresult_to_allure(test, output_folder)
            file_name = f"{test.uuid}-result.json"
            file_path = output_folder / file_name

            with open(file_path, "w") as f:
                json.dump(allure_result, f, indent=2)

            print(f"Wrote Allure result: {file_path}")

def convert_testresult_to_allure(test: TestResult, output_folder: Path) -> dict:
    """
    Convert internal TestResult to Allure's result JSON format.
    Copies and remaps attachments to the output folder.
    """
    allure_status = convert_status_to_allure(test.status)
    attachments_json = []

    for attachment in test.attachments or []:
        ext = Path(attachment.source).suffix or ".txt"
        allure_attachment_name = f"{uuid.uuid4()}-attachment{ext}"
        dest_path = output_folder / allure_attachment_name
        try:
            shutil.copy(attachment.source, dest_path)
        except FileNotFoundError:
            print(f"Warning: attachment not found: {attachment.source}")
            continue

        attachments_json.append({
            "name": attachment.name,
            "source": allure_attachment_name,
            "type": attachment.type or "text/plain"
        })



    allure_result = {
        "uuid": test.uuid,
        "name": test.name,
        "historyId": test.historyId,
        "status": allure_status,
        "labels": [
            {"name": label.name, "value": label.value} for label in (test.labels or [])
        ],
        "start": test.start or 0,
        "stop": test.duration or 0, 
        "parameters": test.parameters or [],
        "steps": [
            {
            "name": "run",
            "statusDetails": 
                {
                "known": True,
                "muted": False,
                "flaky": False,
                "message": "Test output" if test.statusDetails else None,
                "trace": test.statusDetails.trace if test.statusDetails else None
                }
            }
        ],
        
        "attachments": attachments_json,
    }

    return allure_result

def convert_status_to_allure(status: str) -> str:
    """
    Map your status to Allure's expected status.
    """
    if status.lower() in ("passed", "failed", "skipped", "broken"):
        return status.lower()
    return "unknown"

