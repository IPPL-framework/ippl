# compile_reader.py
import json
import hashlib
from pathlib import Path
from typing import List
from uuid import uuid4

from .model import TestResult, Attachment, StatusDetails, Label

def read_compile_test(test_dir: Path, test_run_id: str) -> List[TestResult]:
    """
    Loads a compilation test result from its specific directory.
    
    This function expects the given directory to contain a 'result.json' file
    and its associated log attachments (e.g., stdout.log, stderr.log).
    It returns a list containing a single TestResult, or an empty list if not found.
    """
    result_file = test_dir / "result.json"
    if not result_file.is_file():
        return []

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The stderr log provides the full trace for failures
    stderr_log_path = test_dir / "stderr.log"
    trace = ""
    if data.get("status") == "failed" and stderr_log_path.exists():
        with open(stderr_log_path, "r", encoding="utf-8") as f:
            trace = f.read()

    status_details_data = data.get("statusDetails", {})
    status_details_data['trace'] = trace

    attachments = []
    for att_data in data.get("attachments", []):
        source_path = test_dir / att_data["source"]
        if source_path.exists():
            attachments.append(Attachment(
                name=att_data["name"],
                source=str(source_path.resolve()),
                type=att_data.get("type", "text/plain")
            ))

    test_result = TestResult(
        name=data["name"],
        status=data["status"],
        uuid=str(uuid4()),
        testRunId=test_run_id,
        historyId=generate_history_id(data["name"]),
        statusDetails=StatusDetails(**status_details_data),
        attachments=attachments,
        labels=[
            Label(name="suite", value="Compilation Tests"),
            Label(name="feature", value="Build System")
        ]
    )
    
    return [test_result]

def generate_history_id(name: str) -> str:
    """Generate a stable hash-based historyId from the test name."""
    return hashlib.sha1(name.encode("utf-8")).hexdigest()
