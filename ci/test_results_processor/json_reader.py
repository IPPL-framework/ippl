import json
from pathlib import Path
from typing import List
from .model import TestResult

def read_json_folder(folder: Path, test_run_id: str) -> List[TestResult]:
    """
    Load test results from a folder containing .json files.
    Each JSON file must map directly to a TestResult (or subset thereof).
    """
    results = []
    for file in folder.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Add test_run_id if not present
        data.setdefault("testRunId", test_run_id)

        for attachment in data.get("attachments", []):
            rel_path = Path(attachment["source"])
            abs_path = (folder / rel_path).resolve()
            attachment["source"] = str(abs_path)

        # Let Pydantic handle field validation & defaults
        results.append(TestResult(**data))

    return results

