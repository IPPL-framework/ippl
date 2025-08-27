from pathlib import Path
from typing import List
from .model import TestResult, PipelineResult
from .aggregator import aggregate_pipeline
from .converter import read_gcc_xml_folder
from .compile_reader import read_compile_test
from .json_reader import read_json_folder
import json

FOLDER_READERS = {
    "gcc-xml": read_gcc_xml_folder,
    "json": read_json_folder,
    "compile-test": read_compile_test
}

def load_all_jobs_as_pipeline(pipeline_folder: Path) -> PipelineResult:
    """Load all jobs and aggregate into a single PipelineResult (in memory only)."""
    all_results = []
    pipeline_id = pipeline_folder.name
    jobs_folder = pipeline_folder / "jobs"

    metadata_path = pipeline_folder / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata_path}")

    for job_folder in jobs_folder.iterdir():
        if job_folder.is_dir():
            job_results = load_test_results_from_job_folder(job_folder, test_run_id=pipeline_id)
            all_results.extend(job_results)
    return aggregate_pipeline(pipeline_id, all_results, metadata)

def load_test_results_from_job_folder(job_folder: Path, test_run_id: str) -> List[TestResult]:
    """Load all tests from job folder. Known formats only."""
    test_results = []

    for subfolder in job_folder.iterdir():
        if subfolder.is_dir():
            reader = FOLDER_READERS.get(subfolder.name)
            if reader:
                test_results.extend(reader(subfolder, test_run_id=test_run_id))

    return test_results

