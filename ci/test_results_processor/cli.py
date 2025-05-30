import argparse
from pathlib import Path
from .loader import load_all_jobs_as_pipeline
from .aggregator import aggregate_pipeline
from .model import PipelineResult
from .allure_writer import write_allure_results

def main():
    parser = argparse.ArgumentParser(description="Test Results Processor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_parser = subparsers.add_parser("load-pipeline")
    load_parser.add_argument("pipeline_folder", type=Path)

    allure_parser = subparsers.add_parser("generate-allure")
    allure_parser.add_argument("pipeline_folder", type=Path)
    allure_parser.add_argument("allure_results_folder", type=Path, help="Output folder for Allure results")


    args = parser.parse_args()
    
    if args.command == "load-pipeline":
        run_load_pipeline(args.pipeline_folder)
    elif args.command == "generate-allure":
        run_generate_allure(args.pipeline_folder, args.allure_results_folder)


def run_load_pipeline(pipeline_folder: Path):
    print(f"Loading pipeline from: {pipeline_folder}")
    pipeline_result = load_all_jobs_as_pipeline(pipeline_folder)
    print(f"Loaded {len(pipeline_result.suites)} suites")
    print(pipeline_result.model_dump_json(indent=2))


def run_generate_allure(pipeline_folder: Path, allure_results_folder: Path):
    print(f"Generating Allure results from: {pipeline_folder}")
    pipeline_result = load_all_jobs_as_pipeline(pipeline_folder)
    write_allure_results(pipeline_result, allure_results_folder)
    print(f"Allure results written to: {allure_results_folder}")

if __name__ == "__main__":
    main()
