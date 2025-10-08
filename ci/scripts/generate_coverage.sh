#!/bin/bash

set -e
set -u
set -o pipefail
LOG_FILE="coverage_generation.log"
rm -f "$LOG_FILE"

log() {
  echo ""
  echo "--------------------------------------------------"
  echo "➡️  $1"
  echo "--------------------------------------------------"
}

handle_error() {
  local exit_code=$?
  echo ""
  echo "❌ ERROR: A command failed on line $BASH_LINENO with exit code $exit_code."
  echo "-------------------- Full Log --------------------"
  cat "$LOG_FILE"
  echo "--------------------------------------------------"
  exit $exit_code
}

trap handle_error ERR


LCOV_IGNORE_OPTS="--ignore-errors mismatch,inconsistent"
LCOV_RC_OPTS="--rc geninfo_unexecuted_blocks=1 --rc branch_coverage=1"


log "Capturing baseline and test data..."
(
  lcov --capture --initial --directory build --output-file baseline.info $LCOV_IGNORE_OPTS $LCOV_RC_OPTS
  lcov --capture --directory build --output-file tests.info $LCOV_IGNORE_OPTS $LCOV_RC_OPTS
) >> "$LOG_FILE" 2>&1

log "Combining tracefiles..."
lcov --add-tracefile baseline.info --add-tracefile tests.info \
     --output-file full_report.info $LCOV_RC_OPTS $LCOV_IGNORE_OPTS >> "$LOG_FILE" 2>&1

log "Extracting main source code..."
lcov --extract full_report.info "$CI_PROJECT_DIR/src/*" \
     --output-file final_report.info $LCOV_IGNORE_OPTS $LCOV_IGNORE_OPTS >> "$LOG_FILE" 2>&1


log "Generating HTML report..."
genhtml final_report.info \
        --output-directory coverage_report \
        --prefix "$CI_PROJECT_DIR" \
        --ignore-errors source \
        --rc genhtml_branch_coverage=1 $LCOV_IGNORE_OPTS >> "$LOG_FILE" 2>&1


log "Cleaning up intermediate files..."
rm -f baseline.info tests.info full_report.info src_only.info final_report.info

log "Coverage report generated successfully in 'coverage_report'."
