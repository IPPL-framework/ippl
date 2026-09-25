#!/bin/bash
# -----------------------------------------------------------------------------
# Install the bencher.dev CLI.
#
# Uses the official install script. Set BENCHER_VERSION to pin a version;
# otherwise the latest release is installed.
# -----------------------------------------------------------------------------

set -euo pipefail

curl --proto '=https' --tlsv1.2 -sSfL https://bencher.dev/download/install-cli.sh | sh

bencher --version
