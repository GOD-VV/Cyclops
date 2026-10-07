#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python examples/training/train.py examples/training/config/intensity_phase1.yaml
test -f outputs/phase1/last.ckpt
python examples/training/train.py examples/training/config/intensity_phase2.yaml
