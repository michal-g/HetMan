#!/bin/bash

source activate HetMan
cd ~/compbio/scripts/HetMan
python experiments/baseline_test.py "$@"

