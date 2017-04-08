#!/bin/bash

source activate HetMan
cd ~/compbio/scripts/HetMan
python experiments/upstream_mutex.py "$@"

