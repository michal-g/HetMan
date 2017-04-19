#!/bin/bash

source activate HetMan
cd ~/compbio/scripts
python HetMan/experiments/features/features.py "$@"

