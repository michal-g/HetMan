#!/bin/bash

source activate HetMan
cd ~/compbio/scripts
python HetMan/experiments/drug_predictions/drug_predict2.py "$@"

