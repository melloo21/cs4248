#!/bin/bash

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.1-0.1.json" -s "./experiments-_0.1_0.1" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.1-0.2.json" -s "./experiments-_0.1_0.2" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.1-0.3.json" -s "./experiments-_0.1_0.3" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.2-0.2.json" -s "./experiments-_0.2_0.2" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.3-0.3.json" -s "./experiments-_0.3_0.3" --include-package scicite
