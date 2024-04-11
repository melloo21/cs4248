#!/bin/bash

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.1-0.1.json" -s "./experiments-_0.1_0.1" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.1-0.2.json" -s "./experiments-_0.1_0.2" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.1-0.3.json" -s "./experiments-_0.1_0.3" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.2-0.2.json" -s "./experiments-_0.2_0.2" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.3-0.3.json" -s "./experiments-_0.3_0.3" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-w2v.json" -s "./runs/experiment-0.05-0.05-w2v-1" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-no-elmo.json" -s "./runs/experiment-0.05-0.05-no-elmo" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-elmo-lstm.json" -s "./runs/experiment-0.05-0.05-elmo-lstm" --include-package scicite



python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-elmo-forwardgru.json" -s "./runs/experiment-0.05-0.05-elmo-forwardgru" --include-package scicite


python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-elmo-forwardgru-noattention.json" -s "./runs/experiment-0.05-0.05-elmo-forwardgru-noattention" --include-package scicite


python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-numtokens.json" -s "./runs/experiment-0.05-0.05-numtokens2" --include-package scicite


python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-citetokens.json" -s "./runs/experiment-0.05-0.05-citetokens" --include-package scicite

python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-removecite.json" -s "./runs/experiment-0.05-0.05-removecite" --include-package scicite


python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-bertembeddings.json" -s "./runs/experiment-0.05-0.05-bertembeddings" --include-package scicite
