# <p align=center>`Citation Intent Classification`</p> 

This repository contains datasets and code for running experiments on the Scicite model architecture as well as data for the purpose of classifying citation intents. 

The original code was obtained from the paper: 
["Structural Scaffolds for Citation Intent Classification in Scientific Publications"](https://arxiv.org/pdf/1904.01608.pdf).

## Experiments

The experiments can be replicated by running the following code:

Hyperparameter search for $\lambda_1$ and $\lambda_2$

`cd scaffold/Project/sci-cite/scicite/`

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.1-0.1.json" -s "./experiments-_0.1_0.1" --include-package scicite
`

Change mixing_ratio for $\lambda_1$ and mixing ratio for $\lambda_2$ in json file.

Experiment 1: Baseline

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05.json" -s ".runs/experiments-_0.05-0.05" --include-package scicite
`

Experiment 2: Convert numbers to tokens

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05.json" -s ".runs/experiments-_0.05-0.05" --include-package scicite
`

In `scaffold/Project/sci-cite/scicite/scicite/dataset_readers/citation_data_reader_scicite.py `, change self.convert_num = True

Experiment 3: Remove citations using citestart/end

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05.json" -s "./experiments-_0.05-0.05" --include-package scicite
`
In `scaffold/Project/sci-cite/scicite/scicite/dataset_readers/citation_data_reader_scicite.py `, change self.convert_cite_to_token = True

Experiment 4: Using Word2Vec + ELMO

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-w2v.json" -s "./runs/experiment-0.05-0.05-w2v-1" --include-package scicite
`

Experiment 5: Using LSTM instead of GRU 

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-elmo-lstm.json" -s "./runs/experiment-0.05-0.05-elmo-lstm" --include-package scicite`

Experiment 6: Removing Bi-directionality

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-elmo-forwardgru.json" -s "./runs/experiment-0.05-0.05-elmo-forwardgru" --include-package scicite
`
Experiment 7: Removing ELMo

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-no-elmo.json" -s "./runs/experiment-0.05-0.05-no-elmo" --include-package scicite
`

Experiment 8: Using deduplication 

`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-deduplicate.json" -s "./runs/experiment-0.05-0.05-deduplicate" --include-package scicite
`

Experiment 9: Remove Stopwords
`python scripts/train_local.py train_multitask_2 "./experiment_configs/scicite-experiment-0.05-0.05-removestop.json" -s "./runs/experiment-0.05-0.05-removestop" --include-package scicite`

In `scaffold/Project/sci-cite/scicite/scicite/dataset_readers/citation_data_reader_scicite.py `, change self.remove_stopwords = True

## Setup

The project needs Python 3.6 and is based on the [AllenNLP](https://github.com/allenai/allennlp) library.

#### Setup an environment manually

Use pip to install dependencies in your desired python environment

`pip install -r requirements.in -c constraints.txt`


## Running a pre-trained model on your own data

Download one of the pre-trained models and run the following command:

```bash
allennlp predict [path-to-model.tar.gz] [path-to-data.jsonl] \
--predictor [predictor-type] \
--include-package scicite \
--overrides "{'model':{'data_format':''}}"
```

Where 
* `[path-to-data.jsonl]` contains the data in the same format as the training data.
* `[path-to-model.tar.gz]` is the path to the pretrained model
* `[predictor-type]` is one of `predictor_scicite` (for the SciCite dataset format) or `predictor_aclarc` (for the ACL-ARC dataset format).
* `--output-file [out-path.jsonl]` is an optional argument showing the path to the output. If you don't pass this, the output will be printed in the stdout.

If you are using your own data, you need to first convert your data to be according to the SciCite data format.

#### Pretrained models

We also release our pretrained models; download from the following path:

* __[`SciCite`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/scicite.tar.gz)__
* __[`ACL-ARC`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/aclarc.tar.gz)__

## Training your own models

First you need a `config` file for your training configuration.
Check the `experiment_configs/` directory for example configurations.
Important options (you can specify them with environment variables) are:

```
  "train_data_path":  # path to training data,
  "validation_data_path":  #path to development data,
  "test_data_path":  # path to test data,
  "train_data_path_aux": # path to the data for section title scaffold,
  "train_data_path_aux2": # path to the data for citation worthiness scaffold,
  "mixing_ratio": # parameter \lambda_2 in the paper (sensitivity of loss to the first scaffold)
  "mixing_ratio2": # parameter \lambda_3 in the paper (sensitivity of loss to the second scaffold)
``` 

```
python scripts/train_local.py train_multitask_2 [path-to-config-file.json] \
-s [path-to-serialization-dir/] 
--include-package scicite
```

Where the model output and logs will be stored in `[path-to-serialization-dir/]`


