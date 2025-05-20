# Algorithmic Monoculture and Multiplicity


## Initial steps to reproduce
- create a virtual environment running: `conda env create -f environment.yml`
    - this will install folktexts and tableshift
    - TODO: how to handle forks?
- clone the folktexts package (see [instructions](https://github.com/socialfoundations/folktexts))
- run `python scripts/load_datasets.py` to load all datasets (TODO: from where)
    - note: for BRFSS datasets preprocessor is set to passthrough, s.t. no normalization of numeric features or one-hot-encoding of categorical features is done

## Getting Model Predictions
From inside the folktexts package, run 
```
# 0-shot 
python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir '<path/to/results/folder/>' --task=<task>> --models-dir '<path/to/models/>' --model=google/gemma-2b --fit-threshold=2000 --variation="format=bullet;connector=is;granularity=original;order=AGEP,COW,SCHL,MAR,OCCP,POBP,RELP,WKHP,SEX,RAC1P"

# 10-shot
python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir '<path/to/results/folder/>' --task=<task>> --models-dir '<path/to/models/>' --model=google/gemma-2b --fit-threshold=2000 --reuse-few-shot-examples=True --few-shot=10 --balance-few-shot-examples=True --variation="format=bullet;connector=is;granularity=original;order=AGEP,COW,SCHL,MAR,OCCP,POBP,RELP,WKHP,SEX,RAC1P"
```

See (https://github.com/socialfoundations/folktexts)
## Baseline Predictions
From inside the monoculture package, run
```
python -m  monoculture.baseline.run_acs_benchmark_baseline --model Constant --results-dir '<path/to/results/folder/>' --data-dir '../folktexts-adapted/data/' --task <task>
```


## Analysis model prediction for monoculture and multiplicity 
- metrics are collected in `metrics.py`
- for plotting 


## Available predictions 
TODO: make model predictions on ACSIncome available 




 <!-- Old below -->

## Loading Tableshift Tasks

- clone TableShift fork: https://github.com/milagorecki/tableshift
- create a virtual environment running: `conda env create -f environment.yml`
- run `python scripts/load_datasets.py` to save the full BRFSS datasets <br> 
    (note: preprocessor is set to passthrough, s.t. no normalization of numeric features or one-hot-encoding of categorical features is done)

## Installing Folktexts

- in the above virtual environment, install folktexts using this fork: https://github.com/milagorecki/folktexts/tree/eval-fairness

### Getting Model Predictions
- from the folktexts package, run 
```
# ACS
python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir '/fast/mgorecki/monoculture/results/' --task ACSIncome style='format=bullet,connector=is' 
# Tableshift
python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir '/fast/mgorecki/monoculture/results/' --task BRFSS_Blood_Pressure style='format=bullet,connector=is'
```

```
python -m folktexts.cli.launch_experiments_htcondor --executable-path <path to benchmark python script> --results-dir <path to results dir>
```
- `--executable-path ./folktexts/cli/run_acs_benchmark.py` to run ACS Tasks, `--executable-path ./folktexts/cli/run_tableshift_benchmark.py` to run tableshift tasks
- `--results-dir` directory where results will be saved
- `--task ACSIncome` specificy task(s)
- `--dryrun` to check which jobs will be started without actually starting them 
- `--style='format=bullet,connector=is'` optionally adapt style of the prompt


To locally test, if code is running, just run the benchmark directly
```
python -m folktexts.cli.run_benchmark --model gpt2 --results-dir './results/test/' --data-dir './data' --task BRFSS_Diabetes --subsampling 0.01 --style "format=bullet,connector=is"
```

TODO: check how style is passed

## Baseline Predictions

```
python -m  monoculture.baseline.run_acs_benchmark_baseline --model Constant --results-dir 'results/baselines/' --data-dir '../folktexts-adapted/data/' --task BRFSS_Diabetes
```

## How to create figures

