#!/usr/bin/env bash

# note: make sure the correct conda environment is active
# `conda activate <name>`

export PYTHONPATH="${PYTHONPATH}/c/Users/Ben/PyCharmProjects/thesis-playground"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/PyCharmProjects/thesis-playground/GraphGym/graphgym"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/PyCharmProjects/thesis-playground/GraphGym/run"

echo "setting PYTHONPATH to ${PYTHONPATH}"

CONFIG=ecoli
GRID=compare-ecoli
REPEAT=3
MAX_JOBS=8
SLEEP=1


echo "generating configs..."
# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/${CONFIG}.yaml \
  --grid grids/${GRID}.txt \
  --out_dir configs
#  --config_budget configs/${CONFIG}.yaml \
# run batch of configs
echo "running configs..."
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP

# aggregate results for the batch
# python agg_batch.py --dir results_modularity-base/${CONFIG}_grid_${GRID}
