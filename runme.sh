#!/usr/bin/env bash
#
#  Execute from the current working directory
#$ -cwd
#
#  This is a gpu job
#$ -l gpus=1
#
#  Can use up to 64GB of memory
#$ -l vf=64G
#
source ~/.python/3.5rlab/bin/activate
cd /home/kdu3/rl/baselines/

# IND=$SGE_TASK_ID
mkdir amidarScale10DuelingPrioritized
python -m baselines.deepq.experiments.atari.train --env Amidar --dueling --prioritized --save-dir amidarScale10DuelingPrioritized --scale-by 10
# $IND