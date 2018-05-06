#!/usr/local/bin/bash
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
#$ -N scaleBy_01
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-8.0
export LIBRARY_PATH=$LIBRARY_PATH:${CUDA_HOME}/lib64 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64 
export CPATH=${CUDA_HOME}/include:${CPATH} 
export PATH=${CUDA_HOME}/bin:${PATH} 
export THEANO_FLAGS=device=cuda,floatX=float32,optimizer_including=cudnn,cuda.root=${CUDA_HOME}
source ~/.python/3.5rlab/bin/activate
cd /home/kdu3/rl/baselines/

# IND=$SGE_TASK_ID
# mkdir amidarScale_01DuelingPrioritized
python -m baselines.deepq.experiments.atari.train --env Amidar --dueling --prioritized --save-dir amidarScale_01DuelingPrioritized --scale-by 0.01 --load-on-start
# $IND
