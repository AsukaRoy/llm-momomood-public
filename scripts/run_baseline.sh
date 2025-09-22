#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --output=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/baseline_model-MM-gpu.%J.out
#SBATCH --error=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/baseline_model-MM-gpu.%J.err
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G


# activate conda environment
module load mamba
# module load miniconda
# make sure module is loaded first.
source activate appreview

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

# run batch inference with parameters
srun python3 /m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/llm_momomood/baseline_model.py


echo "Hello $USER! You are on node $HOSTNAME. post-control-1 finished."



