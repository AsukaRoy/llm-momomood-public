#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --time=30:00:00
#SBATCH --mem=80G
#SBATCH --output=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/llama3_in_context_learning.%J.out
#SBATCH --error=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/llama3_in_context_learning.%J.err


# activate conda environment
module load mamba

# make sure module is loaded first.
source activate /m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/llama3-1-env

model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"

output_result_path="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/results/scores/in_context_learning_rs_210"

prompt_styles=("with_init_history" "with_init")

n_shots_list='[0,1,3,5,10,15,25]'


features=("magnitude_max" "battery_avg_afternoon" "screen_off_night" "screen_use_afternoon" "magnitude_max+battery_avg_afternoon+screen_off_night+screen_use_afternoon","magnitude_max+battery_avg_afternoon" "screen_off_night+screen_use_afternoon" "magnitude_max+screen_off_night" "magnitude_max+screen_use_afternoon" "battery_avg_afternoon+screen_off_night" "battery_avg_afternoon+screen_use_afternoon","magnitude_max+battery_avg_afternoon+screen_off_night" "magnitude_max+battery_avg_afternoon+screen_use_afternoon" "magnitude_max+screen_off_night+screen_use_afternoon" "battery_avg_afternoon+screen_off_night+screen_use_afternoon")

for feature in "${features[@]}"; do
    data_path="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/data/feature_selection/state_transition_sequences_${feature}_prompts_3.csv"
    for prompt_style in "${prompt_styles[@]}"; do
        torchrun ../llm-momomood/n_shot_llama3.py \
        --data_path $data_path \
        --model_path $model_path \
        --output_result_path $output_result_path \
        --feature $feature \
        --prompt_style $prompt_style \
        --n_shots_list $n_shots_list
    done
done

conda deactivate
