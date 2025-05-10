#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --output=/m/cs/scratch/networks-nima-mmm2018/yagao/output/state_transition_prediction/finetuning/LLM-MM-gpu.%J.out
#SBATCH --error=/m/cs/scratch/networks-nima-mmm2018/yagao/output/state_transition_prediction/finetuning/LLM-MM-gpu.%J.err

export MASTER_PORT=0

# activate conda environment
module load mamba

# make sure module is loaded first.
source activate /m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/llama3-1-env

model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"

learning_rate="1e-3"

seed_variant=5

output_model_path="/m/cs/scratch/networks-nima-mmm2018/yagao/trained_models/state_transition_prediction/finetuning_phq_full"
output_result_path="/m/cs/scratch/networks-nima-mmm2018/yagao/result/scores/state_transition_prediction/finetuning"

modes=("full" "embedding")
prompt_styles=("with_init" "with_init_history")

features=("magnitude_max" "battery_avg_afternoon" "screen_off_night" "screen_use_afternoon" "magnitude_max+battery_avg_afternoon+screen_off_night+screen_use_afternoon","magnitude_max+battery_avg_afternoon" "screen_off_night+screen_use_afternoon" "magnitude_max+screen_off_night" "magnitude_max+screen_use_afternoon" "battery_avg_afternoon+screen_off_night" "battery_avg_afternoon+screen_use_afternoon","magnitude_max+battery_avg_afternoon+screen_off_night" "magnitude_max+battery_avg_afternoon+screen_use_afternoon" "magnitude_max+screen_off_night+screen_use_afternoon" "battery_avg_afternoon+screen_off_night+screen_use_afternoon")


for feature in "${features[@]}"; do
    data_path="/m/cs/scratch/networks-nima-mmm2018/yagao/data/state_transition_prediction/feature_selection/state_transition_sequences_${feature}_prompts_3.csv"
    for prompt_style in "${prompt_styles[@]}"; do
        for mode in "${modes[@]}"; do
            torchrun ../llm-momomood/state_transition_prediction/fine_tuning.py \
            --data_path $data_path \
            --model_path $model_path \
            --output_result_path $output_result_path \
            --output_model_dir $output_model_path \
            --learning_rate $learning_rate \
            --feature $feature \
            --mode $mode \
            --prompt_style $prompt_style \
            --seed_variant $seed_variant
        done
    done

done


conda deactivate
