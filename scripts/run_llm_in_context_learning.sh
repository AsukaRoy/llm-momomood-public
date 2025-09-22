#!/bin/bash
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --output=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/mistral_few_in_context_learning.%J.out
#SBATCH --error=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/mistral_few_in_context_learning.%J.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yunhao.yuan@aalto.fi

module load triton/2024.1-gcc gcc/12.3.0 cuda/12.2.1

module load mamba

module load model-huggingface

echo "Starting batch runs for state transition prediction"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MASTER_PORT=$((29500 + $RANDOM % 1000))

# make sure module is loaded first.
source activate gemma_env
# sbatch run_llama3_in_context_learning.sh
# Basic,Health_Information,Statistics,All

# Models to loop over
declare -A MODEL_PATHS=(
    #["llama3.1-8b"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3"
    #["qwen3-8b"]="Qwen/Qwen3-8B"
    #['gemma3-4b']="google/gemma-3-4b-it"
)


# Where to put run outputs (parent)
base_output_result_path="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/results/in_context_eval"
# Your existing settings
prompt_styles=("Basic" "Health_Information" "Statistics" "All") #with_init,with_init_history,      Basic,Health_Information,Statistics,All
#prompt_styles=("Basic")
features=("magnitude_max" "battery_afternoon" "screen_off_night_duration" "screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration" "battery_afternoon_magnitude_max" "screen_off_night_duration_magnitude_max" "magnitude_max_screen_use_duration_afternoon" "screen_off_night_duration_screen_use_duration_afternoon"  "battery_afternoon_screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration_magnitude_max" "battery_afternoon_magnitude_max_screen_use_duration_afternoon" "screen_off_night_duration_magnitude_max_screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration_screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration_magnitude_max_screen_use_duration_afternoon")

n_shots_list='[0,1,3,5,10,15,25]'


# Five seeds to loop (pick what you like)
seeds=(1 2 3 4 5)

for model_key in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[$model_key]}"
  for prompt_style in "${prompt_styles[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "seed"
      # Separate output dir per model/seed/prompt_style to avoid collisions
      output_result_path="${base_output_result_path}/${model_key}/seed_variant_${seed}/${prompt_style}"
      mkdir -p "${output_result_path}"
      # Optional: make PyTorch/cuBLAS more deterministic
      #export PYTHONHASHSEED="${seed}"
      #export CUBLAS_WORKSPACE_CONFIG=":4096:8"

      for feature in "${features[@]}"; do
      
        data_path="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/data/revision_prompt/state_transition_sequences_${feature}_prompts_3.csv"

        echo "=== model=${model_key} | seed=${seed} | prompt_style=${prompt_style} | feature=${feature} ==="
        torchrun --master_port=$MASTER_PORT /m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/n_shot_llama3.py \
          --data_path "${data_path}" \
          --model_path "${model_path}" \
          --output_result_path "${output_result_path}" \
          --feature "${feature}" \
          --prompt_style "${prompt_style}" \
          --n_shots_list "${n_shots_list}" \
          --seed "${seed}"
      done
    done
  done
done


conda deactivate
