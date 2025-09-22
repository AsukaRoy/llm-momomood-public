#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --mem=20G
#SBATCH --output=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/LLM-mistral-Statistics-gpu.%J.out
#SBATCH --error=/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/output/LLM-mistral-Statistics-gpu.%J.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yunhao.yuan@aalto.fi

# batch runs for all modes and features (2 modes x 15 features = 30 runs), only need to specify the model, seed, and prompt style (~15h for one batch run)
# Example usage:
# sbatch run_llm_sft_batch_runs.sh --model mistral-7b --seed 1 --prompt-style Basic
# Basic,Health_Information,Statistics,All
# activate conda environment

module load triton/2024.1-gcc gcc/12.3.0 cuda/12.2.1

module load mamba

module load model-huggingface

echo "Starting batch runs for state transition prediction"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MASTER_PORT=$((29500 + $RANDOM % 1000))

# make sure module is loaded first.
source activate gemma_env

declare -A MODEL_PATHS=(
    ["llama3.1-8b"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3"
    ["qwen3-8b"]="Qwen/Qwen3-8B"
    ['gemma3-4b']="google/gemma-3-4b-it"
)

get_model_path() {
    local model_name=$1
    if [[ -n "${MODEL_PATHS[$model_name]}" ]]; then
        echo "${MODEL_PATHS[$model_name]}"
    else

        echo "$model_name"
    fi
}

##### default values for settings #####
#output_model_path="/m/cs/scratch/networks-nima-mmm2018/yagao/trained_models/state_transition_prediction/finetuning_phq_full"

output_model_path="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/results/model/trained_models/finetuning_phq_full"
learning_rate="1e-3"
model_name="qwen3-8b"
model_path="Qwen/Qwen3-8B"
#seed_variant=2 # 2,3,4,5
prompt_style="Basic" #with_init,with_init_history,Basic,Health_Information,Statistics,All
##### default values for settings #####
seeds=(1 2 3 4 5) # Loop through seeds 2 to 5

#modes=("embedding" "full")
modes=("embedding" "full")
features=("magnitude_max" "battery_afternoon" "screen_off_night_duration" "screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration" "battery_afternoon_magnitude_max" "screen_off_night_duration_magnitude_max" "magnitude_max_screen_use_duration_afternoon" "screen_off_night_duration_screen_use_duration_afternoon"  "battery_afternoon_screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration_magnitude_max" "battery_afternoon_magnitude_max_screen_use_duration_afternoon" "screen_off_night_duration_magnitude_max_screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration_screen_use_duration_afternoon" "battery_afternoon_screen_off_night_duration_magnitude_max_screen_use_duration_afternoon")


while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model_name="$2"
            model_path=$(get_model_path "$2")
            shift 2
            ;;
        --list-models)
            echo "Available model shortcuts:"
            for key in "${!MODEL_PATHS[@]}"; do
                echo "  $key -> ${MODEL_PATHS[$key]}"
            done
            exit 0
            ;;
        --seed)
            seed_variant="$2"
            shift 2
            ;;
        --prompt-style)
            prompt_style="$2"
            shift 2
            ;;
        --list-prompt-styles)
            echo "Available prompt styles:"
            echo "  with_init"
            echo "  with_init_history"
            echo "  Basic"
            echo "  Health_Informtion"
            echo "  Statistics"
            echo "  All"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model NAME/PATH   Model name or path (use --list-models to see available shortcuts)"
            echo "  --list-models       Show available model shortcuts"
            echo "  --seed NUMBER       Seed variant (default: 1)"
            echo "  --prompt-style STRING prompt styles (default: Basic)"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

#output_dir="/m/cs/scratch/networks-nima-mmm2018/yagao/result/scores/state_transition_prediction/finetuning/new_experiments/${model_name}/seed_variant_${seed_variant}/${prompt_style}"


for seed_variant in "${seeds[@]}"; do
    #output_dir="/m/cs/scratch/networks-nima-mmm2018/yagao/result/scores/state_transition_prediction/finetuning/new_experiments/${model_name}/seed_variant_${seed_variant}/${prompt_style}"

output_dir="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/results/scores/${model_name}/seed_variant_${seed_variant}/${prompt_style}"


    mkdir -p "$output_dir"
    echo "Running with seed variant: $seed_variant"
    for feature in "${features[@]}"; do
        data_path="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/data/revision_prompt/state_transition_sequences_${feature}_prompts_3.csv"
        if [[ ! -f "$data_path" ]]; then
            echo "Warning: Data file not found: $data_path"
            continue
        fi
        for mode in "${modes[@]}"; do
            output_result_path="$output_dir"
            echo "Running with feature: $feature, mode: $mode"
            torchrun --master_port=$MASTER_PORT /m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/llm_momomood/fine_tuning.py \
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


