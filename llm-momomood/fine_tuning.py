# -*- coding: utf-8 -*-
import os
import ast
import random
import glob
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["BNB_CUDA_VERSION"] = "118"
#os.environ['HF_HOME']='/m/cs/scratch/networks-nima-mmm2018/yunhao/model/'
print(f'HF home directory is {os.environ['HF_HOME']}')
#os.environ['HF_HOME']='/scratch/shareddata/dldata/huggingface-hub-cache/'
import logging
from typing import List, Tuple, Dict
from datetime import datetime
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer, setup_chat_format, SFTConfig

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          DataCollatorForSeq2Seq)
from transformers import AutoConfig
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import resample
import bitsandbytes as bnb
import torch
import torch.nn as nn

from tqdm.auto import tqdm
import argparse
from datasets import load_dataset
import fire
import json
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))


def set_global_seed(seed_variant):
    random.seed(42 * seed_variant)
    np.random.seed(42 * seed_variant)
    torch.manual_seed(42 * seed_variant)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

SYSTEM_PROMPT = """You are an intelligent healthcare agent skilled in predicting changes in mental health conditions.
TASK: Your task is to determine how the user's mental health state changes based on digital data of mobile sensor readings from an individual.
INPUT: The input contains the following contents: data about mobile phone usage over the previous two-week; whether being depressed or not over the previous two-week; data about mobile phone usage over the current two-week.
OUTPUT: To determine how the user's mental health state changes, you should only respond with "Label: Remains" or "Label: More Depressed" or "Label: Less Depressed". Make sure to only return the label and nothing more.
"""



def construct_prompt(user_data, n_shot_examples=None):
    """Constructs a Mistral-compatible training prompt without system role."""
    messages = []
    
    # Add system prompt as first user message if provided
    messages.append({"role": "user", "content": f"Instructions: {SYSTEM_PROMPT}\n\nPlease follow these instructions."})
    messages.append({"role": "assistant", "content": "I understand."})
    
    # Add few-shot examples
    if n_shot_examples:
        for example in n_shot_examples:
            messages.append({"role": "user", "content": example['data']})
            messages.append({"role": "assistant", "content": f'Label: {example["label"]}\n'})
    
    # Add current data point with its label (for training)
    messages.append({"role": "user", "content": user_data['data']})
    messages.append({"role": "assistant", "content": f'Label: {user_data["label"]}\n'})
    
    return messages


def construct_test_prompt(user_data, n_shot_examples=None):
    """Constructs a Mistral-compatible test prompt without system role."""
    messages = []
    
    # Add system prompt as first user message if provided

    messages.append({"role": "user", "content": f"Instructions: {SYSTEM_PROMPT}\n\nPlease follow these instructions."})
    messages.append({"role": "assistant", "content": "I understand."})

    # Add few-shot examples
    if n_shot_examples:
        for example in n_shot_examples:
            messages.append({"role": "user", "content": example['data']})
            messages.append({"role": "assistant", "content": f'Label: {example["label"]}\n'})
    
    # Add current data point without label (for inference)
    messages.append({"role": "user", "content": user_data['data']})
    
    return messages


def select_n_shot_examples(df, label, original_example, n=0):
    """Select n-shot examples from the dataframe based on the label."""
    filtered_df = df[(df['previous depression state'] == label) & (df['data'] != original_example['data'])]
    examples = filtered_df.sample(n=n, replace=False).to_dict('records')
    return examples


def load_data(filename: str, prompt_style: str, seed_variant: int, n_shot=0) -> pd.DataFrame:
    if prompt_style == 'original':
        data = 'data'
    elif prompt_style == 'with_init':
        data = 'data_with_init'
    elif prompt_style == 'with_history':
        data = 'data_with_history'
    elif prompt_style == 'with_init_history':
        data = 'data_with_init_his'
    elif prompt_style == 'Basic':
        data = "Basic"
    elif prompt_style == 'Health_Information':
        data = "Health_Informtion"
    elif prompt_style == 'Statistics':
        data = "Statistics"
    elif prompt_style == 'All':
        data = "All"


    logging.info(f"load the datasets")
    df =  pd.read_csv(filename)

    X_train, X_test = split_data(df, subject_column = data, label_column = 'label', seed_variant=seed_variant, oversample=True) # return with column 'data'

    y_true_text_label = X_test.label

    # Convert the training and eval datasets to prompts
    X_train_prompt = pd.DataFrame(X_train.apply(lambda x: construct_prompt(x, select_n_shot_examples(X_train, x['previous depression state'], x, n_shot)), axis=1), columns=['data'])
    # Convert the test dataset to prompts without the label
    X_test_prompt = pd.DataFrame(X_test.apply(lambda x: construct_test_prompt(x, select_n_shot_examples(X_train, x['previous depression state'], x, n_shot)), axis=1), columns=['data'])

    return X_train_prompt, X_test_prompt, y_true_text_label

def print_gpu_info() -> None:
    """
    Print GPU information if CUDA is available.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')


def evaluate_predictions(y_true, y_preds):
    """Evaluates classification performance and returns a report with various metrics."""
    target_names = ['Remains', 'More Depressed', 'Less Depressed']
    report = classification_report(y_true, y_preds, zero_division=0, target_names =target_names, labels=[0,1,2])
    conf_matrix = confusion_matrix(y_true, y_preds).tolist()
    acc = accuracy_score(y_true, y_preds)
    macro_f1 = f1_score(y_true, y_preds, average='macro')
    weighted_f1 = f1_score(y_true, y_preds, average='weighted')
    macro_precision = precision_score(y_true, y_preds, average='macro')
    weighted_precision = precision_score(y_true, y_preds, average='weighted')
    macro_recall = recall_score(y_true, y_preds, average='macro')
    weighted_recall = recall_score(y_true, y_preds, average='weighted')

    print(report)
    print('Accuracy: ', acc)
    print('Macro Precision: ', macro_precision)
    print('Weighted Precision: ', weighted_precision)
    print('Macro Recall: ', macro_recall)
    print('Weighted Recall: ', weighted_recall)
    print('Macro F1: ', macro_f1)
    print('Weighted F1: ', weighted_f1)
    print('Confusion matrix:', conf_matrix)

    return {
        "Accuracy": acc,
        "Macro Precision": macro_precision,
        "Weighted Precision": weighted_precision,
        "Macro Recall": macro_recall,
        "Weighted Recall": weighted_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Confusion Matrix": conf_matrix,  # For JSON compatibility
        "Classification Report": report
    }

def process_prediction(prediction, prompt):
    """Extracts the prediction from the model output and identifies binary classification result."""
    output_text = prediction[0]['generated_text'][len(prompt):].strip()
    if "Label: Remains" in output_text:
        return 0
    elif "Label: More Depressed" in output_text:
        return 1
    elif "Label: Less Depressed" in output_text:
        return 2
    else:
        return 0  # Indicates an unknown or unexpected response


def predict(test, model, tokenizer, y_true_text_label):

        # Fix dtype mismatch for Qwen3 models
    #if hasattr(model, 'lm_head') and model.lm_head.weight.dtype != torch.bfloat16:
    #    model.lm_head = model.lm_head.to(torch.bfloat16)


    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16
    )
    
    # Gemma 3 specific EOS tokens - adjust based on actual model configuration
    terminators = [
        text_pipeline.tokenizer.eos_token_id,
    ]
    
    # Try to get Gemma 3 specific tokens if they exist
    if hasattr(text_pipeline.tokenizer, 'convert_tokens_to_ids'):
        try:
            # Common Gemma end tokens - adjust based on actual tokenizer
            end_tokens = ["<end_of_turn>", "<|end_of_text|>", "</s>"]
            for token in end_tokens:
                token_id = text_pipeline.tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id != text_pipeline.tokenizer.unk_token_id:
                    terminators.append(token_id)
        except:
            pass
    
    y_true, y_preds = [], []
    for index, row in test.iterrows():
        messages = row['data']
        prompt = text_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                #enable_thinking=False
        )
        print('-----------------------------------------------------------')
        print(prompt)
        generation = text_pipeline(
            prompt,
            max_new_tokens=16,
            eos_token_id=terminators,
            pad_token_id=text_pipeline.tokenizer.eos_token_id
        )

        category_result = process_prediction(generation, prompt)
        test.loc[index, 'generated_text'] = generation[0]['generated_text']
        test.loc[index, 'prediction'] = category_result


        print("LLM answer: {} True Label: {} \n".format(generation[0]['generated_text'][len(prompt):], y_true_text_label[index]))
        print('-----------------------------------------------------------')

        if y_true_text_label[index] == 'Remains':
            true_label = 0
        elif y_true_text_label[index] == 'More Depressed':
            true_label = 1
        elif y_true_text_label[index] == 'Less Depressed':
            true_label = 2
        y_true.append(true_label)
        y_preds.append(category_result)


    return y_preds, y_true

def split_data(df, subject_column, label_column, seed_variant, oversample = False):
    """
    Splits data into training, testing, and evaluation sets while maintaining the class distribution and subject-level consistency.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data and labels.
    - subject_column (str): The column name representing subjects.
    - label_column (str): The column name representing the labels.

    Returns:
    - X_train (DataFrame): Training data subset.
    - X_test (DataFrame): Testing data subset.
    """
    # Split subjects into training and test
    gss = GroupShuffleSplit(test_size=0.4, n_splits=1, random_state=1*seed_variant)
    train_idx, test_idx = next(gss.split(df, groups=df[subject_column]))
    X_train = df.iloc[train_idx]
    X_test = df.iloc[test_idx]
    print(f"Initial split: {len(X_train)} training samples, {len(X_test)} temporary samples.")
    if(oversample):
        # Oversample the underrepresented classes in the training set
        train_labels = X_train[label_column].value_counts()
        n_resample = int(train_labels.mean())
        for label, count in train_labels.items():
            if count < train_labels.mean():
                X_train = pd.concat([X_train, resample(X_train[X_train[label_column] == label],
                                                      replace=True,
                                                      n_samples=n_resample,
                                                      random_state=1)],
                                    ignore_index=True)
        print("Oversampling performed on training data.")
    print(f"Final split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    X_train_final = X_train[[subject_column,'previous depression state', 'label']]
    X_test_final = X_test[[subject_column,'previous depression state','label']]
    # change the column name to 'data' for the subject_column
    X_train_final.rename(columns={subject_column: 'data'}, inplace=True)
    X_test_final.rename(columns={subject_column: 'data'}, inplace=True)
    return X_train_final, X_test_final



def setup_trainer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                      train_data: Dataset, output_model_dir: str, learning_rate: str, mode: str, seed_variant: int) -> SFTTrainer:
    # Implement the trainer setup logic here
    # pass
    """
    Set up the SFTTrainer.

    Args:
        model (AutoModelForCausalLM): The model to train.
        tokenizer (AutoTokenizer): The tokenizer.
        train_data (Dataset): Training dataset.

    Returns:
        SFTTrainer: Configured trainer.
    """

    trainer = None

    if mode == 'embedding':
        training_arguments = SFTConfig(
            output_dir=output_model_dir,                    # directory to save and repository id
            num_train_epochs=1,                       # number of training epochs
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,            # number of steps before performing a backward/update pass
            gradient_checkpointing=False,             # use gradient checkpointing to save memory
            optim="adamw_torch",
            save_steps=0,
            logging_steps=10,                         # log every 10 steps
            learning_rate=float(learning_rate),
            weight_decay=0.001,
            fp16=False,
            bf16=True,  # Enable bfloat16 for consistency
            max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
            group_by_length=False,
            lr_scheduler_type="cosine",               # use cosine learning rate scheduler
            save_strategy="no",
            seed=42*seed_variant,
            dataset_text_field="data",
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
            packing=False,
        )


        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_data,
        )

    elif mode == 'full':
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
        )
        training_arguments = SFTConfig(
            output_dir=output_model_dir,                    # directory to save and repository id
            num_train_epochs=1,                       # number of training epochs
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,            # number of steps before performing a backward/update pass
            gradient_checkpointing=False,             # use gradient checkpointing to save memory
            optim="paged_adamw_32bit",
            save_steps=0,
            logging_steps=10,                         # log every 10 steps
            learning_rate=float(learning_rate),
            weight_decay=0.001,
            fp16=False,
            bf16=True,  # Enable bfloat16 for consistency
            max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
            group_by_length=False,
            lr_scheduler_type="cosine",               # use cosine learning rate scheduler
            save_strategy="no",
            seed=42*seed_variant,
            dataset_text_field="data",
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
        )
        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_data,
            peft_config=peft_config,
        )

    return trainer

def setup_model_and_tokenizer(model_path: str, mode: str):
    # Implement the model and tokenizer setup logic here
    # pass
    """
    Set up the model and tokenizer.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Configured model and tokenizer.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    logging.info(f"HF_HOME {os.environ['HF_HOME']} ends")

    if mode == 'full':
        print(f"Model path: {model_path}")
        compute_dtype = getattr(torch, "bfloat16")
        # Special handling for Qwen3 models
        if 'Qwen3' in model_path or 'qwen3' in model_path.lower():
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_storage=compute_dtype,  # Add this for Qwen3
            )
            
            # Load with specific config for Qwen3
            config = AutoConfig.from_pretrained(model_path)
            config.pretraining_tp = 1  # Important for Qwen3
            if hasattr(config, 'tensor_parallel_size'):
                config.tensor_parallel_size = 1
                
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=device,
                torch_dtype=compute_dtype,
                quantization_config=bnb_config
            )
            if hasattr(model, "lm_head"):
                # Keep lm_head in FP32 (matches many Qwen3 execution paths)
                model.lm_head = model.lm_head.to(torch.float32)
        
                # Safety hook: cast hidden_states to lm_head.weight.dtype at runtime
                def _align_dtype_pre_hook(module, inputs):
                    (hidden_states,) = inputs
                    if hidden_states.dtype != module.weight.dtype:
                        hidden_states = hidden_states.to(module.weight.dtype)
                    return (hidden_states,)
            model.lm_head.register_forward_pre_hook(_align_dtype_pre_hook)
        else:
            compute_dtype = getattr(torch, "bfloat16")
            # Original configuration for other models
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=compute_dtype,
                quantization_config=bnb_config,
            )
                # Apply dtype alignment hook to lm_head for all models
            if hasattr(model, "lm_head"):
                def _align_dtype_pre_hook(module, inputs):
                    (hidden_states,) = inputs
                    if hidden_states.dtype != module.weight.dtype:
                        hidden_states = hidden_states.to(module.weight.dtype)
                    return (hidden_states,)
        
            model.lm_head.register_forward_pre_hook(_align_dtype_pre_hook)


    elif mode == 'embedding':
        print(f"Model path: {model_path}")

        compute_dtype = getattr(torch, "bfloat16")
        
        
        config = AutoConfig.from_pretrained(model_path)
        # Disable any tensor parallel related config
        if hasattr(config, 'tensor_parallel_size'):
            config.tensor_parallel_size = 1
        if hasattr(config, 'pretraining_tp'):
            config.pretraining_tp = 1
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=compute_dtype,
            device_map={"": 0},
        )
        
        
        for param in model.parameters():
            param.requires_grad = False
            
        if hasattr(model.model, 'embed_tokens'):
            for param in model.model.embed_tokens.parameters():
                param.requires_grad = True
                print('param.requires_grad = True')
        elif hasattr(model.model, 'embeddings'):
            for param in model.model.embeddings.parameters():
                param.requires_grad = True
                print('param.requires_grad = True')
        elif hasattr(model, 'embed_tokens'):
            for param in model.embed_tokens.parameters():
                param.requires_grad = True
                print('param.requires_grad = True')
        elif hasattr(model.model.language_model, 'embed_tokens'):
            for param in model.model.language_model.embed_tokens.parameters():
                param.requires_grad = True
                print('param.requires_grad = True')
                # freeze all but embeddings
        if('Qwen' in model_path):
            for p in model.parameters(): 
                p.requires_grad = False
            # Qwen3 embeddings live here:
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                for p in model.model.embed_tokens.parameters(): 
                    p.requires_grad = True


    
        # unfreeze embedding layer
        #for param in model.model.embed_tokens.parameters():
        #    param.requires_grad = True
        # unfreeze classification head
        # for param in model.score.parameters():
        #     param.requires_grad = True

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, device_map="auto")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def save_evaluation_results(X_test, evaluation_results, output_result_path, feature, prompt_style, mode):

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    output_file_json = f"{output_result_path}/fine_tuning_{mode}_{feature}_{prompt_style}_{timestamp}.json"

    # Save to JSON for structured data
    with open(output_file_json, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    X_test.to_csv(f"{output_result_path}/fine_tuning_{mode}_{feature}_{prompt_style}_{timestamp}.csv")

def log_time_info(message: str):
    """Logs the current time with a custom message."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{message} at {current_time}")


def check_results_exist(output_result_path, feature, prompt_style, mode):
    """
    Check if results for the given configuration already exist.
    
    Args:
        output_result_path (str): Path to the output directory
        feature (str): Feature name
        prompt_style (str): Prompt style
        mode (str): Training mode
        force_rerun (bool): If True, ignore existing results and rerun
    
    Returns:
        bool: True if results exist and should be skipped, False otherwise
    """
    
    # Check for existing JSON files matching the pattern
    pattern = f"{output_result_path}/fine_tuning_{mode}_{feature}_{prompt_style}_*.json"
    existing_files = glob.glob(pattern)
    
    if existing_files:
        logger.info(f"Found existing results for {mode}_{feature}_{prompt_style}: {len(existing_files)} files")
        logger.info(f"Files found: {[os.path.basename(f) for f in existing_files]}")
        
        # Check if the most recent file is valid (not corrupted)
        most_recent = max(existing_files, key=os.path.getctime)
        try:
            with open(most_recent, 'r') as f:
                data = json.load(f)
                # Basic validation - check if it has expected keys
                if 'Accuracy' in data and 'Macro F1' in data:
                    logger.info(f"Valid results found in {os.path.basename(most_recent)}. Skipping...")
                    return True
                else:
                    logger.warning(f"Existing file {os.path.basename(most_recent)} appears incomplete. Will rerun.")
                    return False
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error reading existing file {os.path.basename(most_recent)}: {e}. Will rerun.")
            return False
    else:
        logger.info(f"No existing results found for {mode}_{feature}_{prompt_style}")
        return False


def main(data_path: str,
         model_path:str,
         output_result_path: str,
         output_model_dir: str,
         learning_rate: str,
         feature: str,
         mode: str = 'full',
         prompt_style: str = 'original',
         seed_variant: int = 1
        ):

    set_global_seed(seed_variant)
    log_time_info("Starting main function, seed: {}".format(seed_variant*42))

    log_time_info(f"hypeparameter info: prompt style: {prompt_style}, feature: {feature}, learning rate: {learning_rate}, Model_path: {model_path}, seed: {seed_variant}, mode: {mode}")
    if check_results_exist(output_result_path, feature, prompt_style, mode):
        logger.info(f"Skipping {output_result_path} {mode}_{feature}_{prompt_style} - results already exist")
        return
    model, tokenizer = setup_model_and_tokenizer(model_path, mode)
    logger.info(f"model: {model}")
    log_time_info("Model and tokenizer setup completed")

    X_train, X_test, y_true_text_label = load_data(data_path, prompt_style, seed_variant, n_shot = 0)
    log_time_info("Data loading completed, seed: {}".format(seed_variant*1))

    train_data = Dataset.from_pandas(X_train)
    #train_data = train_data.map(lambda x: {"data": tokenizer.apply_chat_template(x["data"], tokenize=False, add_generation_prompt=True, enable_thinking=False)})
    train_data = train_data.map(lambda x: {"data": tokenizer.apply_chat_template(x["data"], tokenize=False, add_generation_prompt=True)})
    print_gpu_info()

    log_time_info("Start training with seed: {}".format(seed_variant*42))
    trainer = setup_trainer(model, tokenizer, train_data, output_model_dir, learning_rate, mode, seed_variant)
    trainer.train()
    log_time_info("Training completed")

    y_preds, y_true = predict(X_test, model, tokenizer, y_true_text_label)

    log_time_info("Prediction completed")
    evaluation_results = evaluate_predictions(y_true, y_preds)
    X_test['preds'] = y_preds
    X_test['true'] = y_true

    save_evaluation_results(X_test, evaluation_results, output_result_path, feature, prompt_style, mode)
    log_time_info("saving the evaluation results")


if __name__ == "__main__":
    fire.Fire(main)

