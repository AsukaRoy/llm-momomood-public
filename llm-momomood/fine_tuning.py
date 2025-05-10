# -*- coding: utf-8 -*-
import os
import ast
import random
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BNB_CUDA_VERSION"] = "118"
os.environ['HF_HOME']='/m/cs/scratch/networks-nima-mmm2018/yunhao/model/'
import logging
from typing import List, Tuple, Dict
from datetime import datetime
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer, setup_chat_format
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline)

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

# -- No EVAL -- #
# -- training mode: 'full' ; 'embedding' ; 'cls' -- #

def construct_prompt(user_data, n_shot_examples):
    """Constructs a chat-like prompt given user data and optional few-shot examples."""
    messages = [{"role": "system",
                 "content": SYSTEM_PROMPT}]

    for example in n_shot_examples:
        messages.append({"role": "user", "content": example['data']})
        messages.append({"role": "assistant", "content": f'Label: {example["label"]} \n'})
    messages.append({"role": "user", "content": user_data['data']})
    messages.append({"role": "assistant", "content": f'Label: {user_data["label"]} \n'})
    return messages

def construct_test_prompt(user_data, n_shot_examples):
    """Constructs a chat-like prompt given user data and optional few-shot examples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in n_shot_examples:
        messages.append({"role": "user", "content": example['data']})
        messages.append({"role": "assistant", "content": f'Label: {example["label"]} \n'})
    messages.append({"role": "user", "content": user_data['data']})

    return messages

def select_n_shot_examples(df, label, original_example, n=0):
    """Select n-shot examples from the dataframe based on the label."""
    # if n==0:
    #     return df.sample(n=0, replace=False).to_dict('records')
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
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    terminators = [
            text_pipeline.tokenizer.eos_token_id,
            text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    y_true, y_preds = [], []
    for index, row in test.iterrows():
        messages = row['data']
        prompt = text_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        print('-----------------------------------------------------------')
        print(prompt)
        generation = text_pipeline(
            prompt,
            max_new_tokens=16,
            eos_token_id=terminators,
            do_sample=False,
            temperature = 0.1,
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
        training_arguments = TrainingArguments(
            output_dir=output_model_dir,                    # directory to save and repository id
            num_train_epochs=1,                       # number of training epochs
            per_device_train_batch_size=1,
            # per_device_eval_batch_size=1,            # batch size per device during training
            # eval_accumulation_steps=1,
            gradient_accumulation_steps=4,            # number of steps before performing a backward/update pass
            gradient_checkpointing=False,             # use gradient checkpointing to save memory
            optim="adamw_torch",
            save_steps=0,
            logging_steps=10,                         # log every 10 steps
            learning_rate=float(learning_rate),
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
            group_by_length=False,
            lr_scheduler_type="cosine",               # use cosine learning rate scheduler
            save_strategy="no", 
            # eval_strategy="steps",
            # eval_steps=40,                            # evaluate every 40 steps
            seed=42*seed_variant
        )


        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_data,
            # eval_dataset=eval_data,
            dataset_text_field="data",
            tokenizer=tokenizer,
            #max_seq_length=max_seq_length,
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },     
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
        training_arguments = TrainingArguments(
            output_dir=output_model_dir,                    # directory to save and repository id
            num_train_epochs=1,                       # number of training epochs
            per_device_train_batch_size=1,
            # per_device_eval_batch_size=1,            # batch size per device during training
            # eval_accumulation_steps=1,
            gradient_accumulation_steps=4,            # number of steps before performing a backward/update pass
            gradient_checkpointing=False,             # use gradient checkpointing to save memory
            optim="paged_adamw_32bit",
            save_steps=0,
            logging_steps=10,                         # log every 10 steps                                                
            learning_rate=float(learning_rate),
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
            group_by_length=False,
            lr_scheduler_type="cosine",               # use cosine learning rate scheduler
            save_strategy="no",
            # eval_strategy="steps",
            # eval_steps=40,                            # evaluate every 40 steps
            seed=42*seed_variant
        )
        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_data,
            # eval_dataset=eval_data,
            peft_config=peft_config,
            dataset_text_field="data",
            tokenizer=tokenizer,
            #max_seq_length=max_seq_length,
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
            
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
        compute_dtype = getattr(torch, "float16")
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
        quantization_config=bnb_config,)

    elif mode == 'embedding':

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
        )
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze embedding layer
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = True
        # unfreeze classification head
        # for param in model.score.parameters():
        #     param.requires_grad = True

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
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

    log_time_info(f"hypeparameter info: prompt style: {prompt_style}, feature: {feature}, learning rate: {learning_rate}")

    model, tokenizer = setup_model_and_tokenizer(model_path, mode)
    log_time_info("Model and tokenizer setup completed")

    X_train, X_test, y_true_text_label = load_data(data_path, prompt_style, seed_variant, n_shot = 0)
    log_time_info("Data loading completed, seed: {}".format(seed_variant*1))

    train_data = Dataset.from_pandas(X_train)
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

