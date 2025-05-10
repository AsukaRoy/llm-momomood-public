# -*- coding: utf-8 -*-
import os
import ast
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BNB_CUDA_VERSION"] = "118"
#os.environ['HF_HOME']='/scratch/shareddata/dldata/huggingface-hub-cache'
os.environ['HF_HOME']='/m/cs/scratch/networks-nima-mmm2018/yunhao/model/'
from datetime import datetime
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

import bitsandbytes as bnb
import torch
import torch.nn as nn
import random
import torch

from tqdm.auto import tqdm
import argparse
from datasets import load_dataset
import fire
import json

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


SYSTEM_PROMPT = """You are an intelligent healthcare agent skilled in predicting changes in mental health conditions.
TASK: Your task is to determine how the user's mental health state changes based on digital data of mobile sensor readings from an individual.
INPUT: The input contains the following contents: data about mobile phone usage over the previous two-week; whether being depressed or not over the previous two-week; data about mobile phone usage over the current two-week.
OUTPUT: To determine how the user's mental health state changes, you should only respond with "Label: Remains" or "Label: More Depressed" or "Label: Less Depressed". Make sure to only return the label and nothing more. 
"""


def construct_prompt(user_data):
    """Constructs a chat-like prompt given user data and optional few-shot examples."""
    messages = [{"role": "system", 
                 "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_data['data']})
    messages.append({"role": "assistant", "content": f'Label: {user_data["label"]}'})
    return messages

def construct_test_prompt(user_data):
    """Constructs a chat-like prompt given user data and optional few-shot examples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    messages.append({"role": "user", "content": user_data['data']})
    

    return messages

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
        return 3  # Indicates an unknown or unexpected response


def predict(test, model, tokenizer, y_true_text_label):
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #max_new_tokens = 1, 
        #temperature = 0.1,
    )
    terminators = [
            text_pipeline.tokenizer.eos_token_id,
            text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    y_true, y_preds = [], []
    for index, row in test.iterrows():
        messages = row['data']
        ##print(type(messages))
        #rint(messages)
        prompt = text_pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        print('-----------------------------------------------------------')
        print(prompt)
        generation = text_pipeline(
            prompt,
            max_new_tokens=1024,
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
        if category_result in [0, 1, 2 ,3]:
            if y_true_text_label[index] == 'Remains': 
                true_label = 0
            elif y_true_text_label[index] == 'More Depressed':
                true_label = 1
            elif y_true_text_label[index] == 'Less Depressed':
                true_label = 2
            y_true.append(true_label)
            y_preds.append(category_result)

    return y_preds, y_true

def split_data(df, subject_column, label_column):
    """
    Splits data into training, testing, and evaluation sets while maintaining the class distribution and subject-level consistency.
    
    Parameters:
    - df (DataFrame): The input DataFrame containing the data and labels.
    - subject_column (str): The column name representing subjects.
    - label_column (str): The column name representing the labels.
    
    Returns:
    - X_train (DataFrame): Training data subset.
    - X_test (DataFrame): Testing data subset.
    - X_eval (DataFrame): Evaluation data subset.
    """
    # Split subjects into training and temp (test + eval)
    gss = GroupShuffleSplit(test_size=0.4, n_splits=1, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df[subject_column]))
    X_train = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Split temp into test and eval
    gss = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=42)
    test_idx, eval_idx = next(gss.split(temp_df, groups=temp_df[subject_column]))
    X_test = temp_df.iloc[test_idx]
    X_eval = temp_df.iloc[eval_idx]
    
    return X_train[['data','label']], X_test[['data','label']], X_eval[['data','label']]


def main(ckpt_dir: str, 
         tokenizer_path: str, 
         data_path: str,
         prompt_format:str,
         model_path:str,
         result_path: str,
         shot_select: str,
         feature_select: str,
         temperature: float = 0.6, 
         top_p: float = 0.5,
         max_seq_len: int = 512, 
         max_batch_size: int = 8):
    filename = "/m/cs/scratch/networks-nima-mmm2018/yagao/data/state_transition_prediction/phq9/state_transition_sequences_phq9_magnitude_prompts_5.csv"
    df =  pd.read_csv(filename)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"working on {device}")
    
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

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
        quantization_config=bnb_config, 
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    
    #max_seq_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    X_train, X_test, X_eval = split_data(df, subject_column = 'data', label_column = 'label')
    
    y_true_text_label = X_test.label
 

    X_train = pd.DataFrame(X_train.apply(construct_prompt, axis=1), 
                           columns=["data"])
    X_eval = pd.DataFrame(X_eval.apply(construct_prompt, axis=1), 
                          columns=["data"])
    X_test = pd.DataFrame(X_test.apply(construct_test_prompt, axis=1), 
                          columns=["data"])

    y_preds, y_true = predict(X_test, model, tokenizer, y_true_text_label)
    evaluate_predictions(y_true, y_preds)
    train_data = Dataset.from_pandas(X_train)
    #X_test = Dataset.from_pandas(X_test)
    eval_data = Dataset.from_pandas(X_eval)
    print('Using device:', device)

    train_data = train_data.map(lambda x: {"data": tokenizer.apply_chat_template(x["data"], tokenize=False, add_generation_prompt=True)})
    
    eval_data = eval_data.map(lambda x: {"data": tokenizer.apply_chat_template(x["data"], tokenize=False, add_generation_prompt=True)})
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    output_dir="/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/scripts/trained_weigths/"
    
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
        output_dir=output_dir,                    # directory to save and repository id
        num_train_epochs=1,                       # number of training epochs
        per_device_train_batch_size=1,            # batch size per device during training
        gradient_accumulation_steps=4,            # number of steps before performing a backward/update pass
        gradient_checkpointing=False,             # use gradient checkpointing to save memory
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=10,                         # log every 10 steps
        learning_rate=2e-4,                       # learning rate, based on QLoRA paper
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
        max_steps=-1,
        warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
        group_by_length=False,
        lr_scheduler_type="cosine",               # use cosine learning rate scheduler
        #report_to="tensorboard",                  # report metrics to tensorboard
        evaluation_strategy="epoch",               # save checkpoint every epoch
        seed=42
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="data",
        tokenizer=tokenizer,
        #max_seq_length=max_seq_length,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )
    trainer.train()
    y_preds, y_true = predict(X_test, model, tokenizer, y_true_text_label)
    
    evaluation_results = evaluate_predictions(y_true, y_preds)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_path = f"/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/results/fine_tuning_eval_{timestamp}.json"
        
    # Save to JSON for structured data
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done! model saved to {output_dir}")

    X_test['preds'] = y_preds
    X_test['true'] = y_true
    X_test.to_csv(f"/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/results/fine_tuning_eval_{timestamp}.csv")

if __name__ == "__main__":
    fire.Fire(main)

