# -*- coding: utf-8 -*-
import os
import ast
import random
import torch
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['HF_HOME']='/scratch/shareddata/dldata/huggingface-hub-cache'
print(f'HF home directory is {os.environ['HF_HOME']}')
from transformers import GenerationConfig

from sklearn.utils import resample
from datetime import datetime
import time
import pandas as pd
#from langchain.prompts import PromptTemplate
#from langchain_community.llms import HuggingFacePipeline
#from langchain.llms import LlamaCpp
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from tqdm.auto import tqdm
import argparse
from datasets import load_dataset
import fire
import json
import glob

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
import numpy as np


torch.cuda.empty_cache()
# after imports, before loading the model
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "true"

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
def get_terminators(tokenizer) -> list:
    """
    Build a robust list of end-of-turn/end-of-text token IDs across different chat templates.
    """
    ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))

    # Common end-of-turn tokens across families
    candidate_tokens = [
        "<|eot_id|>",           # Llama 3.x
        "<|im_end|>",           # Qwen family
        "<end_of_turn>",        # Gemma 3
        "</s>",                 # Mistral (and others)
        "<|eos_token|>",        # Some variants
        "<|endoftext|>",
        "<|end_of_text|>",
        "<eos>",
    ]
    for tok in candidate_tokens:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id and tid != -1:
                ids.add(int(tid))
        except Exception:
            pass

    return list(ids) if ids else None

def ensure_padding(tokenizer):
    # Some models don't ship a pad token; safest fallback is eos
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Create a pad token if truly absent
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer



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
    # if n==0:
    #     return df.sample(n=0, replace=False).to_dict('records')
    filtered_df = df[(df['previous depression state'] == label) & (df['data'] != original_example['data'])]
    examples = filtered_df.sample(n=n, replace=False).to_dict('records')
    return examples

def select_examples_by_user(df, user_id, row_index, shot_select, n=1):
    """Select n examples from the dataframe for the same user, excluding the row with the given index."""
    # Filter the dataframe for the specified user and exclude the row by index
    if(shot_select == "same_subject"):
        # select from same subject
        user_df = df[(df['user'] == user_id) & (df.index != row_index)]
    else:
        # randomly select
        user_df = df[df.index != row_index]
    examples = user_df.sample(min(n, len(user_df)), random_state=SEED).to_dict(orient='records')
    return examples


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

def parse_n_shots(n_shots):
    """Parses the n_shots parameter to ensure it is a list of integers."""
    if isinstance(n_shots, str):
        try:
            return list(ast.literal_eval(n_shots))
        except (ValueError, SyntaxError):
            raise ValueError("Invalid n_shots format. Use a list or string representation like '[0, 1, 3, 5]'.")
    elif isinstance(n_shots, list):
        return n_shots
    else:
        raise ValueError("n_shots should be a list or a string representation of a list.")

def predict(test, model, tokenizer, y_true_text_label):
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype="bfloat16")
    #terminators = [
    #        text_pipeline.tokenizer.eos_token_id,
    #        text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #    ]
    terminators = get_terminators(text_pipeline.tokenizer)
    
    # Fallback: if no terminators found, use eos_token_id
    if not terminators:
        if text_pipeline.tokenizer.eos_token_id is not None:
            terminators = [text_pipeline.tokenizer.eos_token_id]
        else:
            terminators = None  # Let the model use its default
    y_true, y_preds = [], []
    for index, row in tqdm(test.iterrows(), total=test.shape[0], desc="Prediction Progress"):
        messages = row['data']

        
        try:
            # Apply chat template with error handling
            if hasattr(text_pipeline.tokenizer, 'apply_chat_template'):
                prompt = text_pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                # Fallback for tokenizers without chat template
                prompt = ""
                for msg in messages:
                    if msg['role'] == 'system':
                        prompt += f"<bos><start_of_turn>model\n{msg['content']}<end_of_turn>\n"
                    elif msg['role'] == 'user':
                        prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
                    elif msg['role'] == 'assistant':
                        prompt += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
                prompt += "<start_of_turn>model\n"
            
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            # Simple fallback
            prompt = str(messages)
        
        #print(f'---------------------------{index}-----------------------------')
        #print(prompt)
        generation_kwargs = {
            "max_new_tokens": 8,
            "do_sample": False,  # Deterministic
            "pad_token_id": text_pipeline.tokenizer.eos_token_id
        }
        
        # Only add eos_token_id if terminators exist
        if terminators:
            generation_kwargs["eos_token_id"] = terminators
        print_gpu_info()
        n_tokens = len(text_pipeline.tokenizer(prompt)["input_ids"])
        print(f"Prompt token length: {n_tokens}")

        generation = text_pipeline(
            prompt,
            **generation_kwargs
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

def log_time_info(message: str):
    """Logs the current time with a custom message."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{message} at {current_time}")

def save_evaluation_results(X_test, evaluation_results, output_result_path, feature, prompt_style, shot, shot_select):

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    output_file_json = f"{output_result_path}/in_context_{shot}_{feature}_{prompt_style}_{timestamp}.json"

    # Save to JSON for structured data
    with open(output_file_json, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    X_test.to_csv(f"{output_result_path}/in_context_{shot}_{feature}_{prompt_style}_{timestamp}.csv")

    logger.info(f"Results saved to {output_file_json}")


def check_results_exist(output_result_path, feature, prompt_style, n_shots, shot_select):
    """
    Check if results files already exist for the given parameters.
    Returns True if files exist, False otherwise.
    """
    # Create pattern to match existing files
    json_pattern = f"{output_result_path}/in_context_{n_shots}_{feature}_{prompt_style}_*.json"
    csv_pattern = f"{output_result_path}/in_context_{n_shots}_{feature}_{prompt_style}_*.csv"
    
    # Check if any files match the pattern
    json_files = glob.glob(json_pattern)
    csv_files = glob.glob(csv_pattern)
    
    if json_files and csv_files:
        logger.info(f"Results files already exist for {n_shots}-shot, feature: {feature}, prompt_style: {prompt_style}")
        if json_files:
            logger.info(f"Found JSON files: {json_files}")
        if csv_files:
            logger.info(f"Found CSV files: {csv_files}")
        return True
    
    return False

def main(data_path: str,
         model_path: str,
         output_result_path: str,
         feature: str,
         shot_select: str = 'random',
         n_shots_list='[0,1,3,5,10,15,25]',
         prompt_style: str = 'original',
         seed: int = 1):
    set_global_seed(seed)
    log_time_info("Starting main function")
    log_time_info(f"hyperparameter info: prompt style: {prompt_style}, feature: {feature}, shot_select: {shot_select}, model_path: {model_path}, n_shots_list: {n_shots_list}, seed: {seed}")

    #torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")
    print(f"Working on {torch.cuda.device_count()}")
    print(f"Working on {torch.version.cuda}")
    print_gpu_info()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = ensure_padding(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 torch_dtype="bfloat16",         # use bf16 if your GPU supports it
                                                 device_map="auto",
                                                 attn_implementation="sdpa")
    # choose bf16 on A100/H100 etc., otherwise fp16
    major_cc = torch.cuda.get_device_capability(0)[0]
    dtype = torch.bfloat16 if major_cc >= 8 else torch.float16

    device = model.device

    n_shots_list = parse_n_shots(n_shots_list)
    model.config.use_cache = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    for n_shots in n_shots_list:
        log_time_info(f"====================================================")
        log_time_info(f"Begin with {n_shots}-shot prompting")
                # Check if results already exist and skip if requested
        if  check_results_exist(output_result_path, feature, prompt_style, n_shots, shot_select):
            log_time_info(f"Skipping {n_shots}-shot prompting as results already exist")
            log_time_info(f"====================================================")
            continue

        
        X_train, X_test, y_true_text_label = load_data(data_path, prompt_style, seed, n_shot = n_shots)
        log_time_info("Data loading completed")

        y_preds, y_true = predict(X_test, model, tokenizer, y_true_text_label)
        log_time_info('-----------------------------------------------------------')
        #log_time_info('y_preds:', y_preds)
        #log_time_info('y_true:', y_true)
        log_time_info("Prediction completed")
        evaluation_results = evaluate_predictions(y_true, y_preds)
        X_test['preds'] = y_preds
        X_test['true'] = y_true

        save_evaluation_results(X_test, evaluation_results, output_result_path, feature, prompt_style, n_shots, shot_select)
        log_time_info("saving the evaluation results")
        log_time_info(f"====================================================")

if __name__ == "__main__":
    fire.Fire(main)
