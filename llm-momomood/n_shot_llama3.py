# -*- coding: utf-8 -*-
import os
import ast
import random
import torch
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['HF_HOME']='/scratch/shareddata/dldata/huggingface-hub-cache'
os.environ['HF_HOME']='/m/cs/scratch/networks-nima-mmm2018/yunhao/model'

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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
import numpy as np

SEED_COUNT = 5
SEED = 42 * SEED_COUNT

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

def construct_prompt(user_data, n_shot_examples =None):
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


def split_data(df, subject_column, label_column, oversample = False):
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
    gss = GroupShuffleSplit(test_size=0.4, n_splits=1, random_state=SEED_COUNT)
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
                                                      random_state=SEED_COUNT)],
                                    ignore_index=True)
        print("Oversampling performed on training data.")
    print(f"Final split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    X_train_final = X_train[[subject_column,'previous depression state', 'label']]
    X_test_final = X_test[[subject_column,'previous depression state','label']]
    # change the column name to 'data' for the subject_column
    X_train_final.rename(columns={subject_column: 'data'}, inplace=True)
    X_test_final.rename(columns={subject_column: 'data'}, inplace=True)
    return X_train_final, X_test_final


def load_data(filename: str, prompt_style: str, n_shot=0) -> pd.DataFrame:
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

    #df['LLM_prediction'] = None
    #df['binary_classification'] = None

    X_train, X_test = split_data(df, subject_column = data, label_column = 'label', oversample=True) # return with column 'data'

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
        tokenizer=tokenizer)
    terminators = [
            text_pipeline.tokenizer.eos_token_id,
            text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    y_true, y_preds = [], []
    for index, row in tqdm(test.iterrows(), total=test.shape[0], desc="Prediction Progress"):
        messages = row['data']
        prompt = text_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        print(f'---------------------------{index}-----------------------------')
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


def main(data_path: str,
         model_path:str,
         output_result_path: str,
         feature: str,
         shot_select: str = 'random',
         n_shots_list='[15,25]',
         prompt_style: str = 'original'):
    
    log_time_info("Starting main function")
    log_time_info(f"hypeparameter info: prompt style: {prompt_style}, feature: {feature}, shot_select: {shot_select}")

    #torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")
    print(f"Working on {torch.cuda.device_count()}")
    print(f"Working on {torch.version.cuda}")
    print_gpu_info()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    device = model.device

    n_shots_list = parse_n_shots(n_shots_list)

    for n_shots in n_shots_list:
        log_time_info(f"====================================================")
        log_time_info(f"Begin with {n_shots}-shot prompting")
        X_train, X_test, y_true_text_label = load_data(data_path, prompt_style, n_shot = n_shots)
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
