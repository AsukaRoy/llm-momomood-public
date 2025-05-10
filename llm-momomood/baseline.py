import pandas as pd
import json

import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample

import warnings

warnings.filterwarnings('ignore')


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

path = '/m/cs/scratch/networks-nima-mmm2018/yagao/data/state_transition_prediction/feature_selection/state_transition_sequences_magnitude_max+battery_avg_afternoon+screen_off_night+screen_use_afternoon.json'
# Open and read the JSON file

with open(path, 'r') as file:
    data = json.load(file)


all_rows = []

# Iterate over each user in the data
for user_entry in data:
    user_id = user_entry.get("user_id")
    gender = user_entry.get("gender")
    group = user_entry.get("group")
    age   = user_entry.get("age")
    
    user_data = user_entry.get("data", [])
    
    # Iterate over the measurements/data points for this user
    for i, record in enumerate(user_data):
        row_dict = {
            "user_id": user_id,
            "gender": gender,
            "group": group,
            "age": age,
            # Current PHQ_week2
            "phq_week2": record.get("PHQ_week2"),
        }
        
        # Flatten battery_afternoon: battery_afternoon_1 ... battery_afternoon_14
        battery_list = record.get("battery_afternoon", [])
        for idx in range(14):
            col_name = f"battery_afternoon_{idx+1}"
            value = battery_list[idx] if idx < len(battery_list) else None
            row_dict[col_name] = value
        
        # Flatten screen_off_night_duration: screen_off_night_duration_1 ... _14
        off_night_list = record.get("screen_off_night_duration", [])
        for idx in range(14):
            col_name = f"screen_off_night_duration_{idx+1}"
            value = off_night_list[idx] if idx < len(off_night_list) else None
            row_dict[col_name] = value

        # Flatten screen_off_night_duration: screen_use_duration_afternoon_1 ... _14
        screen_afternoon_list = record.get("screen_use_duration_afternoon", [])
        for idx in range(14):
            col_name = f"screen_use_duration_afternoon_{idx+1}"
            value = screen_afternoon_list[idx] if idx < len(screen_afternoon_list) else None
            row_dict[col_name] = value

        # Flatten screen_off_night_duration: screen_use_duration_afternoon_1 ... _14
        magnitude_max_list = record.get("magnitude_max", [])
        for idx in range(14):
            col_name = f"magnitude_max_{idx+1}"
            value = magnitude_max_list[idx] if idx < len(magnitude_max_list) else None
            row_dict[col_name] = value
        
        # future_PHQ_week2: the next record’s PHQ_week2 if it exists
        if (i + 1) < len(user_data):
            future_date = user_data[i+1].get("date")
            current_date = user_data[i].get("date")
            if(str(pd.to_datetime(future_date) - pd.to_datetime(current_date)) == '14 days 00:00:00'):
                row_dict["future_phq_week2"] = user_data[i+1].get("PHQ_week2")

                threshold = 3
                if row_dict["future_phq_week2"] - row_dict["phq_week2"] > threshold:
                    row_dict['label'] = "More Depressed"
                else:
                    if row_dict['phq_week2'] - row_dict['future_phq_week2'] > threshold:
                        row_dict['label'] = "Less Depressed"
                    else:
                        row_dict['label'] = 'Remains'
                all_rows.append(row_dict)

# Convert list of dicts to DataFrame
df = pd.DataFrame(all_rows)

# If you want a specific column order, you can explicitly create a list of columns:
columns_order = [
    "user_id", "gender", "group", "age", 
]

# Add battery_afternoon_1..14
columns_order += [f"battery_afternoon_{i}" for i in range(1, 15)]


# Add screen_off_night_duration_1..14
columns_order += [f"magnitude_max_{i}" for i in range(1, 15)]


# Add screen_off_night_duration_1..14
columns_order += [f"screen_off_night_duration_{i}" for i in range(1, 15)]

# Add screen_off_night_duration_1..14
columns_order += [f"screen_use_duration_afternoon_{i}" for i in range(1, 15)]

# Finally add the future_PHQ_week2 column

columns_order.append("phq_week2")
columns_order.append("future_phq_week2")
columns_order.append("label")
# Reindex the DataFrame to ensure the exact column order
df = df.reindex(columns=columns_order)

df = df.dropna(subset=['future_phq_week2'])


def split_train_test_eval(df, subject_column, label_column, oversample = False, random_state = 1):
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
    gss = GroupShuffleSplit(test_size=0.4, n_splits=1, random_state=random_state)
    train_idx, temp_idx = next(gss.split(df, groups=df[subject_column]))
    X_train = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    print(f"Initial split: {len(X_train)} training samples, {len(temp_df)} temporary samples.")
    if(oversample):
        # Oversample the underrepresented classes in the training set
        train_labels = X_train[label_column].value_counts()
        n_resample = int(train_labels.mean())
        for label, count in train_labels.items():
            if count < train_labels.mean():
                X_train = pd.concat([X_train, resample(X_train[X_train[label_column] == label],
                                                      replace=True,
                                                      n_samples=n_resample,
                                                      random_state=random_state)],
                                    ignore_index=True)
        print("Oversampling performed on training data.")
    # Split temp into test and eval
    gss = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=random_state)
    test_idx, eval_idx = next(gss.split(temp_df, groups=temp_df[subject_column]))
    X_test = temp_df.iloc[test_idx]
    X_eval = temp_df.iloc[eval_idx]
    print(f"Final split: {len(X_train)} training samples, {len(X_test)} testing samples, {len(X_eval)} evaluation samples.")
    return X_train, X_test, X_eval



# ------------------------------------------------------------------------------
# Suppose you have a DataFrame 'df' with columns:
#   [ subject_id, label, previous depression state, battery_afternoon_1..14, screen_off_night_duration_1..14, screen_use_duration_afternoon1..14, etc. ]
# ------------------------------------------------------------------------------
subject_column = 'user_id'   # or whatever your subject column is
label_column   = 'label'        # the column with "More Depressed"/"Less Depressed"/"Remains"

# 1) Group-based split (train, eval, test)
X_train_df, X_test_df, X_eval_df = split_train_test_eval(df, subject_column, label_column, True)

# 2) Define your feature sets for ablation
battery_features     = [f"battery_afternoon_{i}" for i in range(1, 15)]
magnitude_max_feature = [f"magnitude_max_{i}" for i in range(1, 15)]
screen_off_features  = [f"screen_off_night_duration_{i}" for i in range(1, 15)]
screen_use_features  = [f"screen_use_duration_afternoon_{i}" for i in range(1, 15)]


# Potential combinations if you want to test them too:
feature_sets = {
    "Battery Only": battery_features,
    "Screen-Off Only": screen_off_features,
    "Screen-Use Only": screen_use_features,
    "Magnitude Only": magnitude_max_feature,
    "Battery + Screen-Off": battery_features + screen_off_features,
    "Battery + Screen-Use": battery_features + screen_use_features,
    "Screen-Off + Screen-Use": screen_off_features + screen_use_features,
    "Battery + Magnitude": battery_features + magnitude_max_feature,
    "Screen-Off + Magnitude": screen_off_features + magnitude_max_feature,
    "Screen-Use + Magnitude": screen_use_features + magnitude_max_feature,
    "Battery + Screen-Off + Screen-Use": battery_features + screen_off_features + screen_use_features,
    "Magnitude + Screen-Off + Screen-Use": magnitude_max_feature + screen_off_features + screen_use_features,
    "Battery + Magnitude + Screen-Use": battery_features + magnitude_max_feature + screen_use_features,
    "Battery + Screen-Off + Magnitude": battery_features + screen_off_features + magnitude_max_feature,
    "All Features": battery_features + screen_off_features + screen_use_features + magnitude_max_feature
}


# Optionally combine train & eval if you want to train on more data:
combined_train_eval_df = pd.concat([X_train_df, X_eval_df], ignore_index=True)

# 3) Encode label
from sklearn.model_selection import GridSearchCV
label_encoder = LabelEncoder()
df_label_list = list(combined_train_eval_df[label_column])
label_encoder.fit(df_label_list)

# 4) Function to train & evaluate on a given feature set
def train_and_evaluate(train_df, eval_df, feature_list, label_encoder, scoring="f1_macro", param_grid=None, cv=3):
    """
    Train XGBClassifier on the given feature list and return metrics.
    """
    # Filter out any rows with NaN in label or features if needed
    #train_df = train_df.dropna(subset=[label_column] + feature_list)
    #eval_df  = eval_df.dropna(subset=[label_column] + feature_list)
    for col in feature_list:
        train_df.fillna(0, inplace=True)
        eval_df.fillna(0, inplace=True)
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    combined_train_eval_df = pd.concat([train_df, eval_df], ignore_index=True)

    # Prepare data
    X_train_eval = combined_train_eval_df[feature_list]
    y_train_eval = label_encoder.transform(combined_train_eval_df[label_column])

    X_eval  = eval_df[feature_list]
    y_eval  = label_encoder.transform(eval_df[label_column])

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.1, 0.01]
        }

    X = X_train_eval
    y = y_train_eval

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Base XGB model
    xgb_base = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_scaled, y)

    print(f"Best Score (CV avg): {grid_search.best_score_}")
    print(f"Best Params: {grid_search.best_params_}")
    return grid_search.best_params_

# 5) Run ablation study

param_options = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.3, 0.1, 0.05, 0.01, 0.001],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.1, 0.5, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0]
}
best_params = {}
for set_name, features in feature_sets.items():
    best_para = train_and_evaluate(X_train_df, X_eval_df, features, label_encoder, param_grid = param_options)
    best_params[set_name] = best_para
    print(best_para)
    print('==========================================')


def split_data(df, subject_column, label_column, oversample=False, random_state=1):
    """
    Splits data into training, testing, and evaluation sets while maintaining
    the class distribution and subject-level consistency via group shuffle.
    """

    gss = GroupShuffleSplit(test_size=0.4, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[subject_column]))
    X_train_init = df.iloc[train_idx].copy()
    X_test = df.iloc[test_idx].copy()
    print(f"Initial split: {len(X_train_init)} training samples, {len(X_test)} test samples.")
    
    gss = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=random_state)
    train_idx, eval_idx = next(gss.split(X_train_init, groups=X_train_init[subject_column]))
    X_eval = X_train_init.iloc[eval_idx].copy()
    X_train = X_train_init.iloc[train_idx].copy()
    print(f"Split to train and eval: {len(X_train)} training samples, {len(X_eval)} evaluation samples.")

    if oversample:
        # Oversample the underrepresented classes in the training set
        train_labels = X_train[label_column].value_counts()
        n_resample = int(train_labels.mean())
        for label, count in train_labels.items():
            if count < train_labels.mean():
                X_train = pd.concat(
                    [
                        X_train,
                        resample(
                            X_train[X_train[label_column] == label],
                            replace=True,
                            n_samples=n_resample,
                            random_state=random_state
                        )
                    ],
                    ignore_index=True
                )
        print("Oversampling performed on training data.")
    
    print(f"Final split: {len(X_train)} training samples, {len(X_eval)} evaluation samples.")

    return X_train, X_eval


# 4) Function to train & evaluate on a given feature set
def train_and_evaluate(train_df, eval_df, feature_list, label_encoder, b_params, random_state):
    """
    Train XGBClassifier on the given feature list and return metrics.
    """
    # Filter out any rows with NaN in label or features if needed
    #train_df = train_df.dropna(subset=[label_column] + feature_list)
    #eval_df  = eval_df.dropna(subset=[label_column] + feature_list)
    for col in feature_list:
        train_df.fillna(0, inplace=True)
        eval_df.fillna(0, inplace=True)
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
    

    # Prepare data
    X_train = train_df[feature_list]
    y_train = label_encoder.transform(train_df[label_column])
    X_eval  = eval_df[feature_list]
    y_eval  = label_encoder.transform(eval_df[label_column])

    #print("Train label counts:\n",train_df[label_column].value_counts())
    #print("Eval label counts:\n", eval_df[label_column].value_counts())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled  = scaler.transform(X_eval)
    # Build model
    # Ensure certain parameters
    b_params.setdefault("random_state", random_state)
    b_params.setdefault("use_label_encoder", False)
    b_params.setdefault("eval_metric", "mlogloss")

    # Train final model
    print(b_params)
    model = XGBClassifier(**b_params)
    #model = XGBClassifier(
    #    random_state=42,
    #    use_label_encoder=False,
    #    eval_metric='mlogloss'
    #)
    model.fit(X_train_scaled, y_train)


   # -----------------------------------------------------------------
    # 6) Predict & Evaluate
    # -----------------------------------------------------------------
    y_pred = model.predict(X_eval_scaled)
    target_names = label_encoder.classes_  # e.g. ["Less Depressed", "More Depressed", "Remains"]

    report = classification_report(
        y_true=y_eval,
        y_pred=y_pred,
        zero_division=0,                    # handle zero divisions
        target_names=target_names,          # show label names
        labels=list(range(len(target_names)))  # ensure the correct mapping [0,1,2,...]
    )

    # Additional metrics
    conf_matrix = confusion_matrix(y_eval, y_pred).tolist()
    acc = accuracy_score(y_eval, y_pred)
    macro_f1 = f1_score(y_eval, y_pred, average='macro')
    weighted_f1 = f1_score(y_eval, y_pred, average='weighted')
    macro_precision = precision_score(y_eval, y_pred, average='macro')
    weighted_precision = precision_score(y_eval, y_pred, average='weighted')
    macro_recall = recall_score(y_eval, y_pred, average='macro')
    weighted_recall = recall_score(y_eval, y_pred, average='weighted')

    return model, {
        "Accuracy": acc,
        "Macro Precision": macro_precision,
        "Weighted Precision": weighted_precision,
        "Macro Recall": macro_recall,
        "Weighted Recall": weighted_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Confusion Matrix": conf_matrix,
        "Classification Report": report
    }


# 5) Run ablation study
results = []
subject_column = 'user_id'   # or whatever your subject column is
label_column   = 'label'        # the column with "More Depressed"/"Less Depressed"/"Remains"

random_seeds = range(1,6) 
for rs in random_seeds:
    # 1) Group-based split (train, eval, test)
    X_train_df, X_eval_df = split_data(df, subject_column, label_column, True, rs)
    
    # 2) Define your feature sets for ablation
    battery_features     = [f"battery_afternoon_{i}" for i in range(1, 15)]
    magnitude_max_feature = [f"magnitude_max_{i}" for i in range(1, 15)]
    screen_off_features  = [f"screen_off_night_duration_{i}" for i in range(1, 15)]
    screen_use_features  = [f"screen_use_duration_afternoon_{i}" for i in range(1, 15)]
    
    
    # Potential combinations if you want to test them too:
    feature_sets = {
        "Battery Only": battery_features,
        "Screen-Off Only": screen_off_features,
        "Screen-Use Only": screen_use_features,
        "Magnitude Only": magnitude_max_feature,
        "Battery + Screen-Off": battery_features + screen_off_features,
        "Battery + Screen-Use": battery_features + screen_use_features,
        "Screen-Off + Screen-Use": screen_off_features + screen_use_features,
        "Battery + Magnitude": battery_features + magnitude_max_feature,
        "Screen-Off + Magnitude": screen_off_features + magnitude_max_feature,
        "Screen-Use + Magnitude": screen_use_features + magnitude_max_feature,
        "Battery + Screen-Off + Screen-Use": battery_features + screen_off_features + screen_use_features,
        "Magnitude + Screen-Off + Screen-Use": magnitude_max_feature + screen_off_features + screen_use_features,
        "Battery + Magnitude + Screen-Use": battery_features + magnitude_max_feature + screen_use_features,
        "Battery + Screen-Off + Magnitude": battery_features + screen_off_features + magnitude_max_feature,
        "All Features": battery_features + screen_off_features + screen_use_features + magnitude_max_feature
    }
    
    
    # 3) Encode label
    label_encoder = LabelEncoder()
    df_label_list = list(X_train_df[label_column]) + list(X_eval_df[label_column])
    label_encoder.fit(df_label_list)
    
    for set_name, features in feature_sets.items():
        model, cls_report = train_and_evaluate(X_train_df, X_eval_df, features, label_encoder, best_params[set_name], rs * 42)
        #print(cls_report)
        results.append({
            'random_seed': rs,
            "Feature Set": set_name,
            "best_params":best_params[set_name] ,
            "Macro F1": cls_report.get("Macro F1"),
            "Macro Precision": cls_report.get("Macro Precision"),
            "Macro Recall": cls_report.get("Macro Recall"),
            "Accuracy": cls_report.get("Accuracy"),
            "Weighted F1": cls_report.get("Weighted F1"),
            
            #"Feature Set": cls_report.get("Macro F1"),
    
        })
        #print('==========================================')

results_df = pd.DataFrame(results)


results_df.to_csv('/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/results/scores/summary/baseline_xgboost.csv', index = None)