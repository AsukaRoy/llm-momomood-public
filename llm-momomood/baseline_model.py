import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import torch.nn.init as init
import warnings
import json

import numpy as np
warnings.filterwarnings('ignore')

def evaluate_predictions(y_true, y_preds):
    """Evaluates classification performance and returns a report with various metrics."""
    target_names = ['Remains', 'More Depressed', 'Less Depressed']
    report = classification_report(y_true, y_preds, zero_division=0, target_names=target_names, labels=[0,1,2])
    conf_matrix = confusion_matrix(y_true, y_preds).tolist()
    acc = accuracy_score(y_true, y_preds)
    macro_f1 = f1_score(y_true, y_preds, average='macro')
    weighted_f1 = f1_score(y_true, y_preds, average='weighted')
    macro_precision = precision_score(y_true, y_preds, average='macro')
    weighted_precision = precision_score(y_true, y_preds, average='weighted')
    macro_recall = recall_score(y_true, y_preds, average='macro')
    weighted_recall = recall_score(y_true, y_preds, average='weighted')

    return {
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

def split_data(df, subject_column, label_column, oversample=False, random_state=1):
    """
    Splits data into training and evaluation sets while maintaining
    the class distribution and subject-level consistency via group shuffle.
    """
    from sklearn.utils import resample
    
    gss = GroupShuffleSplit(test_size=0.4, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[subject_column]))
    X_train_init = df.iloc[train_idx].copy()
    X_test = df.iloc[test_idx].copy()
    
    gss = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=random_state)
    train_idx, eval_idx = next(gss.split(X_train_init, groups=X_train_init[subject_column]))
    X_eval = X_train_init.iloc[eval_idx].copy()
    X_train = X_train_init.iloc[train_idx].copy()

    if oversample:
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
    
    return X_train, X_eval

def reshape_features_to_timeseries(df, feature_prefixes, sequence_length=14):
    """
    Convert flattened features back to time series format for deep learning models
    """
    X_ts = []
    for idx, row in df.iterrows():
        # Extract time series for each feature type
        ts_features = []
        for prefix in feature_prefixes:
            series = [row[f"{prefix}_{i}"] for i in range(1, sequence_length + 1)]
            ts_features.append(series)
        
        # Transpose to get (time_steps, features) format
        X_ts.append(np.array(ts_features).T)
    
    return np.array(X_ts)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim=50):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True, dropout=0.2)
        self.dropout_lstm = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len) for Conv1d
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = x.transpose(1, 2)  # Back to (batch_size, seq_len, features) for LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last output
        
        x = self.dropout_lstm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last output
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))  # Max sequence length
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 2, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def create_model(model_type, input_size, num_classes, **kwargs):
    """Factory function to create models"""
    if model_type == 'lstm':
        return LSTMModel(input_size, num_classes, **kwargs)
    elif model_type == 'cnn_lstm':
        return CNNLSTMModel(input_size, num_classes, **kwargs)
    elif model_type == 'transformer':
        return TransformerModel(input_size, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_pytorch_model(model, train_loader, val_loader, num_epochs=50, patience=10, device='cpu'):
    """Train PyTorch model with early stopping"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def get_baseline_models():
    """
    Define all baseline models with their parameter grids for hyperparameter tuning
    """
    models = {
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'type': 'traditional',
            'params': {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.1, 0.05, 0.01],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'type': 'traditional',
            'params': {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        # Deep Learning Models
        'LSTM': {
            'model': 'lstm',  # Will be created dynamically
            'type': 'deep_learning',
            'params': {
                'epochs': [30, 50],
                'batch_size': [16, 32],
                'patience': [10, 15]
            }
        },
        'CNN-LSTM': {
            'model': 'cnn_lstm',
            'type': 'deep_learning', 
            'params': {
                'epochs': [30, 50],
                'batch_size': [16, 32],
                'patience': [10, 15]
            }
        },
        'Transformer': {
            'model': 'transformer',
            'type': 'deep_learning',
            'params': {
                'epochs': [30, 50],
                'batch_size': [16, 32],
                'patience': [10, 15],
                'num_transformer_blocks': [1, 2]
            }
        }
    }
    
    return models

def train_and_evaluate_deep_learning(train_df, eval_df, feature_list, label_column, model_type, params, label_encoder, random_state=42):
    """
    Train and evaluate deep learning models using time series format with PyTorch
    """
    # Set random seeds
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Define feature prefixes for reshaping
    feature_prefixes = []
    for feature in feature_list:
        prefix = '_'.join(feature.split('_')[:-1])  # Remove the number suffix
        if prefix not in feature_prefixes:
            feature_prefixes.append(prefix)
    
    # Prepare data with proper handling of missing values
    for col in feature_list:
        train_df[col] = train_df[col].fillna(0)
        eval_df[col] = eval_df[col].fillna(0)
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce").fillna(0)
    
    # Reshape to time series format
    X_train_ts = reshape_features_to_timeseries(train_df, feature_prefixes)
    X_eval_ts = reshape_features_to_timeseries(eval_df, feature_prefixes)
    
    # Labels
    y_train = label_encoder.transform(train_df[label_column])
    y_eval = label_encoder.transform(eval_df[label_column])
    
    # Normalize data for deep learning
    scaler = MinMaxScaler()
    n_samples_train, n_timesteps, n_features = X_train_ts.shape
    n_samples_eval = X_eval_ts.shape[0]
    
    X_train_scaled = scaler.fit_transform(X_train_ts.reshape(-1, n_features)).reshape(n_samples_train, n_timesteps, n_features)
    X_eval_scaled = scaler.transform(X_eval_ts.reshape(-1, n_features)).reshape(n_samples_eval, n_timesteps, n_features)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_eval_tensor = torch.FloatTensor(X_eval_scaled)
    y_eval_tensor = torch.LongTensor(y_eval)
    
    # Create data loaders
    batch_size = params.get('batch_size', 32)
    
    # Split training data for validation
    val_split = 0.2
    n_val = int(len(X_train_tensor) * val_split)
    indices = np.random.permutation(len(X_train_tensor))
    train_indices, val_indices = indices[n_val:], indices[:n_val]
    
    train_dataset = TensorDataset(X_train_tensor[train_indices], y_train_tensor[train_indices])
    val_dataset = TensorDataset(X_train_tensor[val_indices], y_train_tensor[val_indices])
    test_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_classes = len(np.unique(y_train))
    input_size = n_features
    
    # Model-specific parameters
    model_kwargs = {}
    if model_type == 'transformer':
        model_kwargs['num_layers'] = params.get('num_transformer_blocks', 2)
    
    model = create_model(model_type, input_size, num_classes, **model_kwargs)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_pytorch_model(
        model, train_loader, val_loader, 
        num_epochs=params.get('epochs', 50),
        patience=params.get('patience', 10),
        device=device
    )
    
    # Make predictions on test set
    model.eval()
    y_pred_list = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            y_pred_list.extend(predictions.cpu().numpy())
    
    y_pred = np.array(y_pred_list)
    
    # Calculate metrics using existing function
    metrics = evaluate_predictions(y_eval, y_pred)
    
    return model, metrics, params

def train_and_evaluate_model(train_df, eval_df, feature_list, label_column, model_name, model_info, label_encoder, random_state=42):
    """
    Train and evaluate a single model on the given feature list
    """
    if model_info['type'] == 'deep_learning':
        # Handle deep learning models
        best_score = -np.inf
        best_model = None
        best_metrics = None
        best_params = None
        
        # Simple parameter search for deep learning (reduced complexity)
        param_combinations = [
            {'epochs': 30, 'batch_size': 32, 'patience': 10},
            {'epochs': 50, 'batch_size': 16, 'patience': 15}
        ]
        
        if model_info['model'] == 'transformer':
            param_combinations = [
                {'epochs': 30, 'batch_size': 32, 'patience': 10, 'num_transformer_blocks': 1},
                {'epochs': 50, 'batch_size': 16, 'patience': 15, 'num_transformer_blocks': 2}
            ]
        
        for params in param_combinations:
            try:
                model, metrics, _ = train_and_evaluate_deep_learning(
                    train_df.copy(), eval_df.copy(), feature_list, label_column, 
                    model_info['model'], params, label_encoder, random_state
                )
                
                score = metrics["Macro F1"]
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_metrics = metrics
                    best_params = params
                    
            except Exception as e:
                print(f"    Error with {model_name} params {params}: {str(e)}")
                continue
        
        return best_model, best_metrics, best_params
        
    else:
        # Handle traditional ML models (existing code)
        # Prepare data - keep your existing preprocessing
        for col in feature_list:
            train_df[col] = train_df[col].fillna(0)
            eval_df[col] = eval_df[col].fillna(0)
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
        
        X_train = train_df[feature_list]
        y_train = label_encoder.transform(train_df[label_column])
        X_eval = eval_df[feature_list]
        y_eval = label_encoder.transform(eval_df[label_column])
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)
        
        # Hyperparameter tuning with GridSearchCV
        model = model_info['model']
        param_grid = model_info['params']
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model and make predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_eval_scaled)
        
        # Evaluate using your existing function
        metrics = evaluate_predictions(y_eval, y_pred)
        
        return best_model, metrics, grid_search.best_params_

def run_comprehensive_baseline_study(df, subject_column='user_id', label_column='label'):
    """
    Run comprehensive baseline study keeping your existing feature format and combinations
    """
    #df = df.sample(n = 120)
    # Your existing feature sets - keep them exactly as they are
    battery_features = [f"battery_afternoon_{i}" for i in range(1, 15)]
    magnitude_max_feature = [f"magnitude_max_{i}" for i in range(1, 15)]
    screen_off_features = [f"screen_off_night_duration_{i}" for i in range(1, 15)]
    screen_use_features = [f"screen_use_duration_afternoon_{i}" for i in range(1, 15)]
    
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
    
    # Get all baseline models
    baseline_models = get_baseline_models()
    
    # Results storage
    all_results = []
    
    # Multiple random seeds for robust evaluation
    random_seeds = range(1, 6)
    
    print("Starting comprehensive baseline comparison...")
    print(f"Testing {len(baseline_models)} models on {len(feature_sets)} feature combinations across {len(random_seeds)} random seeds")
    print(f"Models include: Traditional ML ({[k for k,v in baseline_models.items() if v['type'] == 'traditional']}) and Deep Learning ({[k for k,v in baseline_models.items() if v['type'] == 'deep_learning']})")
    
    for rs in random_seeds:
        print(f"\nRandom seed: {rs}")
        
        # Split data using your existing function
        X_train_df, X_eval_df = split_data(df, subject_column, label_column, True, rs)
        
        # Encode labels using your existing approach
        label_encoder = LabelEncoder()
        df_label_list = list(X_train_df[label_column]) + list(X_eval_df[label_column])
        label_encoder.fit(df_label_list)
        
        for set_name, features in feature_sets.items():
            print(f"  Feature set: {set_name} ({len(features)} features)")
            
            for model_name, model_info in baseline_models.items():
                print(f"    Training {model_name}...")
                try:
                    model, metrics, best_params = train_and_evaluate_model(
                        X_train_df.copy(), X_eval_df.copy(), features, label_column, 
                        model_name, model_info, label_encoder, rs * 42
                    )
                    
                    if metrics is not None:
                        result = {
                            'random_seed': rs,
                            'model': model_name,
                            'model_type': model_info['type'],
                            'feature_set': set_name,
                            'num_features': len(features),
                            'accuracy': metrics["Accuracy"],
                            'macro_f1': metrics["Macro F1"],
                            'weighted_f1': metrics["Weighted F1"],
                            'macro_precision': metrics["Macro Precision"],
                            'macro_recall': metrics["Macro Recall"],
                            'weighted_precision': metrics["Weighted Precision"],
                            'weighted_recall': metrics["Weighted Recall"],
                            'Confusion Matrix': str(metrics["Confusion Matrix"]),
                            'best_params': str(best_params)  # Convert to string for CSV compatibility
                        }
                        
                        all_results.append(result)
                        print(f"      ✓ Macro F1: {metrics['Macro F1']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
                    else:
                        print(f"      ✗ No metrics returned")
                        
                except Exception as e:
                    print(f"      ✗ Error: {str(e)}")
                    continue
    
    # Convert to DataFrame and analyze results
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        # Calculate mean performance across seeds
        summary_stats = results_df.groupby(['model', 'feature_set']).agg({
            'accuracy': ['mean', 'std'],
            'macro_f1': ['mean', 'std'],
            'weighted_f1': ['mean', 'std'],
            'macro_precision': ['mean', 'std'],
            'macro_recall': ['mean', 'std'],
            'num_features': 'first'
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.reset_index()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BASELINE COMPARISON RESULTS")
        print("="*80)
        
        # Top performers by macro F1
        top_performers = summary_stats.nlargest(10, 'macro_f1_mean')
        print("\nTOP 10 PERFORMERS (by Macro F1):")
        print(top_performers[['model', 'feature_set', 'macro_f1_mean', 'macro_f1_std', 'accuracy_mean']].to_string(index=False))
        
        # Best model for each feature set
        print("\nBEST MODEL FOR EACH FEATURE SET:")
        best_per_feature = summary_stats.loc[summary_stats.groupby('feature_set')['macro_f1_mean'].idxmax()]
        print(best_per_feature[['feature_set', 'model', 'macro_f1_mean', 'accuracy_mean']].to_string(index=False))
        
        # Best feature set for each model
        print("\nBEST FEATURE SET FOR EACH MODEL:")
        best_per_model = summary_stats.loc[summary_stats.groupby('model')['macro_f1_mean'].idxmax()]
        print(best_per_model[['model', 'feature_set', 'macro_f1_mean', 'accuracy_mean']].to_string(index=False))
        
        return results_df, summary_stats
    
    else:
        print("No results generated. Check for errors in model training.")
        return None, None




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
df

results_df, summary_stats = run_comprehensive_baseline_study(df)


# Save results
if results_df is not None:
    results_df.to_csv('comprehensive_baseline_results_detailed.csv', index=False)
    summary_stats.to_csv('comprehensive_baseline_results_summary.csv', index=False)
    
    print(f"\nResults saved. Total experiments: {len(results_df)}")

