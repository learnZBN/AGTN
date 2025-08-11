import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import warnings
from math import gcd
from functools import reduce
import seaborn as sns

# --- IMPORTANT: Ensure these libraries are installed ---
# If you encounter "ModuleNotFoundError: No module named 'xyz'",
# open your terminal/command prompt and run:
# pip install pandas numpy torch scikit-learn matplotlib chinese_calendar seaborn

# Try importing chinese_calendar
try:
    from chinese_calendar import is_holiday, is_workday
    _has_chinese_calendar = True
except ImportError:
    _has_chinese_calendar = False
    warnings.warn("chinese_calendar not found. Holiday/Weekend features might be limited.")

# Try importing torch_geometric for GCN optimization
try:
    from torch_geometric.nn import GCNConv
    _has_torch_geometric = True
except ImportError:
    _has_torch_geometric = False
    warnings.warn("torch_geometric not found. GCN model will use a simplified linear implementation.")

warnings.filterwarnings('ignore')

# --- Helper function for LCM ---
def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return (a * b) // gcd(a, b)

def calculate_lcm_of_list(numbers):
    if not numbers:
        return 1
    if len(numbers) == 1:
        return numbers[0]
    return reduce(lcm, numbers)

# --- 1. Configuration Parameters ---
class Config:
    INPUT_SEQUENCE_LENGTH = 168
    PREDICTION_HORIZON = 12

    LOAD_FEATURES = ['KW', 'CHWTON', 'HTmmBTU']

    AUX_FEATURES_WEATHER_ALL = [
        'Temperature', 'Wind Speed', 'Wind Direction', 'Surface Albedo',
        'Pressure', 'Precipitable Water', 'Solar Zenith Angle',
        'Relative Humidity', 'Dew Point', 'Cloud Type'
    ]
    CALENDAR_FEATURES = ['Day', 'Hour', 'Weekend', 'Holiday']

    CORRELATION_THRESHOLD = 0.4

    ALL_FEATURES = []
    NUM_NODES = 0
    NUM_LOAD_FEATURES = len(LOAD_FEATURES)

    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 8
    DROPOUT_RATE = 0.15
    SSU_BETA_INIT = 0.8

    BATCH_SIZE = 16
    EPOCHS = 500 # Consistent epochs for all models
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-5

    EMBED_DIM_INITIAL_GUESS = 384

    ACCUMULATION_STEPS = 2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'load2023.csv')

print(f"Current device: {Config.DEVICE}")

## 1. Data Preprocessing and Graph Construction

def load_and_preprocess_data():
    global global_scaler, load_features_scaler

    try:
        df = pd.read_csv(Config.DATA_PATH, parse_dates=['tstamp2'])
        print(f"Successfully loaded data. Number of rows: {len(df)}")

        if df.isnull().any().any():
            print("Missing values detected in data. Handling by filling with column mean...")
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            print("Missing values handled.")

        df['Month'] = df['tstamp2'].dt.month
        df['Day'] = df['tstamp2'].dt.day
        df['Hour'] = df['tstamp2'].dt.hour
        df['Weekend'] = (df['tstamp2'].dt.dayofweek >= 5).astype(int)
        if _has_chinese_calendar:
            df['Holiday'] = df['tstamp2'].apply(lambda x: 1 if is_holiday(x.date()) else 0)
        else:
            df['Holiday'] = 0
            print("Warning: 'chinese_calendar' not installed. Holiday feature will be all zeros.")

        print("\n--- Auxiliary Weather Feature Correlation Analysis and Selection ---")

        aux_weather_available = [f for f in Config.AUX_FEATURES_WEATHER_ALL if f in df.columns]

        selected_weather_features = []
        if not aux_weather_available:
            print("Warning: No auxiliary weather features available for correlation analysis.")
        else:
            relevant_cols_for_corr = Config.LOAD_FEATURES + aux_weather_available
            available_relevant_cols = [col for col in relevant_cols_for_corr if col in df.columns]
            if len(available_relevant_cols) > 1:
                corr_matrix = df[available_relevant_cols].corr().abs()

                print("Maximum Absolute Correlation of Each Weather Feature with Any Load (for filtering):")
                for aux_f in aux_weather_available:
                    if aux_f in corr_matrix.index:
                        max_corr_with_any_load = corr_matrix.loc[aux_f, Config.LOAD_FEATURES].max()
                        print(f"  {aux_f}: {max_corr_with_any_load:.4f}")

                        if max_corr_with_any_load >= Config.CORRELATION_THRESHOLD:
                            selected_weather_features.append(aux_f)
                    else:
                        print(f"  {aux_f}: Column not found, skipping correlation analysis.")

                if not selected_weather_features:
                    print(
                        f"Warning: No weather features found with max absolute correlation >= {Config.CORRELATION_THRESHOLD}.")
                else:
                    print(
                        f"Filtered Weather Features (max correlation >= {Config.CORRELATION_THRESHOLD}): {selected_weather_features}")
            else:
                print("Warning: Fewer than 2 columns available for correlation analysis, skipping.")

        Config.ALL_FEATURES = Config.LOAD_FEATURES + selected_weather_features + Config.CALENDAR_FEATURES

        Config.ALL_FEATURES = [f for f in Config.ALL_FEATURES if f in df.columns]

        Config.NUM_NODES = len(Config.ALL_FEATURES)

        if Config.NUM_NODES == 0 or Config.NUM_HEADS == 0:
            raise ValueError("NUM_NODES or NUM_HEADS cannot be zero. Check features and configuration.")

        lcm_of_nodes_heads = calculate_lcm_of_list([Config.NUM_NODES, Config.NUM_HEADS])

        if Config.EMBED_DIM_INITIAL_GUESS % lcm_of_nodes_heads == 0:
            Config.EMBED_DIM = Config.EMBED_DIM_INITIAL_GUESS
        else:
            Config.EMBED_DIM = ((Config.EMBED_DIM_INITIAL_GUESS // lcm_of_nodes_heads) + 1) * lcm_of_nodes_heads

        print(
            f"Based on filtered number of nodes ({Config.NUM_NODES}) and number of attention heads ({Config.NUM_HEADS}), model embedding dimension EMBED_DIM set to: {Config.EMBED_DIM}")

        df_processed = df[Config.ALL_FEATURES].copy()

        print("\n--- Final Feature List and Ranges ---")
        for col in Config.ALL_FEATURES:
            min_val = df_processed[col].min()
            max_val = df_processed[col].max()
            print(f"Min value of column '{col}': {min_val:.2f}, Max value: {max_val:.2f}")

        global_scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(global_scaler.fit_transform(df_processed), columns=Config.ALL_FEATURES,
                                     index=df_processed.index)

        load_features_data_original = df_processed[Config.LOAD_FEATURES].values
        load_features_scaler = MinMaxScaler()
        load_features_scaler.fit(load_features_data_original)

        data = torch.tensor(df_normalized.values, dtype=torch.float32).to(Config.DEVICE)

        print(f"Total number of features (nodes): {Config.NUM_NODES}")
        print(f"Final feature list: {Config.ALL_FEATURES}")

        class TimeSeriesDataset(Dataset):
            def __init__(self, data, input_seq_len, pred_horizon, target_features_idx):
                self.data = data
                self.input_seq_len = input_seq_len
                self.pred_horizon = pred_horizon
                self.target_features_idx = target_features_idx

            def __len__(self):
                return len(self.data) - self.input_seq_len - self.pred_horizon + 1

            def __getitem__(self, idx):
                encoder_input = self.data[idx: idx + self.input_seq_len]
                decoder_target = self.data[
                                 idx + self.input_seq_len: idx + self.input_seq_len + Config.PREDICTION_HORIZON,
                                 self.target_features_idx]
                return encoder_input, decoder_target

        target_feature_indices = [Config.ALL_FEATURES.index(f) for f in Config.LOAD_FEATURES]
        dataset = TimeSeriesDataset(data, Config.INPUT_SEQUENCE_LENGTH, Config.PREDICTION_HORIZON,
                                    target_feature_indices)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=True)

        print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

        # Calculate feature correlation matrix for heatmap
        corr_matrix = df_processed.corr()

        return train_dataloader, test_dataloader, load_features_scaler, df_normalized, target_feature_indices, df_processed, corr_matrix

    except FileNotFoundError:
        print(f"Error: File '{Config.DATA_PATH}' not found. Please ensure the file path is correct.")
        exit()
    except Exception as e:
        print(f"Error during data loading or preprocessing: {e}")
        exit()



# --- 3. Training and Evaluation Process ---

def train_and_evaluate_pytorch_model(model_name, model_instance, train_dataloader, test_dataloader, criterion,
                                     load_scaler_for_metrics, epochs=Config.EPOCHS, learning_rate=Config.LEARNING_RATE,
                                     weight_decay=Config.WEIGHT_DECAY, accumulation_steps=Config.ACCUMULATION_STEPS):
    print(f"\n--- Training and Evaluating PyTorch Model: {model_name} ---")

    optimizer = torch.optim.AdamW(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=epochs // 5,
        T_mult=2,
        eta_min=1e-6
    )

    best_val_loss = float('inf')

    current_model = model_instance

    for epoch in range(epochs):
        current_model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, (encoder_input, decoder_target) in enumerate(train_dataloader):
            encoder_input = encoder_input.to(Config.DEVICE)
            decoder_target = decoder_target.to(Config.DEVICE)

            predictions = current_model(encoder_input)
            loss = criterion(predictions, decoder_target)
            loss = loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps

        if (len(train_dataloader) % accumulation_steps != 0) and (len(train_dataloader) > 0):
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_dataloader)

        current_model.eval()
        total_val_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for encoder_input_val, decoder_target_val in test_dataloader:
                encoder_input_val = encoder_input_val.to(Config.DEVICE)
                decoder_target_val = decoder_target_val.to(Config.DEVICE)

                predictions_val = current_model(encoder_input_val)
                loss_val = criterion(predictions_val, decoder_target_val)
                total_val_loss += loss_val.item()

                all_predictions.append(predictions_val.cpu().numpy())
                all_targets.append(decoder_target_val.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_dataloader)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = f'best_{model_name.lower().replace(" ", "_").replace("-", "_")}_model.pth'
            torch.save(model_instance.state_dict(), model_save_path)

    print(f"--- {model_name} training complete ---")

    # Load best model for final evaluation
    model_save_path = f'best_{model_name.lower().replace(" ", "_").replace("-", "_")}_model.pth'
    if os.path.exists(model_save_path):
        model_instance.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE))
    else:
        print(f"Warning: Best model for {model_name} not found at {model_save_path}. Using current model state.")

    model_instance.eval()

    final_predictions_list = []
    final_targets_list = []
    with torch.no_grad():
        for encoder_input_final, decoder_target_final in test_dataloader:
            encoder_input_final = encoder_input_final.to(Config.DEVICE)
            predictions_final = model_instance(encoder_input_final)
            final_predictions_list.append(predictions_final.cpu().numpy())
            final_targets_list.append(decoder_target_final.cpu().numpy())

    final_predictions = np.concatenate(final_predictions_list, axis=0)
    final_targets = np.concatenate(final_targets_list, axis=0)

    metrics, predictions_rescaled, targets_rescaled, errors = calculate_metrics(
        final_targets, final_predictions, load_scaler_for_metrics, Config.NUM_LOAD_FEATURES
    )

    # For AGTN, also return the last learned adjacency matrix
    if isinstance(model_instance, AGTN):
        return metrics, predictions_rescaled, targets_rescaled, errors, model_instance.graph_learner.last_adj_matrix
    else:
        return metrics, predictions_rescaled, targets_rescaled, errors, None


def main():
    print("--- Starting Data Loading and Preprocessing ---")
    train_dataloader, test_dataloader, load_features_scaler, df_normalized, target_feature_indices, df_processed, corr_matrix = load_and_preprocess_data()

    # Collect all test data and targets for comprehensive evaluation
    all_test_inputs = []
    all_test_targets = []
    for encoder_input, decoder_target in test_dataloader:
        all_test_inputs.append(encoder_input)
        all_test_targets.append(decoder_target)
    all_test_inputs_tensor = torch.cat(all_test_inputs, dim=0)
    all_test_targets_tensor = torch.cat(all_test_targets, dim=0)

    print("\n--- Data Loading and Preprocessing Complete ---")
    print(f"Training set batch count: {len(train_dataloader)}")
    print(f"Test set batch count: {len(test_dataloader)}")

    # --- Initialize and Train All Models ---
    all_models_results = {}
    all_models_predictions_rescaled = {}
    all_models_targets_rescaled = {}  # This will store the consistent actual targets
    all_models_errors = {} # To store errors for box plots
    agtn_learned_adj_matrix = None  # To store AGTN's learned adjacency matrix

    criterion = nn.MSELoss()

    # --- AGTN Model Optimization (Graph-First Only) ---
    agtn_model = AGTN(
        num_nodes=Config.NUM_NODES,
        input_dim=Config.NUM_NODES,
        embed_dim=Config.EMBED_DIM,
        num_heads=Config.NUM_HEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_load_features=Config.NUM_LOAD_FEATURES,
        input_seq_len=Config.INPUT_SEQUENCE_LENGTH,
        pred_horizon=Config.PREDICTION_HORIZON,
        dropout_rate=Config.DROPOUT_RATE,
        graph_first=True
    ).to(Config.DEVICE)
    metrics, predictions_rescaled, targets_rescaled, errors, adj_matrix = train_and_evaluate_pytorch_model(
        "AGTN", agtn_model, train_dataloader, test_dataloader, criterion, load_features_scaler
    )
    all_models_results["AGTN"] = metrics
    all_models_predictions_rescaled["AGTN"] = predictions_rescaled
    all_models_targets_rescaled["AGTN"] = targets_rescaled  # Store actual targets from AGTN run
    all_models_errors["AGTN"] = errors
    agtn_learned_adj_matrix = adj_matrix  # Store the adjacency matrix

    # --- Other Baseline Models ---
    # The 'targets_rescaled' returned from 'train_and_evaluate_pytorch_model' for baselines
    # will be a rescaled version of the test_dataloader targets, which is consistent.
    # We will still use all_models_targets_rescaled["AGTN"] for plotting consistency.

    # 1. LSTM Model
    lstm_model = LSTM_Model(
        input_dim=Config.NUM_NODES,
        hidden_dim=128,
        output_dim=Config.PREDICTION_HORIZON * Config.NUM_LOAD_FEATURES,
        num_layers=2,
        dropout=0.2
    ).to(Config.DEVICE)
    metrics, predictions_rescaled, _, errors, _ = train_and_evaluate_pytorch_model(
        "LSTM", lstm_model, train_dataloader, test_dataloader, criterion, load_features_scaler
    )
    all_models_results["LSTM"] = metrics
    all_models_predictions_rescaled["LSTM"] = predictions_rescaled
    all_models_errors["LSTM"] = errors

    # 2. GCN Model (Simplified/Optimized)
    gcn_model = GCN_Model(
        input_dim=Config.NUM_NODES,
        hidden_dim=64,
        output_dim=Config.PREDICTION_HORIZON * Config.NUM_LOAD_FEATURES,
        num_nodes=Config.NUM_NODES,
        dropout=0.2
    ).to(Config.DEVICE)
    metrics, predictions_rescaled, _, errors, _ = train_and_evaluate_pytorch_model(
        "GCN", gcn_model, train_dataloader, test_dataloader, criterion, load_features_scaler
    )
    all_models_results["GCN"] = metrics
    all_models_predictions_rescaled["GCN"] = predictions_rescaled
    all_models_errors["GCN"] = errors

    # 3. Transformer Model
    transformer_model = Transformer_Model(
        input_dim=Config.NUM_NODES,
        embed_dim=Config.EMBED_DIM // 2,
        num_heads=Config.NUM_HEADS // 2 if Config.NUM_HEADS >= 2 else 1,
        num_layers=Config.NUM_ENCODER_LAYERS // 2 if Config.NUM_ENCODER_LAYERS >= 2 else 1,
        output_dim=Config.PREDICTION_HORIZON * Config.NUM_LOAD_FEATURES,
        max_seq_len=Config.INPUT_SEQUENCE_LENGTH,
        dropout=0.3
    ).to(Config.DEVICE)
    metrics, predictions_rescaled, _, errors, _ = train_and_evaluate_pytorch_model(
        "Transformer", transformer_model, train_dataloader, test_dataloader, criterion, load_features_scaler,
        epochs=Config.EPOCHS # Set to Config.EPOCHS for consistency
    )
    all_models_results["Transformer"] = metrics
    all_models_predictions_rescaled["Transformer"] = predictions_rescaled
    all_models_errors["Transformer"] = errors

    # 4. SmartFormer Model
    smartformer_model = SmartFormer_Model(
        input_dim=Config.NUM_NODES,
        embed_dim=Config.EMBED_DIM // 2,
        num_heads=Config.NUM_HEADS // 2 if Config.NUM_HEADS >= 2 else 1,
        num_layers=Config.NUM_ENCODER_LAYERS // 2 if Config.NUM_ENCODER_LAYERS >= 2 else 1,
        output_dim=Config.PREDICTION_HORIZON * Config.NUM_LOAD_FEATURES,
        max_seq_len=Config.INPUT_SEQUENCE_LENGTH,
        num_nodes=Config.NUM_NODES,
        dropout=0.3
    ).to(Config.DEVICE)
    metrics, predictions_rescaled, _, errors, _ = train_and_evaluate_pytorch_model(
        "SmartFormer", smartformer_model, train_dataloader, test_dataloader, criterion, load_features_scaler,
        epochs=Config.EPOCHS # Set to Config.EPOCHS for consistency
    )
    all_models_results["SmartFormer"] = metrics
    all_models_predictions_rescaled["SmartFormer"] = predictions_rescaled
    all_models_errors["SmartFormer"] = errors

    # --- Print Summary Table ---
    print("\n" + "=" * 100)
    print(" " * 40 + "Model Performance Comparison")
    print("=" * 100)

    # Prepare header
    header_parts = ["Model"]
    for feature in Config.LOAD_FEATURES:
        header_parts.extend([f"{feature} MAE", f"{feature} RMSE", f"{feature} MAPE"])

    # Calculate desired width for each column (e.g., based on max content length)
    max_model_name_len = max(len(name) for name in all_models_results.keys())
    metric_value_max_len = 10
    mape_value_max_len = 10

    col_widths = {
        "Model": max(max_model_name_len + 2, len("Model") + 2),
    }
    for feature in Config.LOAD_FEATURES:
        col_widths[f"{feature} MAE"] = max(len(f"{feature} MAE"), metric_value_max_len)
        col_widths[f"{feature} RMSE"] = max(len(f"{feature} RMSE"), metric_value_max_len)
        col_widths[f"{feature} MAPE"] = max(len(f"{feature} MAPE"), mape_value_max_len)

    # Print formatted header
    header_line_str = f"{'Model':<{col_widths['Model']}}"
    for i, feature in enumerate(Config.LOAD_FEATURES):
        header_line_str += f"| {'MAE':<{col_widths[f'{feature} MAE']}} | {'RMSE':<{col_widths[f'{feature} RMSE']}} | {'MAPE':<{col_widths[f'{feature} MAPE']}} |"
    print(header_line_str)
    print("-" * len(header_line_str))

    # Print feature sub-headers
    feature_subheader_str = f"{'':<{col_widths['Model']}}"
    for i, feature_name in enumerate(Config.LOAD_FEATURES):
        feature_subheader_str += f"| {feature_name:<{col_widths[f'{feature_name} MAE']}s} | {feature_name:<{col_widths[f'{feature_name} RMSE']}s} | {feature_name:<{col_widths[f'{feature_name} MAPE']}s} |"
    print(feature_subheader_str)
    print("-" * len(header_line_str))

    # Updated model order for table
    model_order_for_table = [
        "AGTN",
        "LSTM", "GCN", "Transformer", "SmartFormer"
    ]

    for model_name in model_order_for_table:
        if model_name not in all_models_results:
            continue

        results = all_models_results[model_name]
        row_parts = [f"{model_name:<{col_widths['Model']}}"]

        for feature in Config.LOAD_FEATURES:
            mae = results[feature]['MAE']
            rmse = results[feature]['RMSE']
            mape = results[feature]['MAPE']

            mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"

            row_parts.append(f"{mae:<{col_widths[f'{feature} MAE']}.2f}")
            row_parts.append(f"{rmse:<{col_widths[f'{feature} RMSE']}.2f}")
            row_parts.append(f"{mape_str:<{col_widths[f'{feature} MAPE']}}")

        print(" | ".join(row_parts))
    print("=" * 100)

    # --- Generate Plots ---
    print("\n--- Generating Prediction vs. Actual Plots and Other Visualizations ---")

    # Determine the total hours to plot for a 'month period' (e.g., 7 days * 24 hours)
    plot_duration_hours = 7 * 24
    num_prediction_batches_to_plot = plot_duration_hours // Config.PREDICTION_HORIZON

    # Ensure we don't exceed available test samples for plotting
    max_test_samples_available_agtn = all_models_targets_rescaled["AGTN"].shape[0]
    num_prediction_batches_to_plot = min(num_prediction_batches_to_plot, max_test_samples_available_agtn)

    # Use AGTN's actual data as the ground truth for all plots
    plot_actual_data_base = all_models_targets_rescaled["AGTN"][:num_prediction_batches_to_plot]

    plot_predicted_data_dict = {}

    # Define the models to include in plots
    models_for_plotting = ["AGTN", "LSTM", "GCN", "Transformer", "SmartFormer"]

    for model_name in models_for_plotting:
        if model_name in all_models_predictions_rescaled:
            current_model_samples = all_models_predictions_rescaled[model_name].shape[0]
            current_num_prediction_batches_to_plot = min(num_prediction_batches_to_plot, current_model_samples)

            # Slice the collected predictions for the current model
            plot_predicted_data_dict[model_name] = \
                all_models_predictions_rescaled[model_name][:current_num_prediction_batches_to_plot]
        else:
            print(f"Warning: Prediction data for model {model_name} not available, skipping its line plot.")

    # Line Plots for each Load Feature over the selected duration
    for i, load_name in enumerate(Config.LOAD_FEATURES):
        fig_line, ax_line = plt.subplots(figsize=(20, 7))

        # Plot actual values (consistent across all models)
        actual_plot_data = plot_actual_data_base[:, :, i].flatten()
        ax_line.plot(np.arange(len(actual_plot_data)),
                     actual_plot_data,
                     label=f'Actual ({load_name})', color='black', linewidth=2, marker='o',
                     markersize=3, markevery=Config.PREDICTION_HORIZON)

        # Define consistent color and style palette for models
        model_colors = {
            "AGTN": "dodgerblue",
            "LSTM": "blue",
            "GCN": "purple",
            "Transformer": "orange",
            "SmartFormer": "brown"
        }
        model_linestyles = {
            "AGTN": "-",
            "LSTM": ":",
            "GCN": "--",
            "Transformer": "-.",
            "SmartFormer": ":"
        }
        model_markers = {
            "AGTN": "o",
            "LSTM": "D",
            "GCN": "P",
            "Transformer": "*",
            "SmartFormer": "X"
        }
        agtn_line_width = 3
        agtn_marker_size = 6

        # Plot predicted values for each model
        for model_name in models_for_plotting:
            if model_name in plot_predicted_data_dict:
                predicted_plot_data = plot_predicted_data_dict[model_name][:, :, i].flatten()

                if len(predicted_plot_data) > len(actual_plot_data):
                    predicted_plot_data = predicted_plot_data[:len(actual_plot_data)]
                elif len(predicted_plot_data) < len(actual_plot_data):
                    print(
                        f"Warning: Predicted data for {model_name} is shorter than actual data for plotting. Skipping.")
                    continue

                if model_name == "AGTN":
                    ax_line.plot(np.arange(len(actual_plot_data)), predicted_plot_data,
                                 label=f'Predicted ({model_name})',
                                 color=model_colors[model_name],
                                 linestyle=model_linestyles[model_name],
                                 linewidth=agtn_line_width,
                                 marker=model_markers[model_name],
                                 markersize=agtn_marker_size,
                                 alpha=0.8,
                                 markevery=Config.PREDICTION_HORIZON)
                else:
                    ax_line.plot(np.arange(len(actual_plot_data)), predicted_plot_data,
                                 label=f'Predicted ({model_name})',
                                 color=model_colors[model_name],
                                 linestyle=model_linestyles[model_name],
                                 marker=model_markers[model_name],
                                 markersize=3,
                                 alpha=0.7,
                                 markevery=Config.PREDICTION_HORIZON)

        # --- Y-axis density control & dynamic range ---
        all_y_values = np.concatenate(
            [actual_plot_data] + [plot_predicted_data_dict[m][:, :, i].flatten() for m in models_for_plotting if
                                  m in plot_predicted_data_dict and len(
                                      plot_predicted_data_dict[m][:, :, i].flatten()) == len(actual_plot_data)])
        if all_y_values.size > 0:  # Check if array is not empty
            min_y = np.min(all_y_values) * 0.95
            max_y = np.max(all_y_values) * 1.05
            ax_line.set_ylim([min_y, max_y])
        ax_line.yaxis.set_major_locator(MaxNLocator(nbins=6))

        ax_line.set_title(
            f'{load_name} Prediction vs. Actual (Test Period: {plot_duration_hours} hours forecast)',
            fontsize=16)
        ax_line.set_xlabel('Prediction Time Step (hours)', fontsize=12)
        ax_line.set_ylabel(f'{load_name} Value', fontsize=12)
        ax_line.legend(fontsize=10)
        ax_line.grid(True)
        plt.tight_layout()
        save_path_line_plot = f'line_plot_{load_name}_test_period_{plot_duration_hours}h.png'
        plt.savefig(save_path_line_plot)
        plt.show()
        plt.close(fig_line)
        print(f"Line plot saved to: {save_path_line_plot}")

    # --- Box Plots: Multi-Model Error Comparison ---
    print("\n--- Prediction Error Box Plots ---")
    for load_name in Config.LOAD_FEATURES:
        fig_box, ax_box = plt.subplots(figsize=(12, 7))
        errors_to_plot = {model: all_models_errors[model][load_name] for model in models_for_plotting if
                          model in all_models_errors}
        sns.boxplot(data=list(errors_to_plot.values()), ax=ax_box, palette="deep")
        ax_box.set_xticklabels(list(errors_to_plot.keys()))
        ax_box.set_title(f'Prediction Error Distribution for {load_name}', fontsize=16)
        ax_box.set_xlabel('Model', fontsize=12)
        ax_box.set_ylabel(f'Error ({load_name})', fontsize=12)
        ax_box.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path_box_plot = f'boxplot_error_comparison_{load_name}.png'
        plt.savefig(save_path_box_plot)
        plt.show()
        plt.close(fig_box)
        print(f"Box plot saved to: {save_path_box_plot}")

    # --- Feature Correlation Heatmap ---
    print("\n--- Feature Correlation Heatmap Visualization ---")
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    save_path_corr_matrix = 'feature_correlation_matrix.png'
    plt.savefig(save_path_corr_matrix)
    plt.show()
    plt.close()
    print(f"Feature correlation heatmap saved to: {save_path_corr_matrix}")

    print("\n--- Program Execution Summary ---")
    print("All model evaluations and visualizations completed!")


if __name__ == "__main__":
    main()
