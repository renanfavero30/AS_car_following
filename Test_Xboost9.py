# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 20:01:52 2025
@author: renanfavero
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ Load Model & Scalers
def load_model_and_scalers(model_type, results_folder):
    """
    Loads the trained model and scalers (if used) from the specified folder.

    Parameters:
    - model_type: Type of model (CNN, LSTM, Transformer, RF, LGBM, etc.)
    - results_folder: Path where the trained model and scalers are saved.

    Returns:
    - model: Loaded trained model
    - scaler_x: Loaded X scaler (or None if not used)
    - scaler_y: Loaded Y scaler (or None if not used)
    """
    model_filename = os.path.join(results_folder, f"best_model_{model_type}.pkl")
    model = joblib.load(model_filename)
    print(f"✅ Model loaded from: {model_filename}")

    # ✅ Load scalers only if they exist
    scaler_x_path = os.path.join(results_folder, f"scaler_X_{model_type}.pkl")
    scaler_y_path = os.path.join(results_folder, f"scaler_Y_{model_type}.pkl")

    scaler_x = joblib.load(scaler_x_path) if os.path.exists(scaler_x_path) else None
    scaler_y = joblib.load(scaler_y_path) if os.path.exists(scaler_y_path) else None

    print(f"✅ Scalers loaded: X - {scaler_x_path if scaler_x else 'Not used'}, Y - {scaler_y_path if scaler_y else 'Not used'}")

    return model, scaler_x, scaler_y


# ✅ Load & Preprocess Test Data
def load_test_data(test_file_name, variables_used):
    """
    Loads and preprocesses the test dataset, ensuring feature consistency.

    Parameters:
    - test_file_name: Path to the test dataset CSV file.
    - variables_used: List of feature names used in training.

    Returns:
    - df_test: Processed test DataFrame
    - x_test: Features matrix (unscaled)
    - y_test: Target values (unscaled)
    """
    df_test = pd.read_csv(test_file_name)
    print(f"✅ Test dataset loaded from: {test_file_name}")

    # ✅ Apply preprocessing: Feature selection, column renaming, and target alignment
    X_test = df_test.drop(columns=["Follower_acc"]).copy()
    X_test.columns = X_test.columns.str.replace(r"[\[\]]", "", regex=True)
    X_test = X_test.loc[:, X_test.columns.isin(variables_used)]

    # ✅ Ensure correct feature order before transformation
    expected_feature_order = [col for col in variables_used if col in X_test.columns]
    X_test = X_test[expected_feature_order]

    y_test = df_test["Follower_acc"].values.reshape(-1, 1)  # Convert y_test to NumPy array

    print("✅ Final features used for testing:", list(X_test.columns))
    return df_test, X_test, y_test


# ✅ Create Sequences (Only for CNN, LSTM, Transformer)
def create_sequences(X_scaled, y_scaled, input_size, horizon):
    """
    Creates sequences ensuring the predicted `y` aligns with `y_true` at `t+horizon`.
    X[t-4] ... X[t]  ---> Predict y[t+1] (for horizon=1)
    """
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - input_size - horizon + 1):
        X_seq.append(X_scaled[i:i + input_size])
        y_seq.append(y_scaled[i + input_size + horizon - 1])

    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)


# ✅ Make Predictions
def make_predictions(model, x_test, y_test, scaler_x, scaler_y, model_type, input_size=10, horizon=1):
    """
    Makes predictions using the trained model, ensuring input shape matches training.

    Parameters:
    - model: Trained ML model
    - x_test: Test dataset features (unscaled)
    - y_test: Target values (unscaled)
    - scaler_x: Scaler used for feature normalization
    - scaler_y: Scaler used for target normalization
    - model_type: Model type (CNN, LSTM, Transformer, etc.)
    - input_size: Number of past time steps used for training sequences (if applicable)
    - horizon: Prediction horizon (default=1 step ahead)

    Returns:
    - y_pred_unscaled: Unscaled predictions
    """

    # ✅ Ensure correct feature order
    expected_feature_order = scaler_x.feature_names_in_
    x_test = x_test[expected_feature_order]

    # ✅ Normalize test data
    x_test_scaled = scaler_x.transform(x_test)
    y_test_scaled = scaler_y.transform(y_test) if scaler_y else y_test

    # ✅ Convert test data into sequences (only for time-series models)
    if model_type in ["CNN", "LSTM", "Transformer"]:
        x_test_scaled, _ = create_sequences(x_test_scaled, y_test_scaled, input_size, horizon)
        

    print(f"✅ Reshaped x_test for prediction: {x_test_scaled.shape}")

    # ✅ Make predictions
    y_pred_scaled = model.predict(x_test_scaled)

    # ✅ Unscale predictions
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)) if scaler_y else y_pred_scaled

    return y_pred_unscaled, x_test_scaled, y_test_scaled

def save_predictions(df_test, y_pred_unscaled, results_folder, model_type, input_size=10):
    """
    Saves the predicted and true values in a CSV file.

    Parameters:
    - df_test: Test dataset DataFrame
    - y_pred_unscaled: Unscaled predicted values
    - results_folder: Path to save results
    - model_type: Type of model (CNN, Transformer, LSTM, RF, LGBM, etc.)
    - input_size: Number of past time steps used in sequence models (default = 10)
    """

    # ✅ Check if model requires sequence trimming
    sequence_models = ["CNN", "Transformer", "LSTM"]
    
    

    if model_type in sequence_models:
        df_test_trimmed = df_test.iloc[input_size:].reset_index(drop=True)  # Trim first `input_size` rows
        print(f"✅ Applied sequence trimming for {model_type}: Removed first {input_size} rows.")
    else:
        df_test_trimmed = df_test.copy()  # No trimming needed for RF, LGBM, etc.

    # ✅ Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        "Time_[s]": df_test_trimmed["Time_[s]"],
        "True Acceleration": df_test_trimmed["Follower_acc"],
        "Predicted Acceleration": y_pred_unscaled.flatten()
    })

    # ✅ Save predictions to CSV
    predictions_file = os.path.join(results_folder, f"predictions_test_{model_type}.csv")
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✅ Predictions saved at: {predictions_file}")




def simulate_vehicle_motion(df_test, model, results_folder, scaler_x, scaler_y, model_type, x_test_scaled, y_test_scaled, input_size=10, horizon=1):
    """
    Simulates vehicle motion using the trained model by iteratively predicting acceleration,
    updating speed, and computing position with realistic constraints.

    Returns:
    - df_simulated: DataFrame containing simulated features, speed, and position.
    """
    print(f"🚀 Starting Simulation for {model_type}")

    # ✅ Initialize dataset
    df_simulated = df_test.copy()
    
    is_sequential = model_type.lower() in ["lstm", "gru","cnn"]
    # ✅ Define all simulated columns
    simulated_columns = {
        "Simulated_acc": "Follower_acc",
        "Simulated_sp": "Follower_sp_[ft]",
        "Simulated_pos": "Follower_pos_[ft]",
        "Simulated_delta_s": "delta_s",
        "Simulated_delta_v": "delta_v",
        "Simulated_Follower_acc_t-1": "Follower_acc_t-1"
    }
    
   


    # ✅ Initialize simulated columns
    for sim_col, orig_col in simulated_columns.items():
        df_simulated[f"{sim_col}_{model_type}"] = df_simulated[orig_col]
    
    # ✅ Define feature columns dynamically based on model type
    if is_sequential:
        feature_columns = [f"Simulated_delta_s_{model_type}", f"Simulated_delta_v_{model_type}",
                           f"Simulated_sp_{model_type}", f"Simulated_Follower_acc_t-1_{model_type}"]
    else:
        feature_columns = [f"{col}_{model_type}" for col in simulated_columns.keys()]




    # ✅ Compute time step differences
    dt = df_simulated["Time_[s]"].diff().fillna(1).values

    # ✅ Define speed constraints
    max_speed = 22  # Maximum allowable speed
    min_speed = 0   # Minimum allowable speed (no reverse)

    # ✅ Define input features for model
    #feature_columns = [f"Simulated_delta_s_{model_type}", f"Simulated_delta_v_{model_type}",
                       #f"Simulated_sp_{model_type}", f"Simulated_Follower_acc_t-1_{model_type}"]
                       
    # ✅ Define mapping from simulated column names to original feature names
    feature_mapping = {
        f"Simulated_delta_s_{model_type}": "delta_s",
        f"Simulated_delta_v_{model_type}": "delta_v",
        f"Simulated_sp_{model_type}": "Follower_sp_[ft]",
        f"Simulated_Follower_acc_t-1_{model_type}": "Follower_acc_t-1",
    }

    # ✅ Simulation Loop
    for t in range(input_size, len(df_simulated) - horizon):  
        # Detect trajectory resets when delta_t > 2 or delta_t < 1
        if df_simulated.at[t, "delta_t"] > 1 or df_simulated.at[t, "delta_t"] < 1:  
            print(f"🔄 Resetting simulation at t={t}, new trajectory detected (delta_t={df_simulated.at[t, 'delta_t']}).")
            
            # ✅ Copy original values or use default (0.5 if missing)
            for sim_col, orig_col in simulated_columns.items():
                df_simulated.at[t, f"{sim_col}_{model_type}"] = (
                    df_simulated.at[t, orig_col] if pd.notna(df_simulated.at[t, orig_col]) else 0.5
                )
    
            continue  # Skip update since we initialize with real-world data

        # ✅ Prepare input for prediction
        if is_sequential:
            # Sequential model (LSTM/GRU) - Use past `input_size` steps as input
            past_window = []
            for i in range(t - input_size, t):  
                row = [df_simulated.at[i, col] for col in feature_columns]
                past_window.append(row)
            x_input = np.array(past_window).reshape(1, input_size, len(feature_columns))
            """
            print(f"Expected model input shape: {model.input_shape}")
            print(f"Actual x_input shape: {x_input.shape}")
            print(f"Expected model input shape: {model.input_shape}")  # Should be (None, 10, 4)
            print(f"Actual x_input shape: {x_input.shape}")  # Must be (1, 10, 4)
            print(f"Feature columns used: {feature_columns}")
            print(f"Number of features used: {len(feature_columns)}")"""

            # Non-sequential model (XGBoost, Random Forest) - Use latest simulated values
        else:
            # Remove columns that are not used for prediction
            valid_feature_columns = [col for col in feature_columns if col in feature_mapping]
            # Non-sequential model (LightGBM, RF) - Use latest simulated values
            temp_input = df_simulated.loc[[t], valid_feature_columns].copy()  # ✅ Extract row as DataFrame
            # ✅ Rename only existing columns back to original feature names
            temp_input = temp_input.rename(columns={col: feature_mapping[col] for col in valid_feature_columns})
            x_input = temp_input.values.reshape(1, -1)  # Convert to NumPy array


        # ✅ Make Prediction
        try:
            y_pred_scaled = model.predict(x_input)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)) if scaler_y else y_pred_scaled
        except Exception as e:
            print(f"❌ Prediction failed at t={t}: {e}")
                
        # ✅ Make Prediction
        try:
            y_pred_scaled = model.predict(x_input)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)) if scaler_y else y_pred_scaled
        except Exception as e:
            print(f"❌ Prediction failed at t={t}: {e}")
            continue
    
        # ✅ Get predicted acceleration
        pred_acc = y_pred.flatten()[0]
    
        # ✅ Get current speed
        current_speed = df_simulated.at[t, f"Simulated_sp_{model_type}"]
        
        # ✅ If speed is already at max, ensure acceleration does not increase it further
        if current_speed >= max_speed and pred_acc > 0:
            pred_acc = 0  # Prevent further acceleration if already at max speed
        # Ensure acceleration does not reverse the vehicle
        if current_speed <= min_speed and pred_acc < 0:
            pred_acc = 0  # Prevent negative acceleration from reversing the vehicle
    
        # ✅ Store the corrected acceleration
        df_simulated.at[t + horizon, f"Simulated_acc_{model_type}"] = pred_acc


        # ✅ Compute New Speed with Constraints
        dt_t = dt[t + horizon] if t + horizon < len(dt) else dt[-1]  # Ensure valid index
        new_speed = df_simulated.at[t, f"Simulated_sp_{model_type}"] + pred_acc * dt_t
        # 🚨 Apply Speed Constraints
        new_speed = max(0, min(new_speed, max_speed))  # Ensure speed never goes negative


        # ✅ Store the Corrected Speed
        df_simulated.at[t + horizon, f"Simulated_sp_{model_type}"] = new_speed

        # ✅ Compute New Position
        df_simulated.at[t + horizon, f"Simulated_pos_{model_type}"] = (
            df_simulated.at[t, f"Simulated_pos_{model_type}"] +
            new_speed * dt_t +  # Use corrected speed
            0.5 * pred_acc * (dt_t ** 2)
        )

        # ✅ Update Simulated Feature Columns for Next Step **Before Prediction**
        # ✅ Update delta_s and delta_v using the latest simulated values
        # ✅ Store the last predicted acceleration as Follower_acc_t-1 for the next step
        df_simulated.at[t + horizon, f"Simulated_Follower_acc_t-1_{model_type}"] = pred_acc
        leader_position = df_simulated.at[t + horizon, "Leader_pos_[ft]"] if (t + horizon) < len(df_simulated) else df_simulated.at[t, "Leader_pos_[ft]"]
        leader_speed = df_simulated.at[t + horizon, "Leader_sp_[ft]"] if (t + horizon) < len(df_simulated) else df_simulated.at[t, "Leader_sp_[ft]"]
        follower_position = df_simulated.at[t + horizon, f"Simulated_pos_{model_type}"]

        df_simulated.at[t + horizon , f"Simulated_delta_s_{model_type}"] = leader_position - follower_position
        df_simulated.at[t + horizon, f"Simulated_delta_v_{model_type}"] = leader_speed - new_speed

        # ✅ Debugging Outputs for Verification
        #print(f"✅ Updated at t={t + horizon}: acc={pred_acc:.4f}, sp={new_speed:.4f}, pos={df_simulated.at[t + horizon, f'Simulated_pos_{model_type}']:.4f}, delta_s={df_simulated.at[t + horizon, f'Simulated_delta_s_{model_type}']:.4f}, delta_v={df_simulated.at[t + horizon, f'Simulated_delta_v_{model_type}']:.4f}")

        # ✅ Save the updated simulated dataset
        simulated_file = os.path.join(r"C:\Users\renanfavero\OneDrive - University of Florida\My_research\Lake_Nona_2022\CSV_2", "dataset_test_simulation.csv")
        df_simulated.to_csv(simulated_file, index=False)
        #print(f"✅ Simulated results saved at: {simulated_file}")

    return df_simulated


def simulate_vehicle_motion(df_test, model, results_folder, scaler_x, scaler_y, model_type, x_test_scaled, y_test_scaled, input_size=5, horizon=1):
    """
    Simulates vehicle motion using the trained model by iteratively predicting acceleration,
    updating speed, and computing position with realistic constraints.

    Returns:
    - df_simulated: DataFrame containing simulated features, speed, and position.
    """
    print(f"🚀 Starting Simulation for {model_type}")

    # ✅ Create a copy of the test dataset to avoid modifying the original
    df_simulated = df_test.copy()
    
    is_sequential = model_type.lower() in ["lstm", "gru","cnn"]

    # ✅ Define all simulated columns (keeping prefixed names)
    simulated_columns = {
        f"Simulated_acc_{model_type}": "Follower_acc",
        f"Simulated_sp_{model_type}": "Follower_sp_[ft]",
        f"Simulated_pos_{model_type}": "Follower_pos_[ft]",
        f"Simulated_delta_s_{model_type}": "delta_s",
        f"Simulated_delta_v_{model_type}": "delta_v",
        f"Simulated_Follower_acc_t-1_{model_type}": "Follower_acc_t-1"
    }

    # ✅ Initialize simulated columns (copy original values)
    for sim_col, orig_col in simulated_columns.items():
        df_simulated[sim_col] = df_simulated[orig_col]

    # ✅ Define the mapping between simulated and original feature names for model input
    feature_mapping = {
        f"Simulated_delta_s_{model_type}": "delta_s",
        f"Simulated_delta_v_{model_type}": "delta_v",
        f"Simulated_sp_{model_type}": "Follower_sp_[ft]",
        f"Simulated_Follower_acc_t-1_{model_type}": "Follower_acc_t-1",
    }

    # ✅ Compute time step differences
    dt = df_simulated["Time_[s]"].diff().fillna(1).values

    # ✅ Define speed constraints
    max_speed = 22  # Maximum allowable speed
    min_speed = 0   # Minimum allowable speed (no reverse)

    # ✅ Simulation Loop
    for t in range(input_size, len(df_simulated) - horizon):  
        # Detect trajectory resets when delta_t > 1 or delta_t < 1
        if df_simulated.at[t, "delta_t"] > 1 or df_simulated.at[t, "delta_t"] < 1:  
            # ✅ Copy original values or use default (0.5 if missing)
            for sim_col, orig_col in simulated_columns.items():
                df_simulated.at[t, sim_col] = df_simulated.at[t, orig_col] if pd.notna(df_simulated.at[t, orig_col]) else 0.5
    
            continue  # ✅ Skip update for new trajectory
    
            print(f"🔄 Resetting simulation at t={t}, new trajectory detected.")
            continue  # Skip update for new trajectory

        # ✅ Prepare input for prediction
        if is_sequential:
            # Sequential model (LSTM/GRU) - Use past `input_size` steps as input
            past_window = []
            for i in range(t - input_size, t):  
                row = [df_simulated.at[i, sim_col] for sim_col in feature_mapping.keys()]
                past_window.append(row)
            x_input = np.array(past_window).reshape(1, input_size, len(feature_mapping))

        else:
            # Non-sequential model (LightGBM, RF) - Use latest simulated values
            x_input_df = df_simulated.loc[[t], feature_mapping.keys()].copy()
            x_input_df.rename(columns=feature_mapping, inplace=True)  # Rename to original feature names
            x_input = x_input_df.values

        # ✅ Debugging: Check Input Shape Before Prediction
        print(f"🟢 Predicting at t={t}: x_input.shape = {x_input.shape}")

        # ✅ Make Prediction
        try:
            y_pred_scaled = model.predict(x_input)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)) if scaler_y else y_pred_scaled
        except Exception as e:
            print(f"❌ Prediction failed at t={t}: {e}")
            continue

        # ✅ Get predicted acceleration
        pred_acc = y_pred.flatten()[0]

        # ✅ Get current speed
        current_speed = df_simulated.at[t, f"Simulated_sp_{model_type}"]
        
        # ✅ Apply speed constraints
        if current_speed >= max_speed and pred_acc > 0:
            pred_acc = 0  # Prevent further acceleration if at max speed
        if current_speed <= min_speed and pred_acc < 0:
            pred_acc = 0  # Prevent negative acceleration

        # ✅ Store the corrected acceleration
        df_simulated.at[t + horizon, f"Simulated_acc_{model_type}"] = pred_acc

        # ✅ Compute New Speed with Constraints
        dt_t = dt[t + horizon] if t + horizon < len(dt) else dt[-1]  # Ensure valid index
        new_speed = df_simulated.at[t, f"Simulated_sp_{model_type}"] + pred_acc * dt_t
        new_speed = max(0, min(new_speed, max_speed))  # Apply constraints

        # ✅ Store updated speed
        df_simulated.at[t + horizon, f"Simulated_sp_{model_type}"] = new_speed

        # ✅ Compute New Position
        df_simulated.at[t + horizon, f"Simulated_pos_{model_type}"] = (
            df_simulated.at[t, f"Simulated_pos_{model_type}"] +
            new_speed * dt_t +
            0.5 * pred_acc * (dt_t ** 2)
        )

        # ✅ Update delta_s and delta_v using the latest simulated values
        leader_position = df_simulated.at[t + horizon, "Leader_pos_[ft]"] if (t + horizon) < len(df_simulated) else df_simulated.at[t, "Leader_pos_[ft]"]
        leader_speed = df_simulated.at[t + horizon, "Leader_sp_[ft]"] if (t + horizon) < len(df_simulated) else df_simulated.at[t, "Leader_sp_[ft]"]
        follower_position = df_simulated.at[t + horizon, f"Simulated_pos_{model_type}"]

        df_simulated.at[t + horizon, f"Simulated_delta_s_{model_type}"] = leader_position - follower_position
        df_simulated.at[t + horizon, f"Simulated_delta_v_{model_type}"] = leader_speed - new_speed

        # ✅ Update Follower_acc_t-1 for the next step
        df_simulated.at[t + horizon, f"Simulated_Follower_acc_t-1_{model_type}"] = pred_acc

    # ✅ Save the updated simulated dataset
    
    simulated_file = os.path.join(rf"C:\Users\renanfavero\OneDrive - University of Florida\My_research\Lake_Nona_2022\CSV_2", "dataset_test_simulation.csv")
    df_simulated.to_csv(simulated_file, index=False)
    print(f"✅ Simulated results saved at: {simulated_file}")

    return df_simulated














def evaluate_model(df_test, y_pred_unscaled, results_folder, model_type, input_size=10):
    """
    Computes regression metrics and ensures consistent trimming for models that use sequences.

    Parameters:
    - df_test: Test dataset DataFrame
    - y_pred_unscaled: Unscaled predicted values
    - results_folder: Path to save results
    - model_type: Type of model (CNN, Transformer, LSTM, RF, LGBM, etc.)
    - input_size: Number of past time steps used in sequence models (default = 10)

    Returns:
    - df_test_trimmed: Trimmed test dataset
    - y_pred_trimmed: Trimmed predicted values
    """

    print(f"📏 Original df_test length: {len(df_test)}, y_pred_unscaled length: {len(y_pred_unscaled)}")

    # ✅ Trim first `input_size` rows only for sequence-based models
    sequence_models = ["CNN", "Transformer", "LSTM"]
    if model_type in sequence_models:
        df_test_trimmed = df_test.drop(index=range(input_size)).reset_index(drop=True)
        y_pred_trimmed = y_pred_unscaled # Trim predictions to match

        print(f"✅ Trimmed first {input_size} rows for {model_type} -> New df_test length: {len(df_test_trimmed)}")
    else:
        df_test_trimmed = df_test.copy()  # No trimming needed for RF, LGBM, etc.
        y_pred_trimmed = y_pred_unscaled.copy()

    # ✅ Check for NaNs
    print(f"🔍 NaN Check in df_test_trimmed: {df_test_trimmed.isna().sum().sum()} NaN values")
    print(f"🔍 NaN Check in y_pred_trimmed: {np.isnan(y_pred_trimmed).sum()} NaN values")

    if df_test_trimmed.isna().sum().sum() > 0:
        print("⚠️ Replacing NaNs in df_test_trimmed with column means.")
        df_test_trimmed.fillna(df_test_trimmed.mean(), inplace=True)

    if np.isnan(y_pred_trimmed).sum() > 0:
        print("⚠️ Replacing NaNs in y_pred_trimmed with mean prediction.")
        y_pred_trimmed = np.nan_to_num(y_pred_trimmed, nan=np.nanmean(y_pred_trimmed))

    # ✅ Ensure final lengths match exactly
    if len(df_test_trimmed) != len(y_pred_trimmed):
        raise ValueError(f"❌ Length mismatch after trimming: df_test_trimmed has {len(df_test_trimmed)} rows, "
                         f"but y_pred_trimmed has {len(y_pred_trimmed)}.")
    
    # ✅ Compute regression metrics
    mse = mean_squared_error(df_test_trimmed["Follower_acc"], y_pred_trimmed)
    mae = mean_absolute_error(df_test_trimmed["Follower_acc"], y_pred_trimmed)
    r2 = r2_score(df_test_trimmed["Follower_acc"], y_pred_trimmed)
    rmse = np.sqrt(mse)
    
    # ✅ Compute Normalized RMSE (NRMSE)
    nrmse = rmse / (df_test_trimmed["Follower_acc"].max() - df_test_trimmed["Follower_acc"].min())
    
    # ✅ Store metrics in a dictionary
    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R2 Score": r2,
        "RMSE": rmse,
        "NRMSE": nrmse
    }
    
    # ✅ Save metrics to file efficiently
    metrics_file = os.path.join(results_folder, f"metrics_test_{model_type}.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Model Type: {model_type}\n")
        f.write("\n".join(f"{key}: {value:.6f}" for key, value in metrics.items()) + "\n")
    
    # ✅ Print metrics to console
    print("\n📊 Predictions Metrics:")
    print("\n".join(f"{key}: {value:.6f}" for key, value in metrics.items()))


    print(f"✅ Test dataset metrics saved at: {metrics_file}")
    print(f"📏 Final df_test_trimmed length: {len(df_test_trimmed)}, y_pred_trimmed length: {len(y_pred_trimmed)}")
    
    # ✅ Create a new DataFrame with aligned predictions
    acc_column_name = f'acc_pred_{model_type} '
    df_test_trimmed[acc_column_name] = y_pred_trimmed.flatten()
    
   
    # ✅ Save the new DataFrame
    combined_file = os.path.join(results_folder, f"test_with_predictions.csv")
    df_test_trimmed.to_csv(combined_file, index=False)
    print(f"✅ New test dataset with predictions saved at: {combined_file}")


    return df_test_trimmed, y_pred_trimmed, combined_file


def plot_results(df_test, y_pred_unscaled, results_folder, model_type, input_size=10):
    """
    Plots and saves the Real vs. Predicted Acceleration and Histogram.

    Parameters:
    - df_test: Test dataset DataFrame
    - y_pred_unscaled: Unscaled predicted values
    - results_folder: Path to save results
    - model_type: Type of model (CNN, Transformer, LSTM, RF, LGBM, etc.)
    - input_size: Number of past time steps used in sequence models (default = 10)
    """

    print(f"📏 Initial df_test length: {len(df_test)}, y_pred_unscaled length: {len(y_pred_unscaled)}")
    
    # ✅ Ensure `results_folder` is a valid string
    if not isinstance(results_folder, str):
        results_folder = str(results_folder)  # Convert to string
        print(f"⚠️ Warning: `results_folder` was not a string, converted to: {results_folder}")

    # ✅ Ensure sequence-based models trim the first `input_size` rows
    sequence_models = ["CNN", "Transformer", "LSTM"]
    if model_type in sequence_models:
        #df_test_trimmed = df_test.drop(index=range(input_size)).reset_index(drop=True)
        df_test_trimmed = df_test.reset_index(drop=True)
        #y_pred_trimmed = y_pred_unscaled[input_size:].copy()
        y_pred_trimmed = y_pred_unscaled[:].copy()

        print(f"✅ Trimmed first {input_size} rows for {model_type} -> New df_test length: {len(df_test_trimmed)}, and y_pred_trimmed =  {len(y_pred_trimmed)} ")
    else:
        df_test_trimmed = df_test.copy()
        y_pred_trimmed = y_pred_unscaled.copy()

    # ✅ Ensure no NaNs before plotting
    if df_test_trimmed.isna().sum().sum() > 0:
        print("⚠️ Replacing NaNs in df_test_trimmed with column means.")
        df_test_trimmed.fillna(df_test_trimmed.mean(), inplace=True)

    if np.isnan(y_pred_trimmed).sum() > 0:
        print("⚠️ Replacing NaNs in y_pred_trimmed with mean prediction.")
        y_pred_trimmed = np.nan_to_num(y_pred_trimmed, nan=np.nanmean(y_pred_trimmed))

    # ✅ Ensure final lengths match before plotting
    if len(df_test_trimmed) != len(y_pred_trimmed):
        raise ValueError(f"❌ Length mismatch after trimming: df_test_trimmed has {len(df_test_trimmed)} rows, "
                         f"but y_pred_trimmed has {len(y_pred_trimmed)}.")

    # ✅ Define x-axis as a continuous sequence instead of `Time_[s]`
    x_axis = np.arange(len(df_test_trimmed))

    # ✅ Real vs. Predicted Acceleration Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, df_test_trimmed["Follower_acc"], label="True Acceleration", alpha=1)
    plt.plot(x_axis, y_pred_trimmed, label=f"Predicted Acceleration ({model_type})", linestyle="--", alpha=1)
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration (ft/s²)")
    plt.title(f"Comparison of Real vs Predicted Accelerations ({model_type})")
    plt.savefig(os.path.join(results_folder, f"real_vs_pred_test_{model_type}.png"))
    plt.show()  # ✅ Display the plot
    print(f"✅ Real vs. Predicted plot saved.")

    # ✅ Histogram of True vs. Predicted Acceleration
    plt.figure(figsize=(10, 5))
    plt.hist(df_test_trimmed["Follower_acc"], bins=30, alpha=0.5, label="True Acceleration")
    plt.hist(y_pred_trimmed, bins=30, alpha=0.5, label=f"Predicted Acceleration ({model_type})")
    plt.legend()
    plt.xlabel("Acceleration (ft/s²)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Real vs Predicted Acceleration ({model_type})")
    plt.savefig(os.path.join(results_folder, f"hist_real_vs_pred_test_{model_type}.png"))
    plt.show()  # ✅ Display the plot
    print(f"✅ Histogram plot saved.")

def plot_simulation_results(df_simulated, df_test, results_folder, model_type):
    """
    Plots and saves the comparison of Simulated vs. True values for Acceleration, Speed, and Position.

    Parameters:
    - df_simulated: DataFrame containing simulated acceleration, speed, and position.
    - df_test: Original test dataset DataFrame for comparison.
    - results_folder: Path to save results.
    - model_type: Type of model (CNN, Transformer, LSTM, RF, etc.).
    """

    print(f"📏 Initial df_simulated length: {len(df_simulated)}, df_test length: {len(df_test)}")



    # ✅ Ensure `results_folder` is a valid string
    if not isinstance(results_folder, str):
        results_folder = str(results_folder)
        print(f"⚠️ Warning: `results_folder` was not a string, converted to: {results_folder}")

    # ✅ Ensure no NaNs before plotting
    df_simulated.fillna(df_simulated.mean(), inplace=True)
    df_test.fillna(df_test.mean(), inplace=True)

    # ✅ Define x-axis as a continuous sequence instead of `Time_[s]`
    x_axis = np.arange(len(df_simulated))

    # ✅ Define simulated and true column names
    simulated_acc_col = f"Simulated_acc_{model_type}"
    simulated_sp_col = f"Simulated_sp_{model_type}"
    simulated_pos_col = f"Simulated_pos_{model_type}"

    true_acc_col = "Follower_acc"
    true_sp_col = "Follower_sp_[ft]"
    true_pos_col = "Follower_pos_[ft]"

    # 🚀 **Plot 1: Simulated vs. True Acceleration**
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, df_test[true_acc_col], label="True Acceleration", alpha=1)
    plt.plot(x_axis, df_simulated[simulated_acc_col], label=f"Simulated Acceleration ({model_type})", linestyle="--", alpha=1)
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration (ft/s²)")
    plt.title(f"Comparison of Simulated vs. True Acceleration ({model_type})")
    plt.savefig(os.path.join(results_folder, f"simulated_vs_true_acc_{model_type}.png"))
    plt.show()
    print(f"✅ Acceleration plot saved.")

    # 🚀 **Plot 2: Simulated vs. True Speed**
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, df_test[true_sp_col], label="True Speed", alpha=1)
    plt.plot(x_axis, df_simulated[simulated_sp_col], label=f"Simulated Speed ({model_type})", linestyle="--", alpha=1)
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Speed (ft/s)")
    plt.title(f"Comparison of Simulated vs. True Speed ({model_type})")
    plt.savefig(os.path.join(results_folder, f"simulated_vs_true_sp_{model_type}.png"))
    plt.show()
    print(f"✅ Speed plot saved.")

    # 🚀 **Plot 3: Simulated vs. True Position**
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, df_test[true_pos_col], label="True Position", alpha=1)
    plt.plot(x_axis, df_simulated[simulated_pos_col], label=f"Simulated Position ({model_type})", linestyle="--", alpha=1)
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Position (ft)")
    plt.title(f"Comparison of Simulated vs. True Position ({model_type})")
    plt.savefig(os.path.join(results_folder, f"simulated_vs_true_pos_{model_type}.png"))
    plt.show()
    print(f"✅ Position plot saved.")


def evaluate_simulation(df_simulated, df_test, results_folder, model_type, input_size=10):
    """
    Computes regression metrics for Simulated vs. True values for Acceleration, Speed, and Position.

    Parameters:
    - df_simulated: DataFrame containing simulated acceleration, speed, and position.
    - df_test: Original test dataset DataFrame for comparison.
    - results_folder: Path to save results.
    - model_type: Type of model (CNN, Transformer, LSTM, RF, etc.).
    - input_size: Number of past time steps used in sequence models (default = 10).

    Returns:
    - df_test_trimmed: Trimmed test dataset
    - df_simulated_trimmed: Trimmed simulated dataset
    - metrics_file: Path to saved metrics file
    """

    print(f"📏 Initial df_simulated length: {len(df_simulated)}, df_test length: {len(df_test)}")

    # ✅ Ensure `results_folder` is a valid string
    if not isinstance(results_folder, str):
        results_folder = str(results_folder)
        print(f"⚠️ Warning: `results_folder` was not a string, converted to: {results_folder}")

    # ✅ Trim first `input_size` rows only for sequence-based models
    sequence_models = ["CNN", "Transformer", "LSTM"]
    if model_type in sequence_models:
        df_simulated_trimmed = df_simulated.iloc[input_size:].reset_index(drop=True)
        df_test_trimmed = df_test.iloc[input_size:].reset_index(drop=True)
        print(f"✅ Trimmed first {input_size} rows for {model_type}")
    else:
        df_simulated_trimmed = df_simulated.copy()
        df_test_trimmed = df_test.copy()

    # ✅ Handle NaNs
    df_simulated_trimmed.fillna(df_simulated_trimmed.mean(), inplace=True)
    df_test_trimmed.fillna(df_test_trimmed.mean(), inplace=True)

    # ✅ Define column names for true vs. simulated values
    true_columns = {
        "Acceleration": "Follower_acc",
        "Speed": "Follower_sp_[ft]",
        "Position": "Follower_pos_[ft]"
    }

    simulated_columns = {
        "Acceleration": f"Simulated_acc_{model_type}",
        "Speed": f"Simulated_sp_{model_type}",
        "Position": f"Simulated_pos_{model_type}"
    }

    # ✅ Compute regression metrics for each variable
    metrics = {}
    for metric_name in ["Acceleration", "Speed", "Position"]:
        y_true = df_test_trimmed[true_columns[metric_name]]
        y_simulated = df_simulated_trimmed[simulated_columns[metric_name]]

        mse = mean_squared_error(y_true, y_simulated)
        mae = mean_absolute_error(y_true, y_simulated)
        r2 = r2_score(y_true, y_simulated)
        rmse = np.sqrt(mse)
        nrmse = rmse / (y_true.max() - y_true.min())

        

        metrics[metric_name] = {"MSE": mse, "MAE": mae, "R2": r2, "RMSE": rmse, "NRMSE": nrmse}

    # ✅ Save metrics to a text file
    metrics_file = os.path.join(results_folder, f"simulation_metrics_{model_type}.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Model Type: {model_type}\n\n")
        for metric_name, values in metrics.items():
            f.write(f"Mean Squared Error (MSE): {values['MSE']:.6f}\n")
            f.write(f"Mean Absolute Error (MAE): {values['MAE']:.6f}\n")
            f.write(f"R2 Score: {values['R2']:.6f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {values['RMSE']:.6f}\n")
            f.write(f"Normalized RMSE (NRMSE): {values['NRMSE']:.6f}\n\n")
            
            print("\n📊 Simulation Metrics:")
        for metric_name, values in metrics.items():
            print(f"\n=== {metric_name} ===")
            print("\n".join(f"{key}: {value:.6f}" for key, value in values.items()))
        

    print(f"✅ Simulation metrics saved at: {metrics_file}")

    # ✅ Save dataset with simulated vs. true values
    for metric_name in ["Acceleration", "Speed", "Position"]:
        df_simulated_trimmed[f"{metric_name}_True"] = df_test_trimmed[true_columns[metric_name]]

    combined_file = os.path.join(results_folder, f"simulated_vs_true_{model_type}.csv")
    df_simulated_trimmed.to_csv(combined_file, index=False)
    print(f"✅ Simulated results saved at: {combined_file}")

    return df_test_trimmed, df_simulated_trimmed, metrics_file


# ✅ Main Execution
def main():
    model_type = "XGboost"
    input_size=10
    results_folder = fr"C:\Users\renanfavero\OneDrive - University of Florida\My_research\Lake_Nona_2022\code_paper2\results_{model_type}"

    model, scaler_x, scaler_y = load_model_and_scalers(model_type, results_folder)

    test_file_name = r"C:\Users\renanfavero\OneDrive - University of Florida\My_research\Lake_Nona_2022\CSV_2\dataset_test_simulation.csv"
    variables_used = ['delta_s', 'delta_v', 'Follower_sp_ft', 'Follower_sp_[ft]', 'Follower_acc_t-1']

    # Load test dataset
    df_test, x_test, y_test = load_test_data(test_file_name, variables_used)

    # Make predictions
    y_pred_unscaled, x_test_scaled, y_test_scaled = make_predictions(model, x_test, y_test, scaler_x, scaler_y, model_type)

    # Save Predictions
    save_predictions(df_test, y_pred_unscaled, results_folder, model_type)
    
    # Evaluate model and save a new DF
    df_test_trimmed, y_pred_trimmed, combined_file = evaluate_model(df_test, y_pred_unscaled, results_folder, model_type,input_size)
    
    # Simulate 
    df_simulated = simulate_vehicle_motion(df_test, model, results_folder, scaler_x, scaler_y, model_type, x_test_scaled, y_test_scaled, input_size, horizon=1)

    # Plot, save and visualize predictions results
    plot_results(df_test_trimmed, y_pred_trimmed, results_folder, model_type, input_size)

    # Plot, save and visualize simulation results
    plot_simulation_results(df_simulated, df_test, results_folder, model_type)
    
    # Calculate simulation erros and save a new DF
    df_test_trimmed, df_simulated_trimmed, metrics_file = evaluate_simulation(df_simulated, df_test, results_folder, model_type, input_size)


    

    print(f"✅ All test results saved in: {results_folder}")


# ✅ Run Script
if __name__ == "__main__":
    main()
