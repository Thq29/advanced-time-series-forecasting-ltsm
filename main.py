
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)
random.seed(42)

def generate_synthetic_multivariate(n_series=3, n_steps=800, seed=42):
    np.random.seed(seed)
    t = np.arange(n_steps)
    data = []
    # series 0: trend + yearly seasonality + noise
    s0 = 0.02 * t + 2 * np.sin(2 * np.pi * t / 50) + 0.5 * np.random.randn(n_steps)
    # series 1: seasonal with multiplicative noise
    s1 = 1.5 * np.sin(2 * np.pi * t / 12) + 0.3 * np.sin(2 * np.pi * t / 6) + 0.3 * np.random.randn(n_steps)
    # series 2: exogenous-driven
    exog = 0.5 * np.sin(2 * np.pi * t / 30) + 0.2 * np.random.randn(n_steps)
    s2 = 0.1 * t + 0.8 * exog + 0.4 * np.random.randn(n_steps)
    data = np.vstack([s0, s1, s2]).T
    columns = [f"series_{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    df['time'] = pd.date_range("2000-01-01", periods=n_steps, freq='D')
    df.set_index('time', inplace=True)
    return df

def create_lag_features(df, lags=30):
    df_feat = df.copy()
    for lag in range(1, lags+1):
        for col in df.columns:
            df_feat[f"{col}_lag{lag}"] = df[col].shift(lag)
    df_feat.dropna(inplace=True)
    return df_feat

def train_test_split_rolling(df, horizon=14, initial_train=500, step=50):
    # returns list of (train_idx, val_idx) tuples
    splits = []
    n = len(df)
    start = initial_train
    while start + horizon <= n:
        train_idx = list(range(start))
        val_idx = list(range(start, start+horizon))
        splits.append((train_idx, val_idx))
        start += step
    return splits

def build_simple_lstm(input_shape, units=32):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Reshape
    except Exception as e:
        print('TensorFlow not installed. LSTM model build will fail if executed.')
        raise e
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(units, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def quantile_loss(q, y, f):
    e = (y - f)
    return np.maximum(q*e, (q-1)*e).mean()

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": float(mae), "RMSE": float(rmse)}

def save_metrics(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_series(df, path):
    plt.figure(figsize=(10,4))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    print('Generating synthetic dataset...')
    df = generate_synthetic_multivariate(n_steps=800)
    plot_series(df, os.path.join(OUTPUT_DIR, 'series_plot.png'))

    # Prepare supervised dataset: predict series_0 horizon 14 using past 30 lags of all series
    horizon = 14
    lags = 30
    df_feat = create_lag_features(df, lags=lags)
    target = 'series_0'
    X = df_feat.drop(columns=[target])
    y = df_feat[target].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    splits = train_test_split_rolling(df_feat, horizon=horizon, initial_train=500, step=50)
    print(f'Rolling-origin splits: {len(splits)}')

    # Simple baseline: last observed value as forecast (persistence)
    persistence_metrics = []
    for i, (train_idx, val_idx) in enumerate(splits):
        # persistence forecast: use last value in training for the horizon
        last_val = df[target].iloc[train_idx[-1]]
        y_true = df[target].iloc[val_idx].values
        y_pred = np.repeat(last_val, len(y_true))
        m = evaluate_predictions(y_true, y_pred)
        persistence_metrics.append(m)
    save_metrics(persistence_metrics, os.path.join(OUTPUT_DIR, 'persistence_metrics.json'))

    # Provide Keras model training code (not executed here)
    model_code_path = os.path.join(OUTPUT_DIR, 'model_notes.txt')
    with open(model_code_path, 'w') as f:
        f.write('''
This repository includes example Keras model training code. To run full training, install TensorFlow and execute training functions.
Example outline:
- Build LSTM with sequence input shape (timesteps, features_per_step)
- Use early stopping and small epochs for student machine
- Use Monte Carlo Dropout or quantile loss for uncertainty
''')

    # Save a short CSV sample of the dataset
    df.head(200).to_csv(os.path.join(OUTPUT_DIR, 'sample_data.csv'))

    # Create a short report placeholder (the full report.md is provided separately)
    with open(os.path.join(OUTPUT_DIR, 'run_summary.txt'), 'w') as f:
        f.write('Lightweight run summary:\n- Synthetic dataset generated\n- Persistence baseline evaluated on rolling-origin splits\n- Saved metrics and sample data.\n')

    print('Project scaffold and lightweight evaluation complete. See outputs/ folder.')

if __name__ == '__main__':
    main()
