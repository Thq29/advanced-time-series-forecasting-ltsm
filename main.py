
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_multivariate(n_series=3, n_steps=600, seed=42):
    np.random.seed(seed)
    t = np.arange(n_steps)
    s0 = 0.02 * t + 2 * np.sin(2 * np.pi * t / 50) + 0.5 * np.random.randn(n_steps)
    s1 = 1.5 * np.sin(2 * np.pi * t / 12) + 0.3 * np.sin(2 * np.pi * t / 6) + 0.3 * np.random.randn(n_steps)
    exog = 0.5 * np.sin(2 * np.pi * t / 30) + 0.2 * np.random.randn(n_steps)
    s2 = 0.1 * t + 0.8 * exog + 0.4 * np.random.randn(n_steps)
    data = np.vstack([s0, s1, s2]).T
    cols = [f"series_{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    df['time'] = pd.date_range("2000-01-01", periods=n_steps, freq='D')
    df.set_index('time', inplace=True)
    return df

def create_lag_features(df, lags=20):
    df_feat = df.copy()
    for lag in range(1, lags+1):
        for col in df.columns:
            df_feat[f"{col}_lag{lag}"] = df[col].shift(lag)
    df_feat.dropna(inplace=True)
    return df_feat

def train_test_splits(df_feat, target_col='series_0', horizon=7, initial_train=350, step=50):
    n = len(df_feat)
    splits = []
    start = initial_train
    while start + horizon <= n:
        train_idx = list(range(start))
        val_idx = list(range(start, start+horizon))
        splits.append((train_idx, val_idx))
        start += step
    return splits

def mase(y_true, y_pred, y_train):
    # Mean Absolute Scaled Error (using naive one-step seasonal naive from training)
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum()/(n-1)
    return np.mean(np.abs(y_true - y_pred)) / d if d!=0 else np.nan

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom==0] = 1e-8
    return 100*np.mean(2*np.abs(y_pred - y_true)/denom)

# Keras LSTM builder
def build_lstm_model(input_shape, units=32):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, InputLayer, Dropout, Reshape
    except Exception as e:
        print("TensorFlow/Keras required to build model. Install tensorflow.")
        raise e
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(units, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Quantile loss for training separate models
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return np.mean(np.maximum(q*e, (q-1)*e))
    return loss

def evaluate_and_save(y_true, y_pred, y_train, prefix):
    metrics = {}
    metrics['MAE'] = float(mean_absolute_error(y_true, y_pred))
    metrics['RMSE'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics['sMAPE'] = float(smape(y_true, y_pred))
    metrics['MASE'] = float(mase(y_true, y_pred, y_train))
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics

def plot_forecast(y_true, y_pred, path):
    plt.figure(figsize=(8,3))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Pred')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def run_pipeline(run_train=True):
    print("Generating synthetic dataset...")
    df = generate_synthetic_multivariate(n_steps=600)
    df.to_csv(os.path.join(OUTPUT_DIR, "full_synthetic.csv"))
    df_feat = create_lag_features(df, lags=20)
    target = 'series_0'
    X = df_feat.drop(columns=[target]).values
    y = df_feat[target].values
    splits = train_test_splits(df_feat, target_col=target, horizon=7, initial_train=350, step=50)
    print(f"Prepared {len(splits)} rolling-origin splits.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "scaler_scale.npy"), scaler.scale_)

    # Persistence baseline (last observed)
    baseline_metrics = []
    for i, (train_idx, val_idx) in enumerate(splits):
        last_val = df[target].iloc[train_idx[-1]]
        y_true = df[target].iloc[val_idx].values
        y_pred = np.repeat(last_val, len(y_true))
        baseline_metrics.append(evaluate_and_save(y_true, y_pred, df[target].iloc[train_idx].values, f"baseline_split{i}"))
    with open(os.path.join(OUTPUT_DIR, "baseline_aggregate.json"), 'w') as f:
        json.dump(baseline_metrics, f, indent=2)

    if not run_train:
        print("Skipping model training as requested.")
        return

    # For speed: train a small LSTM on the last split available only (demonstration)
    last_train_idx, last_val_idx = splits[-1]
    X_train = Xs[last_train_idx]
    y_train = y[last_train_idx]
    X_val = Xs[last_val_idx]
    y_val = y[last_val_idx]

    # reshape for LSTM: (samples, timesteps=1, features)
    X_train_r = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_r = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    try:
        import tensorflow as tf
        tf.random.set_seed(42)
        model = build_lstm_model(input_shape=(X_train_r.shape[1], X_train_r.shape[2]), units=32)
        # small epochs for quick run; increase for better results
        model.fit(X_train_r, y_train, validation_data=(X_val_r, y_val), epochs=6, batch_size=32, verbose=0)
        y_pred = model.predict(X_val_r).ravel()
        metrics = evaluate_and_save(y_val, y_pred, y_train, "lstm_lastsplit")
        plot_forecast(y_val, y_pred, os.path.join(OUTPUT_DIR, "lstm_forecast.png"))
        model.save(os.path.join(OUTPUT_DIR, "lstm_model.h5"))
        print("Trained LSTM and saved outputs.")
    except Exception as e:
        print("TensorFlow training failed or not available:", e)
        # fallback: use simple linear persistence prediction
        y_pred = np.repeat(df[target].iloc[last_train_idx[-1]], len(last_val_idx))
        metrics = evaluate_and_save(y_val, y_pred, y_train, "lstm_fallback")
        plot_forecast(y_val, y_pred, os.path.join(OUTPUT_DIR, "lstm_fallback.png"))

    # Uncertainty: quick quantile approximation by training small separate models for 0.1,0.5,0.9 using pinball loss via Keras
    try:
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import InputLayer, LSTM, Dense
        def pinball_loss(q):
            def loss_fn(y_true, y_pred):
                e = y_true - y_pred
                return K.mean(K.maximum(q*e, (q-1)*e))
            return loss_fn
        qs = [0.1, 0.5, 0.9]
        preds_q = {}
        for q in qs:
            qmodel = Sequential([InputLayer(input_shape=(X_train_r.shape[1], X_train_r.shape[2])), LSTM(24), Dense(1)])
            qmodel.compile(optimizer='adam', loss=pinball_loss(q))
            qmodel.fit(X_train_r, y_train, epochs=4, batch_size=32, verbose=0)
            preds_q[q] = qmodel.predict(X_val_r).ravel()
        # save quantile predictions as CSV
        qdf = pd.DataFrame(preds_q)
        qdf.to_csv(os.path.join(OUTPUT_DIR, "quantile_predictions.csv"), index=False)
        # compute coverage of 0.1-0.9 interval
        lower = preds_q[0.1]
        upper = preds_q[0.9]
        coverage = np.mean((y_val >= lower) & (y_val <= upper))
        with open(os.path.join(OUTPUT_DIR, "quantile_coverage.json"), "w") as f:
            json.dump({"coverage_0.1_0.9": float(coverage)}, f, indent=2)
    except Exception as e:
        print("Quantile training skipped (tensorflow/pinball issues):", e)

    # SHAP explainability (quick KernelExplainer on a small sample)
    try:
        import shap
        # use a small background sample
        bg = X_train[:50]
        # use model.predict wrapper - if model exists
        def f(x):
            xr = x.reshape((x.shape[0], 1, x.shape[1]))
            return model.predict(xr).ravel()
        explainer = shap.KernelExplainer(f, bg)
        shap_vals = explainer.shap_values(X_val[:20], nsamples=50)
        # save a simple summary plot
        shap.summary_plot(shap_vals, pd.DataFrame(X_val[:20]), show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"))
        plt.close()
    except Exception as e:
        print("SHAP analysis skipped or failed:", e)

    # Save final metrics summary (if exists)
    print("Pipeline finished. Outputs saved in outputs/.")

if __name__ == "__main__":
    run_pipeline(run_train=True)
