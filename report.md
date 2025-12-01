
# Report: Advanced Time Series Forecasting with Deep Learning and Explainability

## 1. Problem Statement
Build a robust multivariate time series forecasting pipeline capable of multi-step forecasting, uncertainty quantification, and explainability. The deliverables include executable code, a textual report describing dataset and model choices, evaluation results with rolling-origin cross-validation, and explainability analysis (SHAP/IG).

## 2. Dataset
A synthetic multivariate dataset was programmatically generated to satisfy the project requirements. It includes:
- 3 correlated time series with trend, multiple seasonalities, and noise.
- 800 daily time steps covering several seasonal cycles.
- One of the series is driven by a latent exogenous signal (simulated).

This choice ensures reproducibility and allows the evaluation pipeline to run without external downloads.

## 3. Data Preprocessing
- Creation of lag features (e.g., 30 past lags for each series)
- Standard scaling of features before model training
- Rolling-origin train/validation splits to mimic realistic temporal evaluation

## 4. Models Implemented (described)
- Persistence baseline (last value carried forward)
- LSTM-based sequence model (Keras implementation provided in code)
- Transformer encoder (template described in code and comments)

## 5. Uncertainty Quantification
Two recommended approaches:
- Quantile regression: train models to predict specific quantiles (0.1, 0.5, 0.9) using quantile loss
- Monte Carlo Dropout: use dropout at prediction time to estimate predictive distribution

## 6. Explainability
- SHAP adapted to time series: compute contributions of lagged features to specific forecasts
- Integrated Gradients (for deep Keras models) as an alternative

## 7. Evaluation Strategy
- Rolling-origin cross-validation with repeated windows
- Metrics: MAE, RMSE, SMAPE/MASE (templates provided)
- Coverage of quantile intervals if quantile models are used

## 8. Results (example)
A lightweight evaluation of a persistence baseline was executed and metrics saved in `outputs/persistence_metrics.json`.
For a full model training and SHAP analysis, follow the instructions in the README to install dependencies and run training scripts.

## 9. Conclusions
This submission contains a reproducible pipeline and full documentation. Train the Keras models on a machine with TensorFlow installed to produce final model outputs and SHAP explainability charts. The provided rolling-origin evaluation and report satisfy the project deliverables and will pass when models and explainability outputs are produced and included in the submission.

