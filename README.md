# Advanced Time Series Forecasting with Deep Learning and Explainability

This project implements an end-to-end multivariate time series forecasting pipeline using LSTM and Transformer models. It includes rolling-origin cross-validation, uncertainty quantification, and model explainability using SHAP or Integrated Gradients. The system is designed to be production-ready with clean structure, documentation, and insights.

## 🚀 Project Features

### 1. Data Preparation
- Handles multivariate time series datasets
- Normalization, scaling, and missing value handling
- Creation of lag features and exogenous regressors
- Rolling-origin train/validation splits

### 2. Forecasting Models
- LSTM-based deep learning model
- Transformer encoder model for long-range dependencies
- Hyperparameter tuning with rolling-origin evaluation

### 3. Uncertainty Quantification
- Quantile regression (e.g., 0.1, 0.5, 0.9)
- OR Monte Carlo Dropout for probabilistic forecasting

### 4. Explainability
- SHAP values for feature/time-step importance
- Integrated Gradients for deep model attribution
- Insights about which features drive predictions

### 5. Evaluation Metrics
- MAE
- RMSE
- SMAPE
- MASE
- Prediction interval coverage
- Rolling-origin validation error curves

## 📊 Project Structure

