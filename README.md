# Advanced Time Series Forecasting with Deep Learning and Explainability

This project contains a runnable example that trains a small LSTM on a synthetic multivariate dataset, evaluates with rolling-origin CV, computes uncertainty via quantile models (0.1,0.5,0.9), runs a basic SHAP analysis, and reports MAE, RMSE, sMAPE, and MASE. The pipeline is configured to run quickly for demonstration (small model, few epochs).

## Quick start
1. Create a venv and install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline (this will train small models and save outputs to `outputs/`):
```bash
python main.py
```

Results (metrics, plots, SHAP figures) will be in the `outputs/` folder.
