# Advanced Time Series Forecasting with Deep Learning and Explainability

This project implements an end-to-end multivariate time series forecasting pipeline using LSTM and Transformer models. It includes rolling-origin cross-validation, uncertainty quantification (quantile regression + MC Dropout), and model explainability using SHAP or Integrated Gradients. The system is designed to be production-ready with clear structure and documentation.

## How to use
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main pipeline (generates synthetic dataset, trains small models, and saves outputs):
   ```bash
   python main.py
   ```
3. Results (models, plots, metrics) will be in the `outputs/` folder.
