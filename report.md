
# Report: Advanced Time Series Forecasting with Deep Learning and Explainability

## 1. Problem Statement
Implement an end-to-end multivariate time series forecasting pipeline with deep learning models (LSTM/Transformer), uncertainty quantification, rolling-origin validation, and explainability.

## 2. Dataset
A reproducible synthetic multivariate dataset was generated (3 series, trend + multiple seasonalities + exogenous signal) with 600 daily steps. This ensures the pipeline runs without external data downloads.

## 3. Preprocessing
- Created 20 lag features for each series.
- Standard scaling applied to features.
- Rolling-origin splits (initial train 350, horizon 7, step 50).

## 4. Models and Training
- **LSTM:** small Keras LSTM trained on the last rolling split for demonstration (6 epochs). Model saved to `outputs/lstm_model.h5` if TensorFlow available.
- **Quantile models:** three small models trained with pinball loss for q=0.1,0.5,0.9 to produce prediction intervals.

## 5. Evaluation Metrics
The pipeline computes and saves metrics (MAE, RMSE, sMAPE, MASE) for baseline and LSTM outputs in `outputs/` as JSON files. Run `python main.py` to produce these metrics on your machine.

### Metric definitions implemented:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- sMAPE: symmetric Mean Absolute Percentage Error
- MASE: Mean Absolute Scaled Error (scaled by in-sample naive differences)

## 6. Uncertainty Quantification
Quantile predictions are saved to `outputs/quantile_predictions.csv`. Coverage of the 0.1-0.9 interval is saved in `outputs/quantile_coverage.json` to assess interval calibration.

## 7. Explainability
SHAP KernelExplainer is used on a small sample to compute approximate feature attributions. SHAP summary plot saved as `outputs/shap_summary.png` if SHAP and TensorFlow are installed.

## 8. Results (How to produce)
1. Install requirements.
2. Run `python main.py` — this trains a quick LSTM and quantile models, then saves metrics and plots in `outputs/`.
3. Open the JSON metric files and images to include in your final submission.

## 9. Conclusion
This submission includes executable code, uncertainty quantification, explainability steps, and evaluation metrics required by the assignment. Run the pipeline on a machine with TensorFlow and SHAP to generate final artifacts for upload. The code is intentionally conservative in compute so it completes quickly for student machines.
