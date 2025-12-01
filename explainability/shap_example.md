
# SHAP explainability notes

- Install shap: pip install shap
- Use `shap.KernelExplainer` for models without direct SHAP support, or `shap.DeepExplainer`/`shap.GradientExplainer` if using TensorFlow.
- For time series, compute SHAP values for lagged features and map back to original time steps.
