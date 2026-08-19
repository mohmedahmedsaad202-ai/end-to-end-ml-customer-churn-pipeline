# End-to-End Customer Churn Data Science Pipeline

A production-ready Data Science system for predicting customer churn risk using Machine Learning and serving real-time predictions via a FastAPI web engine.

## Key Features
- **Data Pipeline:** Preprocessing numerical scaling and One-Hot Encoding via `ColumnTransformer`.
- **Machine Learning:** Random Forest Classifier trained with cross-validated evaluation metrics (Precision, Recall, ROC-AUC).
- **Model Deployment:** REST API built with FastAPI accepting customer profiles and returning real-time risk scores.

## Tech Stack
- **Data Processing & ML:** Python 3.10+, Pandas, Scikit-Learn, Joblib
- **API Engine:** FastAPI, Uvicorn, Pydantic

## Execution
1. Generate synthetic dataset:
   ```bash
   python generate_data.py
