import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_pipeline():
    # 1. Load Data
    df = pd.read_csv('customer_churn.csv')
    
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    
    # 2. Define Feature Types
    numeric_features = ['tenure_months', 'monthly_charges', 'total_support_calls']
    categorical_features = ['contract_type']
    
    # 3. Build Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    
    # 4. Full Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 5. Train & Evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    print("--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, model_pipeline.predict_proba(X_test)[:, 1]):.4f}")
    
    # 6. Save Artifact
    joblib.dump(model_pipeline, 'churn_model.pkl')
    print("Model pipeline successfully saved to 'churn_model.pkl'")

if __name__ == "__main__":
    train_pipeline()
