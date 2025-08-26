# test_artifacts.py
import joblib
import pandas as pd

artifacts = [
    'xgboost_final_model.pkl',
    'catboost_final_model.pkl',
    'rf_final_model.pkl',
    'preprocessor_final.pkl',
    'selected_feature_indices.pkl',
    'ensemble_weights.pkl',
    'kmeans_model.pkl',
    'poly_transformer.pkl',
    'feature_importance.csv'
]
for artifact in artifacts:
    try:
        if artifact.endswith('.csv'):
            obj = pd.read_csv(f'data/{artifact}')
        else:
            obj = joblib.load(f'data/{artifact}')
        print(f"Loaded {artifact} successfully")
    except Exception as e:
        print(f"Failed to load {artifact}: {str(e)}")