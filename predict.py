import numpy as np
import pandas as pd
import joblib

def predict_price(data):
    data_path = 'data/'
    # Load artifacts
    xgb_model = joblib.load(data_path + 'xgboost_final_model.pkl')
    cat_model = joblib.load(data_path + 'catboost_final_model.pkl')
    rf_model = joblib.load(data_path + 'rf_final_model.pkl')
    preprocessor = joblib.load(data_path + 'preprocessor_final.pkl')
    feature_indices = joblib.load(data_path + 'selected_feature_indices.pkl')
    ensemble_weights = joblib.load(data_path + 'ensemble_weights.pkl')

    # Convert input data to DataFrame
    input_data = pd.DataFrame([data])

    # Feature engineering
    input_data['bedrooms_area_marla'] = input_data['bedrooms'] * input_data['area_marla']
    input_data['baths_area_marla'] = input_data['baths'] * input_data['area_marla']
    input_data['lat_lon_interaction'] = input_data['latitude'] * input_data['longitude']

    # Distance to city center
    city_centers = {'Lahore': (31.5204, 74.3587), 'Karachi': (24.8607, 67.0011), 'Islamabad': (33.6844, 73.0479)}
    def calculate_distance(row):
        if row['city'] in city_centers:
            lat_center, lon_center = city_centers[row['city']]
            return np.sqrt((row['latitude'] - lat_center)**2 + (row['longitude'] - lon_center)**2)
        return 0.0
    input_data['distance_to_center'] = input_data.apply(calculate_distance, axis=1)

    # Load KMeans and PolynomialFeatures
    kmeans = joblib.load(data_path + 'kmeans_model.pkl')
    input_data['location_cluster'] = kmeans.predict(input_data[['latitude', 'longitude']])
    poly = joblib.load(data_path + 'poly_transformer.pkl')
    poly_cols = ['area_marla', 'bedrooms']
    poly_features = poly.transform(input_data[poly_cols])
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    for i, name in enumerate(poly_feature_names):
        input_data[f'poly_{name}'] = poly_features[:, i]

    # Handle NaNs
    numerical_cols = ['latitude', 'longitude', 'baths', 'area_marla', 'bedrooms', 'year', 'month', 'day',
                      'bedrooms_area_marla', 'baths_area_marla', 'lat_lon_interaction', 'distance_to_center']
    numerical_cols.extend([f'poly_{name}' for name in poly_feature_names])
    for col in numerical_cols:
        input_data[col] = input_data[col].replace([np.inf, -np.inf], np.nan).fillna(input_data[col].median())

    # Transform with preprocessor
    X = preprocessor.transform(input_data)
    X_filtered = X[:, feature_indices]

    # Predict
    y_pred_xgb_log = xgb_model.predict(X_filtered)
    y_pred_cat_log = cat_model.predict(X_filtered)
    y_pred_rf_log = rf_model.predict(X_filtered)
    y_pred_log = (ensemble_weights[0] * y_pred_xgb_log +
                  ensemble_weights[1] * y_pred_cat_log +
                  ensemble_weights[2] * y_pred_rf_log)
    predicted_price = np.expm1(y_pred_log)[0]
    return predicted_price