# resave_artifacts.py
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import joblib
import os

# Load raw data
data_path = 'data/'
X_train_raw = pd.read_csv(data_path + 'X_train_raw.csv')
X_test_raw = pd.read_csv(data_path + 'X_test_raw.csv')
y_train = np.load(data_path + 'y_train.npy', allow_pickle=True)
y_test = np.load(data_path + 'y_test.npy', allow_pickle=True)

# Drop price_bin
X_train_raw = X_train_raw.drop(columns=['price_bin'])
X_test_raw = X_test_raw.drop(columns=['price_bin'])
print("Dropped 'price_bin' to prevent leakage")

# Feature engineering
X_train_raw['bedrooms_area_marla'] = X_train_raw['bedrooms'] * X_train_raw['area_marla']
X_test_raw['bedrooms_area_marla'] = X_test_raw['bedrooms'] * X_test_raw['area_marla']
X_train_raw['baths_area_marla'] = X_train_raw['baths'] * X_train_raw['area_marla']
X_test_raw['baths_area_marla'] = X_test_raw['baths'] * X_test_raw['area_marla']
X_train_raw['lat_lon_interaction'] = X_train_raw['latitude'] * X_train_raw['longitude']
X_test_raw['lat_lon_interaction'] = X_test_raw['latitude'] * X_test_raw['longitude']
city_centers = {'Lahore': (31.5204, 74.3587), 'Karachi': (24.8607, 67.0011), 'Islamabad': (33.6844, 73.0479)}
def calculate_distance(row, city):
    if row['city'] in city_centers:
        lat_center, lon_center = city_centers[row['city']]
        return np.sqrt((row['latitude'] - lat_center)**2 + (row['longitude'] - lon_center)**2)
    return 0.0
X_train_raw['distance_to_center'] = X_train_raw.apply(lambda row: calculate_distance(row, row['city']), axis=1)
X_test_raw['distance_to_center'] = X_test_raw.apply(lambda row: calculate_distance(row, row['city']), axis=1)
kmeans = KMeans(n_clusters=10, random_state=42)
X_train_raw['location_cluster'] = kmeans.fit_predict(X_train_raw[['latitude', 'longitude']])
X_test_raw['location_cluster'] = kmeans.predict(X_test_raw[['latitude', 'longitude']])
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_cols = ['area_marla', 'bedrooms']
poly_features_train = poly.fit_transform(X_train_raw[poly_cols])
poly_features_test = poly.transform(X_test_raw[poly_cols])
poly_feature_names = poly.get_feature_names_out(poly_cols)
for i, name in enumerate(poly_feature_names):
    X_train_raw[f'poly_{name}'] = poly_features_train[:, i]
    X_test_raw[f'poly_{name}'] = poly_features_test[:, i]

# Define columns
numerical_cols = ['latitude', 'longitude', 'baths', 'area_marla', 'bedrooms', 'year', 'month', 'day',
                 'bedrooms_area_marla', 'baths_area_marla', 'lat_lon_interaction', 'distance_to_center']
numerical_cols.extend([f'poly_{name}' for name in poly_feature_names])
categorical_cols = ['property_type', 'location', 'city', 'locality', 'purpose', 'location_cluster']

# Handle NaNs
for col in numerical_cols:
    X_train_raw[col] = X_train_raw[col].replace([np.inf, -np.inf], np.nan).fillna(X_train_raw[col].median())
    X_test_raw[col] = X_test_raw[col].replace([np.inf, -np.inf], np.nan).fillna(X_train_raw[col].median())

# Create preprocessor
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
print(f"Preprocessed shapes: X_train {X_train.shape}, X_test {X_test.shape}")

# Save preprocessor
joblib.dump(preprocessor, data_path + 'preprocessor_final.pkl')
print("Saved preprocessor_final.pkl")

# Create holdout set
X_test, X_holdout, y_test, y_holdout = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

# Train models
xgb_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.8, random_state=42, n_jobs=1, early_stopping_rounds=10)
cat_model = CatBoostRegressor(random_state=42, verbose=0, thread_count=1, early_stopping_rounds=10)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
rf_model.fit(X_train, y_train)

# Feature selection
importance = xgb_model.feature_importances_
feature_names = preprocessor.get_feature_names_out().tolist()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
top_features = importance_df[importance_df['Importance'] > 0.0001]['Feature'].tolist()
feature_indices = [feature_names.index(f) for f in top_features]

# Filter data
X_train_filtered = X_train[:, feature_indices]
X_test_filtered = X_test[:, feature_indices]
X_holdout_filtered = X_holdout[:, feature_indices]

# Retrain models
xgb_model_filtered = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.8, random_state=42, n_jobs=1, early_stopping_rounds=10)
cat_model_filtered = CatBoostRegressor(random_state=42, verbose=0, thread_count=1, early_stopping_rounds=10)
rf_model_filtered = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
xgb_model_filtered.fit(X_train_filtered, y_train, eval_set=[(X_test_filtered, y_test)], verbose=False)
cat_model_filtered.fit(X_train_filtered, y_train, eval_set=(X_test_filtered, y_test))
rf_model_filtered.fit(X_train_filtered, y_train)

# Save models and artifacts
joblib.dump(xgb_model_filtered, data_path + 'xgboost_final_model.pkl', compress=3)
joblib.dump(cat_model_filtered, data_path + 'catboost_final_model.pkl', compress=3)
joblib.dump(rf_model_filtered, data_path + 'rf_final_model.pkl', compress=3)
importance_df.to_csv(data_path + 'feature_importance.csv', index=False)
joblib.dump(feature_indices, data_path + 'selected_feature_indices.pkl')
joblib.dump([0.6, 0.3, 0.1], data_path + 'ensemble_weights.pkl')
print("Saved all artifacts")

# Validate
y_pred_xgb_log = xgb_model_filtered.predict(X_test_filtered)
y_pred_cat_log = cat_model_filtered.predict(X_test_filtered)
y_pred_rf_log = rf_model_filtered.predict(X_test_filtered)
y_pred_log = 0.6 * y_pred_xgb_log + 0.3 * y_pred_cat_log + 0.1 * y_pred_rf_log
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)
print("Validation Results:")
print(f"R2: {r2_score(y_test_orig, y_pred):.2f}")