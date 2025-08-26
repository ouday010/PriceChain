# generate_missing_artifacts.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import joblib

# Load raw data
data_path = 'data/'
X_train_raw = pd.read_csv(data_path + 'X_train_raw.csv')

# Feature engineering for KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
X_train_raw['location_cluster'] = kmeans.fit_predict(X_train_raw[['latitude', 'longitude']])
joblib.dump(kmeans, data_path + 'kmeans_model.pkl')
print("Saved kmeans_model.pkl")

# Feature engineering for PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_cols = ['area_marla', 'bedrooms']
poly.fit(X_train_raw[poly_cols])
joblib.dump(poly, data_path + 'poly_transformer.pkl')
print("Saved poly_transformer.pkl")