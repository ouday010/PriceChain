# PropValuationApp
Machine learning model to predict property prices in Pakistan using 191,394 listings. Features XGBoost, CatBoost, RandomForest ensemble (R² ~0.86), KMeans clustering, and polynomial features. Deployed as a Flask web app with Leaflet.js map. #MachineLearning #RealEstate #Python

# Property Price Predictor

A machine learning model to predict property prices in Pakistan using a dataset of 153,112 real estate listings. Built with an ensemble of XGBoost, CatBoost, and RandomForest, achieving an R² score of ~0.86. Deployed as a Flask web app with a Leaflet.js map for location visualization. 🏠📍

## Features
- Predicts property prices based on features like location, bedrooms, area (marla), etc.
- Uses ensemble modeling (XGBoost, CatBoost, RandomForest) with feature engineering (KMeans clustering, polynomial features).
- Interactive web app with a modern UI and map visualization.
- Data cleaning and hyperparameter optimization for high accuracy.

## Dataset
- **Source**: Real estate listings in Pakistan (153,112 samples).
- **Files**:
  - `data/Property_with_Feature_Engineering.csv`: Full dataset with features like `latitude`, `longitude`, `baths`, `area_marla`, `bedrooms`, `year`, `month`, `day`, `property_type`, `location`, `city`, `locality`, `purpose`.
  - `data/X_train_raw.csv`, `data/X_test_raw.csv`, `data/y_train.npy`, `data/y_test.npy`: Pre-split training and test sets.
- **Note**: If using `Property_with_Feature_Engineering.csv`, split it into train/test sets (80/20) to create `X_train_raw.csv`, `X_test_raw.csv`, `y_train.npy`, `y_test.npy`. Example script:
  ```python
  import pandas as pd
  from sklearn.model_selection import train_test_split
  import numpy as np
  data = pd.read_csv('data/Property_with_Feature_Engineering.csv')
  X = data.drop(columns=['price'])
  y = np.log1p(data['price'])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train.to_csv('data/X_train_raw.csv', index=False)
  X_test.to_csv('data/X_test_raw.csv', index=False)
  np.save('data/y_train.npy', y_train)
  np.save('data/y_test.npy', y_test)
