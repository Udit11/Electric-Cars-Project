import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    conn = sqlite3.connect('data/Database.db')
    data = pd.read_sql_query("Select * from Electric_cars", conn)
    conn.close()
    return data

def preprocess_data(data):
    columns_to_convert = ['profile_id', 'u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'pm']
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    data.fillna(data.mean(), inplace=True)
    outlier_counts = {}

    for col in data.select_dtypes(include=[np.number]).columns:  # Select numeric columns
        Q1 = data[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = data[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outlier_counts[col] = outliers

    outlier_counts = {col: count for col, count in outlier_counts.items() if count > 0}

    for col in outlier_counts.keys():
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    
    # Feature-target separation
    X = data.drop(columns=['pm'])  # Replace 'pm' with the actual target column
    y = data['pm']
    
    return X, y

def dim_red(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(X_scaled)

    return X_reduced, pca

def train_model(X, y,original_columns=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation:\nMSE: {mse}\nR2 Score: {r2}")
    
    if original_columns is not None:
        return model, original_columns
    else:
        # If original columns are not provided, return indices
        return model, range(X_train.shape[1])

def save_model(model, feature_columns, pca):
    joblib.dump(model, 'Project 3 (Electric Motor temperature prediction)/models/motor_temp_model.pkl')
    joblib.dump(feature_columns, 'Project 3 (Electric Motor temperature prediction)/models/feature_columns.pkl')
    joblib.dump(pca, 'Project 3 (Electric Motor temperature prediction)/models/pca_transformer.pkl')

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)
    X_PCA, PCA= dim_red(X)
    # Train and save model
    model, feature_columns = train_model(X_PCA, y)
    save_model(model, feature_columns, PCA)
    print("Model and feature columns saved successfully.")
