import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_model():
    model = joblib.load('Project 3 (Electric Motor temperature prediction)/motor_temp_model.pkl')
    pca = joblib.load('Project 3 (Electric Motor temperature prediction)/pca_transformer.pkl')
    return model, pca

def preprocess_input(data, feature_columns, pca):
    df = pd.DataFrame(data, index=[0])
    df = df[feature_columns]
    df_scaled = StandardScaler().fit_transform(df)  # Ensure input data is scaled
    df_pca = pca.transform(df_scaled)  # Apply PCA
    return df_pca

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        model, pca= load_model()
        feature_columns = ['profile_id', 'u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient']
        processed_data = preprocess_input(input_data, feature_columns, pca)
        prediction = model.predict(processed_data)[0]
        return jsonify({"predicted_temperature": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)