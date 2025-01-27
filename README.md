# 🔥 Electric Motor Temperature Prediction

### 🌟 Project Description
This project develops a machine learning model to predict the **permanent magnet temperature ('pm')** of a **Permanent Magnet Synchronous Motor (PMSM)**. The project includes:
- Data preprocessing with outlier handling, scaling, and dimensionality reduction.
- Model training using a **Random Forest Regressor**.
- Deployment of a **Flask API** for real-time predictions.

---

### 📂 Repository Structure
project-root/                     # Root directory of the project
├── data/                         # Directory for datasets or database files
│   └── Database.db               # SQLite database containing the "Electric_cars" table
│
├── src/                          # Directory for source code
│   ├── Training.py               # Script for model training and evaluation
│   ├── prediction.py             # Flask API for real-time predictions
│
├── models/                       # Directory for storing trained models and related files
│   ├── motor_temp_model.pkl      # Trained Random Forest model
│   ├── pca_transformer.pkl       # PCA transformer for dimensionality reduction
│   └── feature_columns.pkl       # File containing selected feature names
│
├── requirements.txt              # File listing project dependencies
├── README.md                     # Documentation for the project (this file)

---

### 🛠️ Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/Electric-Motor-Temperature-Prediction.git
   cd Electric-Motor-Temperature-Prediction
Set Up a Virtual Environment:
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate

Install Dependencies:
pip install -r requirements.txt
Ensure the Database File (Database.db) is in the /data/ Directory.

---

### 🖥️ Usage
1. **Training the Model:**
Run the training.py script to train and save the model:
    python src/training.py
Outputs:
    motor_temp_model.pkl (Trained model)
    pca_transformer.pkl (PCA transformer)
    feature_columns.pkl (Selected feature columns)

2. **Starting the Flask API**
Run the prediction.py script to start the API for real-time predictions:
    python src/prediction.py
Access the API at http://127.0.0.1:5000/predict.

3. **Making Predictions**
Send a POST request to the /predict endpoint with the input data in JSON format.
Example input:
    {
    "profile_id": 1,
    "u_q": 220.5,
    "coolant": 23.3,
    "u_d": 100.1,
    "motor_speed": 3000,
    "i_d": 1.2,
    "i_q": 5.8,
    "ambient": 25.0
    }
Example response:
{
  "predicted_temperature": 75.34
}

---

### 📊 Results
Model Used: Random Forest Regressor
Evaluation Metrics:
Mean Squared Error (MSE): 6.701167597909399
R² Score: 0.983285437151667

---

### 📚 Technologies Used
Programming Language: Python
Libraries: Pandas, Scikit-learn, Flask, SQLite3, PCA
Model: Random Forest Regressor

---

### 🔍 Future Improvements
Add support for multivariate prediction models for dynamic driving conditions.
Optimize model for large-scale production using MLOps pipelines.
Deploy the API to a cloud service (e.g., AWS, Azure) for scalability.

---

### 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### 🙌 Acknowledgments
Inspired by real-world applications in electric vehicles and industrial automation.

---

### 🤝 Contact
For questions or collaborations, feel free to reach out:
Email: uditsrivastava2347@gmail.com
LinkedIn: [linkedin.com/in/udit-srivastava](https://www.linkedin.com/in/udit-srivastava/)

---

You can replace `<your-username>` in the repository URL with your GitHub username. Let me know if you need further adjustments! 🚀
