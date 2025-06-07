# 🔥 Electric Motor Temperature Prediction

A robust machine learning project designed to predict the **permanent magnet temperature ('pm')** of a **Permanent Magnet Synchronous Motor (PMSM)**. This system leverages a Random Forest model and is deployable via a Flask API for real-time inference.

## 🌟 Project Highlights

- 📊 Comprehensive data preprocessing including outlier handling, scaling, and PCA-based dimensionality reduction.
- 🌲 Model training using a **Random Forest Regressor** for accurate temperature prediction.
- 🌐 Flask-based REST API for live prediction use cases.

---

## 📂 Repository Structure

```
project-root/
├── data/
│   └── Database.db               # SQLite database with "Electric_cars" table
├── src/
│   ├── training.py               # Script to train and evaluate the model
│   ├── prediction.py             # Flask API for serving predictions
├── models/
│   ├── motor_temp_model.pkl      # Trained Random Forest model
│   ├── pca_transformer.pkl       # PCA transformer for dimensionality reduction
│   └── feature_columns.pkl       # Feature names used in training
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Udit11/Electric-Motor-Temperature-Prediction.git
cd Electric-Motor-Temperature-Prediction
```

2. **Set up a virtual environment (recommended):**

```bash
python -m venv env
# On Unix or MacOS:
source env/bin/activate
# On Windows:
env\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Ensure that `Database.db` is placed inside the `/data` directory.

---

## 🖥️ Usage

1. **Train the Model:**

```bash
python src/training.py
```

Artifacts generated:
- `motor_temp_model.pkl`
- `pca_transformer.pkl`
- `feature_columns.pkl`

2. **Start the Flask API:**

```bash
python src/prediction.py
```

The API will be accessible at: [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)

3. **Make a Prediction:**

Send a POST request with JSON input to `/predict`.  
**Example input:**

```json
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
```

**Example response:**

```json
{
  "predicted_temperature": 75.34
}
```

---

## 📊 Model Performance

- **Model**: Random Forest Regressor
- **Mean Squared Error (MSE)**: 6.70
- **R² Score**: 0.983

---

## 🧰 Technologies Used

- **Language**: Python
- **Libraries**: pandas, scikit-learn, Flask, SQLite3, PCA
- **Model**: Random Forest Regressor

---

## 🔍 Future Enhancements

- Integrate multivariate models for dynamic environments.
- Introduce MLOps pipelines for scalable production.
- Deploy the solution to cloud platforms (AWS, Azure).

---

## 📝 License

Licensed under the MIT License. See the `LICENSE` file for full details.

---

## 🙌 Acknowledgments

Inspired by real-world use cases in electric mobility and industrial automation.

---

## 🤝 Contact

For inquiries or collaboration opportunities:

- 📧 Email: uditsrivastava2347@gmail.com  
- 🔗 LinkedIn: [Udit Srivastava](https://www.linkedin.com/in/udit-srivastava/)
