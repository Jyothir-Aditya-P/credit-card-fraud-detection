CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING AND FASTAPI
1. INTRODUCTION
This repository contains an end-to-end credit card fraud detection system developed using machine learning and deployed through FastAPI. The objective is to accurately identify fraudulent transactions using the European Credit Card Fraud Dataset, which consists of 284,807 transactions with only 492 fraud cases (0.172%).
Fraud detection is challenging due to extreme class imbalance, anonymized PCA features, and the need for real-time predictions. This project includes exploratory analysis, preprocessing, oversampling with SMOTE, training multiple models, hyperparameter tuning, and deploying the final model (tuned XGBoost) behind a FastAPI interface.
Important:
 GitHub cannot preview the PDF project report due to its size.
 Please download the PDF to read it.

2. TECH STACK AND COMPONENT OVERVIEW
Machine Learning & Data Processing
Python
Pandas, NumPy
Scikit-Learn
Imbalanced-Learn (SMOTE)
XGBoost (final model)
LightGBM, Logistic Regression, Neural Network, Autoencoder (additional models)
Deployment
FastAPI
Uvicorn
Pydantic
Joblib


Tools
Jupyter Notebook
Matplotlib and Seaborn

3. PROJECT STRUCTURE
credit-card-fraud-detection/
 │
 ├── dataset/
 │ creditcard.csv
 │
 ├── notebooks/
 │ fraud_detection.ipynb
 │
 ├── models/
 │ fraud_model.joblib
 │
 ├── api/
 │ main.py
 │ schema.py
 │ preprocessing.py
 │
 ├── report/
 │ Final_Report.pdf
 │
 ├── README.md
 └── requirements.txt

4. MODEL PIPELINE OVERVIEW
Full Workflow
Raw data loading


Exploratory analysis


Preprocessing


Stratified train-test split


SMOTE oversampling (training only)


Model training (LR, XGBoost, LGBM, NN, Autoencoder)


Evaluation using precision, recall, F1


Hyperparameter tuning (XGBoost)


Model selection


Saving model with Joblib


FastAPI deployment


Text Diagram
Raw Data
 ↓
 Preprocessing
 ↓
 Train/Test Split
 ↓
 SMOTE (Train Only)
 ↓
 Model Training
 ↓
 Evaluation
 ↓
 Tuning
 ↓
 Final Model Saved
 ↓
 FastAPI Deployment
 ↓
 Client Sends JSON → Model Predicts → Response Returned

5. FASTAPI SYSTEM OVERVIEW
Purpose
 The API provides real-time fraud predictions based on JSON transaction input.
How It Works
Model is loaded at server startup


User sends data to /predict


Pydantic checks and validates inputs


Preprocessing is applied


XGBoost generates probability & classification


API returns structured JSON output


Architecture
Client
 ↓
 POST /predict
 ↓
 FastAPI
 ↓
 Validation
 ↓
 Preprocessing
 ↓
 Model
 ↓
 Response Returned

6. HOW TO RUN THE PROJECT LOCALLY
Step 1: Clone
 git clone https://github.com/yourusername/credit-card-fraud-detection.git
 cd credit-card-fraud-detection
Step 2: Install Dependencies
 pip install -r requirements.txt
Step 3: Run API
 cd api
 uvicorn main:app --reload


7. USER MANUAL
Sending a Prediction
 Send a POST request to /predict with all required fields (Time, Amount, V1–V28).
Example (structure only):
 Time: 10000
 Amount: 150.67
 V1: -1.23
 V2: 0.57
 ...
 V28: -0.18
Using curl:
 curl -X POST http://127.0.0.1:8000/predict -H Content-Type:application/json -d @input.json
Example Output
 prediction: Fraudulent
 fraud_probability: 0.9824
Notes
All 30 required fields must be present and numeric


Incorrect field names or missing values result in validation errors



8. TROUBLESHOOTING
Model Missing
 Place fraud_model.joblib inside the models/ directory.
Validation Errors
 Verify JSON body contains Time, Amount, V1–V28.
Server Not Starting
 Ensure Uvicorn is installed:
 pip install uvicorn

9. FUTURE ENHANCEMENTS
SHAP explanations


Real-time monitoring dashboard


Sequence modeling (LSTM / Transformers)


Automated retraining


Git LFS support for large datasets



Some Python files include long separator lines like:
-------------------------------------------------
These are simply manual visual dividers added to make the code easier to read.

