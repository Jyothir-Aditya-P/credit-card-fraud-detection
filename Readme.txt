Credit Card Fraud Detection Using Machine Learning and FastAPI
1. Introduction
This project implements an end-to-end credit card fraud detection system using machine learning techniques and a real-time prediction API.
The goal is to classify each credit card transaction as Fraudulent or Genuine, using the European Credit Card Fraud Dataset (284,807 transactions with only 492 frauds – 0.172%).
The project addresses the following challenges:
* Extremely imbalanced dataset

* PCA-transformed anonymized features

* Outlier-heavy amount values

* Requirement for high precision and recall

* Real-time deployment for practical integration

The final system includes data preprocessing, SMOTE oversampling, training multiple models, selecting the best classifier (tuned XGBoost), and deploying the model using FastAPI.
________________


2. Tech Stack and Component Overview
Machine Learning and Data Processing
   1. Python – Main development language

   2. Pandas, NumPy – Data handling and numerical operations

   3. Scikit-Learn – Preprocessing, train/test split, evaluation

   4. Imbalanced-Learn (SMOTE) – Handling class imbalance

   5. XGBoost – Final selected classifier

   6. LightGBM, Logistic Regression, Neural Network, Autoencoder – Secondary models for comparison

Deployment
      1. FastAPI – High-performance web framework for building the API

      2. Uvicorn – ASGI server to run the FastAPI application

      3. Pydantic – Input validation for prediction requests

      4. Joblib – Saving and loading the trained model

Development Tools
         * Jupyter Notebook – Exploratory analysis & model development

         * Matplotlib/Seaborn – Data visualization

________________


3. Project Structure
credit-card-fraud-detection/
│
├── dataset/
│     └── creditcard.csv
│
├── notebooks/
│     └── fraud_detection.ipynb
│
├── models/
│     └── fraud_model.joblib
│
├── api/
│     ├── main.py
│     ├── schema.py
│     └── preprocessing.py
│
├── report/
│     └── Final_Report.pdf
│
├── README.md
└── requirements.txt


________________


4. Model Pipeline Overview
Complete Data and Model Pipeline
            1. Raw Data Loading

            2. Exploratory Data Analysis

            3. Preprocessing

               * Handling skewed Amount field

               * Scaling Time and Amount using RobustScaler

               * Dropping original columns

                  4. Train/Test Split

                     * Stratified to preserve fraud ratio

                        5. SMOTE Oversampling on Training Data Only

                        6. Model Training

                           * Logistic Regression

                           * XGBoost

                           * LightGBM

                           * Neural Network

                           * Autoencoder

                              7. Model Evaluation

                                 * Precision, Recall, F1-score

                                    8. Hyperparameter Tuning

                                       * RandomizedSearchCV for XGBoost

                                          9. Model Selection (XGBoost)

                                          10. Model Saving with Joblib

                                          11. FastAPI Deployment

Pipeline Diagram (Text Representation)
Raw Data
   ↓
Preprocessing (Scaling, Feature Replacement)
   ↓
Train/Test Split (Stratified)
   ↓
SMOTE Oversampling (Training set only)
   ↓
Model Training (LR, XGBoost, LGBM, NN, Autoencoder)
   ↓
Model Evaluation (Precision, Recall, F1)
   ↓
Hyperparameter Tuning (RandomizedSearchCV)
   ↓
Best Model Saved (joblib)
   ↓
FastAPI Deployment
   ↓
User Sends JSON → Model Predicts → Response Returned


________________


5. FastAPI System Overview
Purpose
To provide a real-time fraud prediction service which accepts transaction data as JSON and returns both:
                                             * Fraud/Genuine classification

                                             * Fraud probability score

How FastAPI Works in This Project
                                                * Loads the trained XGBoost model

                                                * Accepts POST requests at /predict

                                                * Validates input via Pydantic schema

                                                * Applies preprocessing (scaling)

                                                * Passes data to the model

                                                * Returns prediction as JSON

API Architecture (Text Diagram)
Client Application
        ↓
POST /predict (JSON Input)
        ↓
FastAPI Endpoint
        ↓
Pydantic Validation
        ↓
Preprocessing Module
        ↓
XGBoost Model (joblib)
        ↓
Prediction (label + probability)
        ↓
JSON Response


________________


6. How to Run the Project Locally
Step 1: Clone the Repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection


Step 2: Install Dependencies
pip install -r requirements.txt


Step 3: Start the FastAPI Server
cd api
uvicorn main:app --reload


Server runs at:
http://127.0.0.1:8000


Open documentation:
http://127.0.0.1:8000/docs


________________


7. User Manual
7.1 Sending a Prediction Request
POST request to /predict
Example JSON body:
{
    "Time": 10000,
    "Amount": 150.67,
    "V1": -1.23,
    "V2": 0.57,
    ...
    "V28": -0.18
}


Use:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @input.json


7.2 Example Response
{
    "prediction": "Fraudulent",
    "fraud_probability": 0.9824
}


________________


8. Troubleshooting
Common Issues
                                                   * Model not found
 Ensure fraud_model.joblib is in models/.

                                                   * Validation error
 All 30 numerical fields must be present.

                                                   * Server not starting
 Check Uvicorn installation and Python path.

________________


9. Future Enhancements
                                                      * Add SHAP model explainability to API

                                                      * Implement real-time monitoring dashboard

                                                      * Integrate LSTM/Transformers for sequence modeling

                                                      * Automate retraining for concept drift