from flask import Flask, render_template, request
import pandas as pd
import joblib  # For loading the models
import numpy as np
from utils.features_extraction import extract_features  # Import the function
import tensorflow as tf  # Make sure to import TensorFlow
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load your models
rf_model = joblib.load(r'saved_models/Random_Forest.pkl')
xgb_model = joblib.load(r'saved_models/XGBoost.pkl')
log_reg_model=joblib.load(r'saved_models/LogisticRegression.pkl')
decision_tree_model=joblib.load(r'saved_models/DecisionTree.pkl')
knn_model=joblib.load(r'saved_models/KNN.pkl')
svm_model=joblib.load(r'saved_models/LogisticRegression.pkl')
nb_model=joblib.load(r'saved_models/Naive_Bayes.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    features = extract_features(url)  # Extract features from the URL

    # Prepare the feature array for prediction
    feature_array = np.array([[features["url_length"],
                                features["num_dots"],
                                features["num_special_chars"],
                                features["has_at_symbol"],
                                features["has_hyphen"],
                                features["is_ip"],
                                features["has_redirect"],
                                features["num_subdomains"],
                                features["is_shortened"],
                                features["url_entropy"],
                                features["has_suspicious_keyword"],
                                features["is_brand_spoofed"],
                                features["domain_age"],
                                features["domain_expiry"],
                                features["whois_private"],
                                features["https_used"]]])

    # Make predictions using the models
    rf_prediction = rf_model.predict(feature_array)
    xgb_prediction = xgb_model.predict(feature_array)
    log_reg_prediction = log_reg_model.predict(feature_array)
    decision_tree_prediction = decision_tree_model.predict(feature_array)
    knn_prediction = knn_model.predict(feature_array)
    svm_prediction = svm_model.predict(feature_array)
    nb_prediction = nb_model.predict(feature_array)

    # Interpret the predictions
    rf_result = "Phishing" if rf_prediction[0] == 0 else "Legitimate"
    xgb_result = "Phishing" if xgb_prediction[0] == 0 else "Legitimate"
    log_reg_result = "Phishing" if log_reg_prediction[0] == 0 else "Legitimate"
    decision_tree_result = "Phishing" if decision_tree_prediction[0] == 0 else "Legitimate"
    knn_result = "Phishing" if knn_prediction[0] == 0 else "Legitimate"
    svm_result = "Phishing" if svm_prediction[0] == 0 else "Legitimate"
    nb_result = "Phishing" if nb_prediction[0] == 0 else "Legitimate"

    return render_template('index.html', 
                           prediction_rf=rf_result, 
                           prediction_xgb=xgb_result,
                           prediction_log_reg=log_reg_result,
                           prediction_decision_tree=decision_tree_result,
                           prediction_knn=knn_result,
                           prediction_svm=svm_result,
                           prediction_nb=nb_result
                           )

if __name__ == '__main__':
    app.run(debug=True)