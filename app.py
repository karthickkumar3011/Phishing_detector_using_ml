from flask import Flask, render_template, request, flash, redirect
import joblib 
import os
import numpy as np
from utils.features_extraction import extract_features 
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key') 

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
    try:
        features = extract_features(url)  # Extract features from the URL
    except Exception as e:
        flash(f"Error extracting features: {str(e)}", "danger")
        return render_template('index.html')

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
    predictions = [rf_prediction[0], xgb_prediction[0], log_reg_prediction[0], decision_tree_prediction[0], knn_prediction[0], svm_prediction[0], nb_prediction[0]]
    results = ["Phishing" if pred == 0 else "Legitimate" for pred in predictions]
    phishing_count = results.count("Phishing")

    # Determine the majority vote based on the phishing count
    if phishing_count > 5:
        majority_vote = "Phishing"
    elif phishing_count == 4:
        majority_vote = "70% chance of being Phishing"
    elif phishing_count == 3:
        majority_vote = "50% chance of being Phishing"
    else:
        majority_vote = "Legitimate (URL seems safe to use)"

    return render_template('index.html', 
                           majority_vote=majority_vote
                           )

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Corrected path

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Received data: Name={name}, Email={email}, Message={message}")  # Debugging log

        # Save the message (if using a database, add the logic here)
        flash(f"Thank you, {name}! Your message has been saved.", "success")
        print("Message saved successfully.")  # Debugging log
    except Exception as e:
        print(f"Error in /send_message: {str(e)}")  # Debugging log
        flash("An error occurred while saving your message.", "danger")
    return redirect('/contact')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

if __name__ == '__main__':
    app.run(debug=True)