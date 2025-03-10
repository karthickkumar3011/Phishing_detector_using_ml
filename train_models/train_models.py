import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv(r"data\phishing_detector_ml_df.csv")

# Prepare Features & Labels
X = df.drop(columns=["result"])  # Drop non-feature columns
y = df["result"]  # Target variable (1 for legit, -1 for phishing)

# Split into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def save_model(model, model_path):
    """Save trained model using pickle."""
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"\nModel saved at: {model_path}")

# Model Training & Evaluation Function
def train_and_evaluate(model, model_name):
    print(f"\nTraining {model_name}...\n")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary", pos_label=1)
    recall = recall_score(y_test, y_pred, average="binary", pos_label=1)
    class_report = classification_report(y_test, y_pred)
    
    # Cross Validation Score
    cross_val = cross_val_score(model, X_train, y_train, cv=5).mean()
    
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Cross Validation Score: {cross_val:.4f}")
    print("\nClassification Report:\n", class_report)
    
    print("Saving model...")
    save_model(model, rf"saved_models\{model_name}.pkl")
    
    return model

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_evaluate(rf_model, "Random_Forest")

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
train_and_evaluate(xgb_model, "XGBoost")

# Train Logistic Regression
log_reg_model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
train_and_evaluate(log_reg_model, "LogisticRegression")

# Train Decision Tree
decision_tree_model = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
train_and_evaluate(decision_tree_model, "DecisionTree")

# Train SVM
svm_model = SVC(kernel="rbf", probability=True)
train_and_evaluate(svm_model, "SVM")

# Train KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate(knn_model, "KNN")