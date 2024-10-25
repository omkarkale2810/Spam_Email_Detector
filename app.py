from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Paths to save the model, vectorizer, and metrics
MODEL_PATH = './spam_detector_model.pkl'
VECTORIZER_PATH = './vectorizer.pkl'
METRICS_PATH = './metrics.pkl'

metrics = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None}

def train_and_save_model():
    # Load and preprocess dataset
    raw_mail_data = pd.read_csv("./mail_data.csv")
    mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

    # Label encoding: 'ham' as 1 and 'spam' as 0
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

    # Split into features (X) and labels (Y)
    x = mail_data['Message']
    y = mail_data['Category']

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

    # Feature extraction
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    x_train_features = feature_extraction.fit_transform(x_train)
    x_test_features = feature_extraction.transform(x_test)
    
    # Model training
    model = LogisticRegression()
    model.fit(x_train_features, y_train.astype('int'))

    # Save the model and vectorizer using pickle
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(VECTORIZER_PATH, 'wb') as vectorizer_file:
        pickle.dump(feature_extraction, vectorizer_file)

    # Calculate metrics
    y_pred = model.predict(x_test_features)
    metrics['accuracy'] = accuracy_score(y_test.astype('int'), y_pred)
    metrics['precision'] = precision_score(y_test.astype('int'), y_pred)
    metrics['recall'] = recall_score(y_test.astype('int'), y_pred)
    metrics['f1'] = f1_score(y_test.astype('int'), y_pred)

    # Save metrics to a file
    with open(METRICS_PATH, 'wb') as metrics_file:
        pickle.dump(metrics, metrics_file)

# Load model, vectorizer, and metrics if they exist; otherwise, train and save them
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(METRICS_PATH):
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        feature_extraction = pickle.load(vectorizer_file)
    with open(METRICS_PATH, 'rb') as metrics_file:
        metrics = pickle.load(metrics_file)
else:
    train_and_save_model()

@app.route('/')
def index():
    return render_template('index.html', accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'], f1=metrics['f1'])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the email content from the form
        input_mail = [request.form['email']]
        
        # Convert the email text into feature vectors
        input_data_features = feature_extraction.transform(input_mail)
        
        # Predict whether it's spam or ham
        prediction = model.predict(input_data_features)
        
        # Render the result
        result = "Ham (Not Spam)" if prediction[0] == 1 else "Spam"
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
