from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset and preprocess
raw_mail_data = pd.read_csv("./mail_data.csv")
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Label encoding: 'ham' as 1 and 'spam' as 0
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Split the data into features (X) and labels (Y)
x = mail_data['Message']
y = mail_data['Category']

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Feature extraction using TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# Convert labels to integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Model training using Logistic Regression
model = LogisticRegression()
model.fit(x_train_features, y_train)

@app.route('/')
def index():
    return render_template('index.html')

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
        if prediction[0] == 1:
            result = "Ham (Not Spam)"
        else:
            result = "Spam"
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
