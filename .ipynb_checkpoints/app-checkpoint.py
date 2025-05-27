from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Loading trained model
model = pickle.load(open('random_forest_model_updated.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # HTML page for user input

@app.route('/predict', methods=['POST'])
    
def predict():
    # Getting form inputs
    credit_score = float(request.form['credit_score'])
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    tenure = int(request.form['tenure'])
    balance = float(request.form['balance'])
    num_of_products = int(request.form['num_of_products'])
    has_credit_card = int(request.form['has_credit_card'])
    is_active_member = int(request.form['is_active_member'])
    estimated_salary = float(request.form['estimated_salary'])
    geography = int(request.form['geography'])  # dropdown values: 0 (France), 1 (Germany), 2 (Spain)

    # Converting geography to one-hot encoded format
    geography_germany = 1 if geography == 1 else 0
    geography_spain = 1 if geography == 2 else 0

    # Create the feature array in the correct order
    raw_features = np.array([
        credit_score, gender, age, tenure, balance, num_of_products,
        has_credit_card, is_active_member, estimated_salary,
        geography_germany, geography_spain
    ]).reshape(1, -1)

    # Scaling only the numerical features
    scaled_numerical_features = scaler.transform(raw_features[:, [0, 2, 3, 4, 5, 8]])

    # Replace the numerical features in the original array with the scaled ones
    raw_features[:, [0, 2, 3, 4, 5, 8]] = scaled_numerical_features

    # Make prediction
    prediction = model.predict(raw_features)
    return f"Prediction: {'Churn' if prediction[0] == 1 else 'Not Churn'}"

if __name__ == '__main__':
    app.run(debug=True)

