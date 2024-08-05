import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, render_template
import numpy as np

# Sample DataFrame (replace this with loading your dataset)
df = pd.read_csv('C:\\Studentbot\\Student_performance_data _ - Copy.csv')

# Dropping unnecessary columns based on correlation analysis
selected_features = [
    'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
      'GPA'
]

X = df[selected_features]
y = df['GradeClass']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
RFC = RandomForestClassifier(random_state=42)
RFC.fit(x_train, y_train)

# Save the model as a pickle file
with open('RFC_std.pkl', 'wb') as file:
    pickle.dump(RFC, file)

# Initialize the Flask app
app = Flask(__name__)

# Define the grade mapping
grade_mapping = {
    0: 'Fail 😖',
    1: 'Pass 🤩',
    2: 'Good 🥉 ',
    3: 'Very Good 🥈 ',
    4: 'Excellent 🥇'
}

# Load the model
model_path = 'C:\\Studentbot\\RFC_std.pkl'
with open(model_path, 'rb') as model_file:
    RFC = pickle.load(model_file)

# Home page
@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html template for the homepage

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    features = [float(x) for x in request.form.values()]
    
    # Reshape to match the expected input dimensions of RandomForestClassifier
    final_features = np.array(features).reshape(1, -1)  # Shape: (1, n_features)
    
    # Make prediction
    prediction = RFC.predict(final_features)
    
    # Convert prediction to human-readable label
    predicted_label = grade_mapping[prediction[0]]
    
    # Return prediction
    return render_template('index.html', prediction_text=f'Predicted GradeClass: {predicted_label}')

if __name__ == "__main__":
    app.run(debug=True)
