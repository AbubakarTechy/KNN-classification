from flask import Flask, render_template, request
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Initialize the Flask app
app = Flask(__name__)

# Load the KNN model and test data
model_dir = 'model_data'
knn_model = joblib.load(os.path.join(model_dir, 'knn_model.joblib'))
X_test = joblib.load(os.path.join(model_dir, 'X_test.joblib'))
y_test = joblib.load(os.path.join(model_dir, 'y_test.joblib'))
X_train = joblib.load(os.path.join(model_dir, 'X_train.joblib')) # Added this line
y_train = joblib.load(os.path.join(model_dir, 'y_train.joblib')) # Added this line

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Make prediction for user input
        user_input = np.array([[feature1, feature2]])
        user_prediction = knn_model.predict(user_input)[0]

        # Make predictions on the test set for evaluation metrics
        y_pred = knn_model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Received input: Feature 1 = {feature1}, Feature 2 = {feature2}")
        print(f"User Prediction: {user_prediction}")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

        # Generate Matplotlib plot
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot training data
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', label='Training Data')
        # Plot test data
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Test Data')
        # Plot user input
        ax.scatter(feature1, feature2, c='red', marker='*', s=300, label=f'User Input (Predicted: {user_prediction})')

        # Add decision boundary if possible (more complex for KNN, but can visualize regions)
        # For simplicity, we'll just show the points for now.

        ax.set_title('KNN Classification with User Input')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True)

        # Save plot to a BytesIO object and encode to base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig) # Close the figure to free memory
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')

        return render_template('result.html',
                               feature1=feature1,
                               feature2=feature2,
                               prediction=user_prediction,
                               accuracy=accuracy,
                               precision=precision,
                               recall=recall,
                               f1=f1,
                               plot_url=plot_url)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
