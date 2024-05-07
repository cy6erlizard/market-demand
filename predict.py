from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the Random Forest model
model = joblib.load("random_forest_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json['data']
    input_df = pd.DataFrame(input_data)

    # Make predictions
    predictions = model.predict(input_df)

    # Return predictions as JSON response
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
