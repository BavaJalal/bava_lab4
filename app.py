from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and label encoder
model = joblib.load("fish_species_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    # Render the main page (HTML frontend)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        data = request.form.to_dict()
        # Convert string inputs to float values in the proper order
        features = [float(data[x]) for x in ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
        # Scale features
        features_scaled = scaler.transform([features])
        # Predict species
        pred = model.predict(features_scaled)
        # Convert numerical label back to species name
        species = le.inverse_transform(pred)[0]
        return render_template('index.html', prediction_text="Predicted Fish Species: {}".format(species))
    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
