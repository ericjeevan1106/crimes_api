from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("crime_predictor_model.pkl")
input_encoders = joblib.load("input_label_encoders.pkl")
output_encoders = joblib.load("output_label_encoders.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Encode input fields
        for col in ["Area_Name", "Zone_Name"]:
            le = input_encoders[col]
            df[col] = le.transform(df[col])

        # Predict
        prediction = model.predict(df)[0]

        # Decode outputs
        output = {}
        output_columns = list(output_encoders.keys())
        for i, col in enumerate(output_columns):
            val = prediction[i]
            le = output_encoders[col]
            decoded = le.inverse_transform([int(round(val))])[0]
            output[col] = decoded

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=["GET"])
def home():
    return "✅ Crime Prediction API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
