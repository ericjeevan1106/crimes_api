from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(_name_)

# Load model and encoders
model = joblib.load("crime_predictor_model.pkl")
input_encoders = joblib.load("input_label_encoders.pkl")
output_encoders = joblib.load("output_label_encoders.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure all required fields are present
        required_fields = ["Area_Name", "Pincode", "Latitude", "Longitude", "Zone_Name"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Convert input data to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical input features
        for col in ["Area_Name", "Zone_Name"]:
            if col in input_encoders:
                le = input_encoders[col]
                if df[col].values[0] not in le.classes_:
                    return jsonify({"error": f"Unseen label in column '{col}': {df[col].values[0]}"}), 400
                df[col] = le.transform(df[col])

        # Predict using model
        prediction = model.predict(df)[0]

        # Decode each predicted label
        decoded_output = {}
        output_columns = list(output_encoders.keys())
        for i, col in enumerate(output_columns):
            le = output_encoders[col]
            value = int(round(prediction[i]))
            if value >= len(le.classes_):
                decoded_output[col] = f"Unknown ({value})"
            else:
                decoded_output[col] = le.inverse_transform([value])[0]

        return jsonify(decoded_output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Crime Prediction API is running!"

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)
