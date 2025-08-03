from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from external domains

# 🔄 Load trained ML model
model = joblib.load("chennai_crime_predictor (3).joblib")

# 🔮 Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ✅ Ensure required fields are present
        required_fields = ['Area_Name', 'Pincode', 'Latitude', 'Longitude', 'Zone_Name']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing input fields'}), 400

        # 🧾 Convert input to DataFrame
        input_df = pd.DataFrame([{
            'Area_Name': data['Area_Name'],
            'Pincode': data['Pincode'],
            'Latitude': data['Latitude'],
            'Longitude': data['Longitude'],
            'Zone_Name': data['Zone_Name']
        }])

        # 🔮 Make prediction
        prediction = model.predict(input_df)

        # 🧾 Output features in order (same as training)
        output_features = [
            'Crime_Type', 'Crime_Subtype', 'Crime_Severity', 'Victim_Age_Group',
            'Victim_Gender', 'Suspect_Count', 'Weapon_Used', 'Gang_Involvement',
            'Vehicle_Used', 'CCTV_Captured', 'Reported_By', 'Response_Time_Minutes',
            'Arrest_Made', 'Crime_History_Count', 'Crimes_Same_Type_Count', 'Risk_Level'
        ]

        # 🧠 Convert result to readable format (already strings)
        result = {feature: prediction[0][i] for i, feature in enumerate(output_features)}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🚀 Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
