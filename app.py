from flask import Flask, render_template, request
import pickle
import pandas as pd
import joblib
import numpy as np

soil_model = joblib.load('soil_health_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')





# Load model once
with open("rotation_model.pkl", "rb") as f:
    rotation_model, le_crop, le_soil, le_season, le_target = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/crop")
def crop_page():
    return render_template("crop.html")

@app.route("/soil")
def soil_page():
    return render_template("soil.html")

@app.route("/weather")
def weather_page():
    return render_template("weather.html")

# === FORM HANDLERS ===

@app.route("/recommend", methods=["POST"])
def recommend_crop():
    ph = float(request.form["ph"])
    nitrogen = float(request.form["nitrogen"])
    phosphorus = float(request.form["phosphorus"])
    potassium = float(request.form["potassium"])
    location = request.form["location"]

    # Predict using the ML model
    input_data = np.array([[nitrogen, phosphorus, potassium, ph]])
    predicted_crop = crop_model.predict(input_data)[0]

    message = f"🌱 Recommended Crop for {location}: <strong>{predicted_crop}</strong>"
    return render_template("result.html", message=message)


@app.route("/predict-soil", methods=["POST"])
def predict_soil():
    try:
        ph = float(request.form["ph"])
        n = float(request.form["nitrogen"])
        p = float(request.form["phosphorus"])
        k = float(request.form["potassium"])
        oc = float(request.form["organic_carbon"])
        moisture = float(request.form["moisture"])

        # Simple rule-based prediction
        score = 0

        if 6 <= ph <= 7.5:
            score += 1
        if n > 80:
            score += 1
        if p > 30:
            score += 1
        if k > 150:
            score += 1
        if 0.5 <= oc <= 1.0:
            score += 1
        if 20 <= moisture <= 40:
            score += 1

        if score >= 5:
            status = "Good"
        elif 3 <= score < 5:
            status = "Moderate"
        else:
            status = "Poor"

        message = f"🌱 Soil Health Status: {status}<br>"

        if status == "Poor":
            message += "🔧 Suggestion: Add compost, reduce chemical use, rotate with legumes."
        elif status == "Moderate":
            message += "✨ Suggestion: Monitor nitrogen and organic content, improve drainage."
        else:
            message += "✅ Suggestion: Maintain current practices and test soil quarterly."

    except Exception as e:
        message = f"❌ Error: {e}"

    return render_template("result.html", message=message)


@app.route("/weather", methods=["POST"])
def weather():
    city = request.form["city"]

    # Placeholder
    alert = "Rain expected tomorrow. Prepare accordingly!"
    message = f"Weather Alert for {city}: {alert}"
    return render_template("result.html", message=message)


@app.route("/rotation")
def rotation_form():
    return render_template("rotation.html")

@app.route("/rotation-result", methods=["POST"])
def rotation_result():
    last_crop = request.form["last_crop"]
    soil_type = request.form["soil_type"]
    season = request.form["season"]

    input_df = pd.DataFrame({
        "last_crop": [le_crop.transform([last_crop])[0]],
        "soil_type": [le_soil.transform([soil_type])[0]],
        "season": [le_season.transform([season])[0]]
    })

    pred = rotation_model.predict(input_df)[0]
    recommendation = le_target.inverse_transform([pred])[0]

    message = f"🧠 Based on your input, our model recommends: {recommendation.title()}"
    

    return render_template("result.html", message=message)

@app.route('/predict_soil_health', methods=['POST'])
def predict_soil_health():
    try:
        # Get form data
        pH = float(request.form['ph'])
        om = float(request.form['om'])
        n = float(request.form['n'])
        p = float(request.form['p'])
        k = float(request.form['k'])

        # Create input array
        features = np.array([[pH, om, n, p, k]])

        # Predict
        prediction = soil_model.predict(features)
        label = label_encoder.inverse_transform(prediction)[0]

        return render_template('result_soil.html', label=label)

    except Exception as e:
        return f"Error: {e}"
    
    @app.route('/soil-form')
    def soil_form():
        return render_template('soil_form.html')





if __name__ == "__main__":
    app.run(debug=True, port=8000)
