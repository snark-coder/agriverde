from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests
from datetime import datetime, timedelta

import os

from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import joblib


# Load model once
with open("rotation_model.pkl", "rb") as f:
    rotation_model, le_crop, le_soil, le_season, le_target = pickle.load(f)

app = Flask(__name__)

WEATHER_API_KEY = "cf0c4fed7889832b998b3d94c52ab29a"
BASE_WEATHER_URL = "https://api.openweathermap.org/data/2.5/"
GEOCODING_API_URL = "http://api.openweathermap.org/geo/1.0/reverse" # For reverse geocoding

def get_weather_data(location=None, lat=None, lon=None):
    """Fetches current and forecast weather data from OpenWeatherMap API using either city name or coordinates."""
    try:
        if location:
            current_url = f"{BASE_WEATHER_URL}weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
            forecast_url = f"{BASE_WEATHER_URL}forecast?q={location}&appid={WEATHER_API_KEY}&units=metric"
        elif lat and lon:
            current_url = f"{BASE_WEATHER_URL}weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            forecast_url = f"{BASE_WEATHER_URL}forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        else:
            return None, "Error: Location not provided.", None

        current_response = requests.get(current_url)
        forecast_response = requests.get(forecast_url)

        current_response.raise_for_status()
        forecast_response.raise_for_status()

        current_data = current_response.json()
        forecast_data = forecast_response.json()['list']

        fetched_location = location
        # Get city name if coordinates were used
        if not location and lat and lon:
            geocode_url = f"{GEOCODING_API_URL}?lat={lat}&lon={lon}&limit=1&appid={WEATHER_API_KEY}"
            geocode_response = requests.get(geocode_url)
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()
            if geocode_data:
                fetched_location = f"{geocode_data[0]['name']}, {geocode_data[0]['country']}"

        return current_data, forecast_data, fetched_location
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching weather data: {e}", None
    except KeyError:
        return None, "Error parsing weather data. Invalid location or API response.", None

def format_weather(data):
    """Formats weather data into a readable dictionary."""
    if data:
        return {
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "precipitation": data.get('rain', {}).get('1h', 0) + data.get('snow', {}).get('1h', 0),
            "description": data['weather'][0]['description'],
            "wind_speed": data['wind']['speed']
        }
    return None

def get_weekly_forecast(forecast_data):
    """Extracts and formats the weather forecast for the next 7 days."""
    daily_forecasts = {}
    for item in forecast_data:
        timestamp = datetime.fromtimestamp(item['dt'])
        date = timestamp.date()
        if date not in daily_forecasts:
            daily_forecasts[date] = {
                "min_temp": float('inf'),
                "max_temp": float('-inf'),
                "humidity": 0,
                "precipitation": 0,
                "descriptions": [],
                "wind_speeds": []
            }
        daily_forecasts[date]['min_temp'] = min(daily_forecasts[date]['min_temp'], item['main']['temp'])
        daily_forecasts[date]['max_temp'] = max(daily_forecasts[date]['max_temp'], item['main']['temp'])
        daily_forecasts[date]['humidity'] += item['main']['humidity']
        daily_forecasts[date]['precipitation'] += item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0)
        daily_forecasts[date]['descriptions'].append(item['weather'][0]['description'])
        daily_forecasts[date]['wind_speeds'].append(item['wind']['speed'])

    weekly_data = {}
    for date, data in daily_forecasts.items():
        weekly_data[date.strftime('%Y-%m-%d')] = {
            "min_temperature": round(data['min_temp'], 1),
            "max_temperature": round(data['max_temp'], 1),
            "avg_humidity": round(data['humidity'] / len(data['descriptions'])),
            "total_precipitation": round(data['precipitation'], 1),
            "most_common_description": max(set(data['descriptions']), key=data['descriptions'].count),
            "avg_wind_speed": round(sum(data['wind_speeds']) / len(data['wind_speeds']), 1)
        }

    # Get the next 7 days (including today if data is available)
    today = datetime.now().date()
    next_7_days_data = {}
    for i in range(7):
        target_date = today + timedelta(days=i)
        date_str = target_date.strftime('%Y-%m-%d')
        if date_str in weekly_data:
            next_7_days_data[date_str] = weekly_data[date_str]
    return next_7_days_data

def get_crop_advice(crop, current_weather, weekly_forecast):
    """Provides crop-specific advice based on weather conditions."""
    advice = {"dos": [], "donts": []}

    if current_weather:
        if current_weather['temperature'] > 35:
            advice["dos"].append(f"For {crop}: Provide shade if possible, ensure adequate irrigation.")
            advice["donts"].append(f"For {crop}: Avoid heavy fertilization during peak heat.")
        elif current_weather['temperature'] < 10:
            advice["dos"].append(f"For {crop}: Consider frost protection measures like covering young plants.")
            advice["donts"].append(f"For {crop}: Avoid transplanting sensitive seedlings.")

        if current_weather['precipitation'] > 5:  # Heavy rain in the last hour
            advice["dos"].append(f"For {crop}: Ensure good drainage to prevent waterlogging.")
            advice["donts"].append(f"For {crop}: Avoid spraying pesticides or fertilizers as they might wash off.")
        elif "rain" in current_weather['description'].lower():
            advice["dos"].append(f"For {crop}: Monitor for signs of fungal diseases.")

        if "wind" in current_weather['description'].lower() or current_weather['wind_speed'] > 30:
            advice["dos"].append(f"For {crop}: Provide support for tall or weak-stemmed plants.")
            advice["donts"].append(f"For {crop}: Avoid pruning during high winds.")

    for date, forecast in weekly_forecast.items():
        if forecast['max_temperature'] > 35:
            advice["dos"].append(f"In the coming week for {crop}: Plan for increased irrigation.")
        if forecast['total_precipitation'] > 10: # Significant rainfall expected
            advice["dos"].append(f"In the coming week for {crop}: Prepare drainage systems and monitor for waterlogging.")
            advice["donts"].append(f"In the coming week for {crop}: Avoid planting or sowing if heavy rain is expected.")

    # Add more crop-specific advice based on different weather conditions
    if crop.lower() == "rice":
        if current_weather and current_weather['precipitation'] < 1:
            advice["dos"].append("For Rice: Ensure sufficient water levels in paddy fields.")
        if weekly_forecast and any(f['total_precipitation'] > 15 for f in weekly_forecast.values()):
            advice["donts"].append("For Rice: Be cautious of potential flooding in paddy fields.")
    elif crop.lower() == "wheat":
        if current_weather and current_weather['humidity'] > 80:
            advice["dos"].append("For Wheat: Monitor for fungal diseases like rust.")
        if weekly_forecast and any(f['max_temperature'] > 30 for f in weekly_forecast.values()):
            advice["donts"].append("For Wheat: High temperatures during grain filling can reduce yield.")
    elif crop.lower() in ["maize", "corn"]:
        if current_weather and current_weather['temperature'] < 15:
            advice["donts"].append("For Maize/Corn: Low temperatures can hinder germination and early growth.")
        if weekly_forecast and any(f['max_temperature'] > 40 for f in weekly_forecast.values()):
            advice["donts"].append("For Maize/Corn: Very high temperatures during pollination can reduce kernel set.")
        if current_weather and current_weather['wind_speed'] > 40:
            advice["dos"].append("For Maize/Corn: Inspect for stalk lodging after strong winds.")
    elif crop.lower() == "barley":
        if current_weather and current_weather['precipitation'] > 10:
            advice["donts"].append("For Barley: Excessive rainfall, especially during harvest, can lead to grain sprouting.")
        if weekly_forecast and any(f['max_temperature'] > 32 for f in weekly_forecast.values()):
            advice["donts"].append("For Barley: High temperatures during grain filling can reduce grain quality.")
    elif crop.lower() == "cotton":
        if current_weather and current_weather['precipitation'] > 15:
            advice["donts"].append("For Cotton: Heavy rainfall can damage open bolls and reduce fiber quality.")
        if weekly_forecast and any(f['min_temperature'] < 15 for f in weekly_forecast.values()):
            advice["donts"].append("For Cotton: Cool nights can negatively impact boll development.")
    elif crop.lower() in ["sugarcane", "cane"]:
        if current_weather and current_weather['temperature'] < 18:
            advice["donts"].append("For Sugarcane: Low temperatures can slow down growth.")
        if weekly_forecast and any(f['total_precipitation'] < 5 for f in weekly_forecast.values()):
            advice["dos"].append("For Sugarcane: Ensure adequate irrigation during dry spells.")
    elif crop.lower() == "potato":
        if current_weather and current_weather['precipitation'] > 20:
            advice["donts"].append("For Potato: Waterlogged conditions can lead to tuber rot.")
        if weekly_forecast and any(f['min_temperature'] < 5 for f in weekly_forecast.values()):
            advice["donts"].append("For Potato: Frost can damage potato foliage.")
    elif crop.lower() == "tomato":
        if current_weather and current_weather['humidity'] > 85:
            advice["dos"].append("For Tomato: Ensure good air circulation to prevent fungal diseases.")
            advice["donts"].append("For Tomato: Avoid overhead watering in humid conditions.")
        if weekly_forecast and any(f['min_temperature'] < 10 for f in weekly_forecast.values()):
            advice["donts"].append("For Tomato: Protect young plants from potential frost.")
    elif crop.lower() == "carrot":
        if current_weather and current_weather['precipitation'] > 15:
            advice["donts"].append("For Carrot: Excessive moisture can lead to root cracking.")
        if weekly_forecast and any(f['max_temperature'] > 30 for f in weekly_forecast.values()):
            advice["donts"].append("For Carrot: High soil temperatures can affect root development.")
    elif crop.lower() == "peas":
        if current_weather and current_weather['precipitation'] > 10:
            advice["donts"].append("For Peas: Waterlogged soil can cause root rot.")
        if weekly_forecast and any(f['max_temperature'] > 25 for f in weekly_forecast.values()):
            advice["donts"].append("For Peas: High temperatures during flowering and pod development can reduce yield.")

    # Add advice for other crops as needed

    return advice

def calculate_sustainability_score(practices, current_weather, weekly_forecast):
    """Calculates a sustainability score based on farming practices and weather."""
    score = 50  # Start with a base score
    max_possible_increase = 50
    increase = 0

    # --- Irrigation Practices ---
    irrigation = practices.get('irrigation_method')
    if irrigation == 'drip':
        increase += 10
    elif irrigation == 'sprinkler':
        increase += 5

    # --- Pesticide Use ---
    pesticide = practices.get('pesticide_use')
    if pesticide == 'organic':
        increase += 15
    elif pesticide == 'integrated_pest_management':
        increase += 10
    elif pesticide == 'minimal_chemical':
        increase += 5

    # --- Soil Management ---
    tillage = practices.get('tillage_practice')
    if tillage == 'no_till':
        increase += 10
    elif tillage == 'reduced':
        increase += 5

    cover_crops = practices.get('cover_crops')
    if cover_crops == 'yes':
        increase += 5

    organic_matter = practices.get('organic_matter')
    try:
        organic_matter_val = float(organic_matter)
        if organic_matter_val >= 1.0:
            increase += 5
        elif 0.5 <= organic_matter_val < 1.0:
            increase += 2
    except (ValueError, TypeError):
        pass

    crop_rotation = practices.get('rotation_diversity')
    try:
        rotation_val = int(crop_rotation)
        if rotation_val >= 3:
            increase += 5
        elif rotation_val == 2:
            increase += 2
    except (ValueError, TypeError):
        pass

    # --- Weather Considerations (Penalties for unsustainable responses) ---
    if current_weather:
        if current_weather['precipitation'] < 1 and irrigation == 'flood':
            score -= 2 # Inefficient water use in dry conditions

    if weekly_forecast and any(f['total_precipitation'] > 15 for f in weekly_forecast.values()):
        if practices.get('drainage') == 'poor':
            score -= 5 # Increased risk of waterlogging with poor drainage

    score += min(increase, max_possible_increase)
    return max(0, min(score, 100)) # Ensure score is within 0-100

def generate_sustainability_suggestions(practices, score, current_weather, weekly_forecast):
    """Generates sustainability improvement suggestions."""
    suggestions = []
    if score < 60:
        suggestions.append("Overall sustainability score is low. Consider implementing multiple of the following suggestions.")

    irrigation = practices.get('irrigation_method')
    if irrigation == 'flood' and score < 80:
        suggestions.append("Consider switching to more water-efficient irrigation methods like drip or sprinkler systems.")

    pesticide = practices.get('pesticide_use')
    if pesticide == 'chemical' and score < 70:
        suggestions.append("Explore Integrated Pest Management (IPM) strategies or consider using organic pesticides to reduce environmental impact.")
    elif pesticide == 'minimal_chemical' and score < 85:
        suggestions.append("Further reduce reliance on chemical pesticides by enhancing biological controls and preventative measures.")

    tillage = practices.get('tillage_practice')
    if tillage == 'conventional' and score < 75:
        suggestions.append("Adopting reduced or no-till farming practices can improve soil health and reduce erosion.")
    elif tillage == 'reduced' and score < 85:
        suggestions.append("Consider transitioning to no-till farming for maximum soil health benefits.")

    cover_crops = practices.get('cover_crops')
    if cover_crops == 'no' and score < 70:
            suggestions.append("Planting cover crops during off-seasons can protect your soil, improve fertility, and suppress weeds.")

    organic_matter = practices.get('organic_matter')
    try:
        organic_matter_val = float(organic_matter)
        if organic_matter_val < 0.5 and score < 70:
            suggestions.append("Consider practices to increase soil organic matter, such as adding compost or manure.")
    except (ValueError, TypeError):
        if score < 70:
            suggestions.append("Providing information about your soil organic matter can help us give more specific advice.")

    crop_rotation = practices.get('rotation_diversity')
    try:
        rotation_val = int(crop_rotation)
        if rotation_val <= 1 and score < 70:
            suggestions.append("Implement a more diverse crop rotation to improve soil health and reduce pest/disease pressure.")
        elif rotation_val == 2 and score < 80:
            suggestions.append("Increasing the diversity of your crop rotation can further enhance sustainability.")
    except (ValueError, TypeError):
        if score < 70:
            suggestions.append("Providing the number of crops in your rotation helps assess its diversity.")

    drainage = practices.get('drainage')
    if weekly_forecast and any(f['total_precipitation'] > 15 for f in weekly_forecast.values()):
        if drainage == 'poor' and score < 70:
            suggestions.append("With heavy rainfall expected, ensure your farm has adequate drainage to prevent waterlogging.")

    return suggestions

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/rotation")
def rotation_form():
    return render_template("rotation.html")

@app.route("/soil")
def soil_page():
    return render_template("soil.html")

@app.route("/weather")
def weather_page():
    return render_template("weather.html")


@app.route("/sustainability")
def sustainability_page():
    return render_template("sustainability.html")  # <- likely you want to show the form here



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
    crop = request.form.get("crop")
    location_option = request.form.get("location_option")
    manual_location = request.form.get("location")
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    map_latitude = request.form.get("map_latitude")
    map_longitude = request.form.get("map_longitude")

    if not crop:
        return render_template("weather_results.html", message="Please provide the crop being farmed.")

    fetched_location = None
    current_weather_data, forecast_data, fetched_location = get_weather_data(
        location=manual_location if location_option == 'manual' and manual_location else None,
        lat=latitude if location_option == 'live' and latitude else map_latitude if location_option == 'map' and map_latitude else None,
        lon=longitude if location_option == 'live' and longitude else map_longitude if location_option == 'map' and map_longitude else None
    )

    if current_weather_data and forecast_data:
        current_weather = format_weather(current_weather_data)
        weekly_forecast = get_weekly_forecast(forecast_data)
        crop_advice = get_crop_advice(crop, current_weather, weekly_forecast)

        return render_template(
            "weather_results.html",
            crop=crop,
            location=fetched_location if fetched_location else (manual_location if location_option == 'manual' else f"Latitude: {latitude}, Longitude: {longitude}" if location_option == 'live' else f"Latitude: {map_latitude}, Longitude: {map_longitude}" if location_option == 'map' else "Not Specified"),
            current_weather=current_weather,
            weekly_forecast=weekly_forecast,
            crop_advice=crop_advice,
        )
    else:
        return render_template("weather_results.html", message=forecast_data)
    





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

@app.route("/calculate_sustainability", methods=["POST"])
def calculate_sustainability():
    crop = request.form.get("crop")
    location_option = request.form.get("location_option")
    manual_location = request.form.get("location")
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    map_latitude = request.form.get("map_latitude")
    map_longitude = request.form.get("map_longitude")
    irrigation_method = request.form.get("irrigation_method")
    pesticide_use = request.form.get("pesticide_use")
    tillage_practice = request.form.get("tillage_practice")
    cover_crops = request.form.get("cover_crops")
    organic_matter = request.form.get("organic_matter")
    rotation_diversity = request.form.get("rotation_diversity")
    drainage = request.form.get("drainage")

    farming_practices = {
        "irrigation_method": irrigation_method,
        "pesticide_use": pesticide_use,
        "tillage_practice": tillage_practice,
        "cover_crops": cover_crops,
        "organic_matter": organic_matter,
        "rotation_diversity": rotation_diversity,
        "drainage": drainage,
    }

    fetched_location = None
    current_weather_data, forecast_data, fetched_location = get_weather_data(
        location=manual_location if location_option == 'manual' and manual_location else None,
    lat=latitude if location_option == 'live' and latitude else map_latitude if location_option == 'map' and map_latitude else None,
        lon=longitude if location_option == 'live' and longitude else map_longitude if location_option == 'map' and map_longitude else None
    )
    current_weather = format_weather(current_weather_data) if current_weather_data else None
    weekly_forecast = get_weekly_forecast(forecast_data) if forecast_data else None

    sustainability_score = calculate_sustainability_score(farming_practices, current_weather, weekly_forecast)
    sustainability_suggestions = generate_sustainability_suggestions(farming_practices, sustainability_score, current_weather, weekly_forecast)

    return render_template(
        "sustainability_report.html",
        sustainability_score=sustainability_score,
        sustainability_suggestions=sustainability_suggestions,
        practices=farming_practices,
        location=fetched_location if fetched_location else (manual_location if location_option == 'manual' else f"Latitude: {latitude}, Longitude: {longitude}" if location_option == 'live' else f"Latitude: {map_latitude}, Longitude: {map_longitude}" if location_option == 'map' else "Not Specified"),
        crop=crop
    )



if __name__ == "__main__":
    app.run(debug=True, port=8000)
