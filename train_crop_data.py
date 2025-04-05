import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your dataset
df = pd.read_csv("crop_rotation_data.csv")

# Encode each input column
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_season = LabelEncoder()
le_target = LabelEncoder()

X = pd.DataFrame({
    "last_crop": le_crop.fit_transform(df["last_crop"]),
    "soil_type": le_soil.fit_transform(df["soil_type"]),
    "season": le_season.fit_transform(df["season"])
})
y = le_target.fit_transform(df["recommended_crop"])

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save model + label encoders to file
with open("rotation_model.pkl", "wb") as f:
    pickle.dump((model, le_crop, le_soil, le_season, le_target), f)

print("✅ Model trained and saved as rotation_model.pkl")
