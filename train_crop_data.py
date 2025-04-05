import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib


# Load the Excel file to examine its structure
file_path = "soil_data.xlsx"
soil_data = pd.read_excel(file_path)

# Display the first few rows and column names
#print(soil_data.head(), soil_data.columns)# Create a function to assign soil health labels based on defined rules
def classify_soil_health(row):
    score = 0
    if 6.0 <= row['pH'] <= 7.5:
        score += 1
    if row['O.M. %'] > 1.0:
        score += 1
    if row['N_NO3 ppm'] > 10:
        score += 1
    if row['P ppm'] > 15:
        score += 1
    if row['K ppm '] > 150:
        score += 1

    if score <= 2:
        return 'Poor'
    elif score <= 4:
        return 'Moderate'
    else:
        return 'Good'

# Apply the classification function to the DataFrame
soil_data['Soil_Health_Label'] = soil_data.apply(classify_soil_health, axis=1)

# Show a sample of the labeled dataset
#print(soil_data[['pH', 'O.M. %', 'N_NO3 ppm', 'P ppm', 'K ppm ', 'Soil_Health_Label']].head(10))


# Step 1: Select features and target
features = ['pH', 'O.M. %', 'N_NO3 ppm', 'P ppm', 'K ppm ']
X = soil_data[features]
y = soil_data['Soil_Health_Label']

# Step 2: Encode the target labels (Good, Moderate, Poor → numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 6: Save the model and label encoder to use in Flask later
joblib.dump(clf, 'soil_health_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')


