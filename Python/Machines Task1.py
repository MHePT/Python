import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

weather_data = pd.read_csv('weatherAUS.csv')

# check and display first 5 records
print(weather_data.head())

# Now, let's check the data types and explore data using info()
weather_data.info()

# Drop columns with excessive missing values
weather_data_cleaned = weather_data.drop(columns=["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm", "Date"])

# Drop rows where the target column 'RainTomorrow' has missing values
weather_data_cleaned = weather_data_cleaned.dropna(subset=['RainTomorrow'])

print(weather_data_cleaned.head())

# Fill missing values in numeric columns with median values
numeric_columns = weather_data_cleaned.select_dtypes(include=["float64"]).columns
weather_data_cleaned[numeric_columns] = weather_data_cleaned[numeric_columns].fillna(weather_data_cleaned[numeric_columns].median())

# Fill missing values in categorical columns with the most frequent value
categorical_columns = weather_data_cleaned.select_dtypes(include=["object"]).columns.difference(["Date", "RainTomorrow"])
weather_data_cleaned[categorical_columns] = weather_data_cleaned[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))

print(weather_data_cleaned.head())

# Convert 'RainToday' and 'RainTomorrow' to binary (1 for 'Yes', 0 for 'No')
weather_data_cleaned['RainToday'] = weather_data_cleaned['RainToday'].map({'Yes': 1, 'No': 0})
weather_data_cleaned['RainTomorrow'] = weather_data_cleaned['RainTomorrow'].map({'Yes': 1, 'No': 0})

print(weather_data_cleaned.head())

# Use one-hot encoding for the remaining categorical columns
weather_data_encoded = pd.get_dummies(weather_data_cleaned, columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"], drop_first=True)

print(weather_data_encoded.head())

# Separate features and target variable
X = weather_data_encoded.drop(columns=["RainTomorrow"])
y = weather_data_encoded["RainTomorrow"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Accuracy = ",accuracy*100)

