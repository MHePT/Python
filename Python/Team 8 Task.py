import pandas as pd
import numpy as np

data = pd.read_csv(r'weatherAUS.csv')

print("Sample Data:")
print(data.head())

data.info()

print("\n\nMissing values in each column:\n", data.isnull().sum(), end="\n\n")

# Drop non-informative columns
data_cleaned = data.drop(columns=['Date', 'Location'])
print("\nData after dropping non-informative columns:")
print(data_cleaned.head())

# Fill missing values with the mean for numeric columns and mode for categorical columns
for column in data_cleaned.columns:
    if data_cleaned[column].dtype == 'object':
        # Fill missing values with the mode for categorical columns
        mode_value = data_cleaned[column].mode()[0]
        data_cleaned[column] = data_cleaned[column].fillna(mode_value)
    else:
        # Fill missing values with the mean for numeric columns
        mean_value = data_cleaned[column].mean()
        data_cleaned[column] = data_cleaned[column].fillna(mean_value)

print("\n\nData after filling missing values:")
print(data_cleaned.head())

# Label encoding for categorical columns using factorize
for column in data_cleaned.select_dtypes(include=['object']).columns:
    data_cleaned[column], _ = pd.factorize(data_cleaned[column])

print("\n\nData after label encoding:")
print(data_cleaned.head())

# Normalize numeric columns to have zero mean and unit variance
for column in data_cleaned.select_dtypes(include=[np.number]).columns:
    data_cleaned[column] = (data_cleaned[column] - data_cleaned[column].mean()) / data_cleaned[column].std()

print("\n\nData after normalization:")
print(data_cleaned.head())

# Split data into features (X) and target (y)
X = data_cleaned.drop(columns=['RainTomorrow']).values
y = data_cleaned['RainTomorrow'].values.astype(int)  # Ensure target is integer

# Split the data manually (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Implement a simple model
majority_class = np.argmax(np.bincount(y_train))
y_pred = np.full(y_test.shape, majority_class)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)

print("\n\nCleaned Data Sample (first 5 rows):")
print(data_cleaned.head())
print(f"Model Accuracy: {accuracy * 100:.2f}%")