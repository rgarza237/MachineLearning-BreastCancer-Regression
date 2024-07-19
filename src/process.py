import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


file_path = '../data/raw/Breast Cancer Wisconsin.csv'
data = pd.read_csv(file_path)

#show data
print("First few rows of the dataset:")
print(data.head())

#print missing values
print("\nMissing values in the dataset:")
missing_values = data.isnull().sum()
print(missing_values)

# Drop the 'id' column and the last column 'Unnamed: 32' which appears to be mostly NaN
data_cleaned = data.drop(['id', 'Unnamed: 32'], axis=1)

# Encode the 'diagnosis' column (M = 1, B = 0)
label_encoder = LabelEncoder()
data_cleaned['diagnosis'] = label_encoder.fit_transform(data_cleaned['diagnosis'])

# Scale the numerical features
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_cleaned.drop('diagnosis', axis=1)), columns=data_cleaned.columns[1:])
data_scaled['diagnosis'] = data_cleaned['diagnosis']

# Split the data into features (X) and labels (y)
X = data_scaled.drop('diagnosis', axis=1)
y = data_scaled['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Export the training and testing sets to CSV files
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)

# Display the shapes of the training and testing sets
print("\nShapes of the training and testing sets:")
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)
