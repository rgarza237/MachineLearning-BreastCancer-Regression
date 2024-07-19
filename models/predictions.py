import pandas as pd
import joblib

# Load the model from the file
model_filename = 'logistic_regression_model.joblib'
lr = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Load the test data (or new data)
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Make predictions
y_test_pred = lr.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
test_accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)

print("\nLogistic Regression Model Performance on Test Data:")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
