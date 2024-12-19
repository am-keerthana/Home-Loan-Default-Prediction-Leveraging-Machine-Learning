import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load and prepare data
print("Loading data...")
df = pd.read_csv('application_train.csv')

# Print class distribution
print("\nClass distribution in training data:")
print(df['TARGET'].value_counts(normalize=True))

# Select important features
features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH']

# Prepare features
X = df[features].copy()
y = df['TARGET']

# Convert DAYS features to positive years
X['DAYS_BIRTH'] = abs(X['DAYS_BIRTH']) / 365  # Convert to age in years
X['DAYS_EMPLOYED'] = abs(X['DAYS_EMPLOYED']) / 365  # Convert to employment length in years
X['DAYS_ID_PUBLISH'] = abs(X['DAYS_ID_PUBLISH']) / 365  # Convert to credit history length in years

# Calculate debt ratio and other derived features
X['debt_ratio'] = (X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']) * 100
X['credit_to_income'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
X['young_age_risk'] = (X['DAYS_BIRTH'] < 25).astype(int)
X['short_employment'] = (X['DAYS_EMPLOYED'] < 2).astype(int)
X['limited_credit_history'] = (X['DAYS_ID_PUBLISH'] < 1).astype(int)
X['high_debt_ratio'] = (X['debt_ratio'] > 50).astype(int)
X['high_credit_amount'] = (X['credit_to_income'] > 3).astype(int)

# Calculate risk score (0-100)
X['risk_score'] = (
    (X['young_age_risk'] * 20) +
    (X['short_employment'] * 20) +
    (X['limited_credit_history'] * 20) +
    (X['high_debt_ratio'] * 20) +
    (X['high_credit_amount'] * 20)
)

# Rename columns for clarity
feature_cols = ['income', 'credit_amount', 'age', 'employment_length', 'credit_history_length', 
                'debt_ratio', 'credit_to_income', 'young_age_risk', 'short_employment', 
                'limited_credit_history', 'high_debt_ratio', 'high_credit_amount', 'risk_score']
X.columns = feature_cols

# Handle missing values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print feature ranges
print("\nFeature ranges in training data:")
for col in X.columns:
    print(f"{col}:")
    print(f"  Min: {X[col].min():.2f}")
    print(f"  Max: {X[col].max():.2f}")
    print(f"  Mean: {X[col].mean():.2f}")
    print()

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights
n_samples = len(y)
n_classes = len(np.unique(y))
class_weights = {0: 1.0, 1: (n_samples / (n_classes * np.sum(y == 1)))}
print("\nClass weights:", class_weights)

# Train the model with custom class weights
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight=class_weights
)
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Print feature importances
print("\nFeature importances:")
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances)

# Save the model and scaler
print("\nSaving model and scaler...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Done! Model and scaler have been saved.")
