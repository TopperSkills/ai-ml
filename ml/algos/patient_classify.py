import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset directly from an online source
url = "https://githubusercontent.com"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

# 2. Separate Data into Features (X) and Target Label (y)
X = df.drop(columns=['Outcome'])  # Everything except the outcome
y = df['Outcome']                 # Only the outcome (0 or 1)

# 3. Split the data into Training set (80%) and Testing set (20%)
# This ensures we have unseen data left over to test if the model actually learned!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing: Scale the features
# Algorithms perform much better when all numbers are on a similar scale (e.g., between -3 and 3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize the AI Algorithm
# We are using "Random Forest", a highly powerful algorithm made of multiple decision trees
algorithm = RandomForestClassifier(random_state=42)

# 6. MODEL TRAINING (The .fit() magic!)
# The algorithm studies X_train_scaled and y_train to find the patterns. 
# Once this line finishes executing, 'algorithm' becomes our trained "Model".
model = algorithm.fit(X_train_scaled, y_train)

# 7. Make Predictions on Unseen Test Data
predictions = model.predict(X_test_scaled)

# 8. Evaluate how smart our model is
accuracy = accuracy_score(y_test, predictions)
print(f"🎉 Model Training Complete! Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Performance Report:")
print(classification_report(y_test, predictions))
