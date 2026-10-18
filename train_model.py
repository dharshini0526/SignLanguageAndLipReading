import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the collected dataset
data = pd.read_csv("sign_data.csv")

# 2. Check the dataset
print("Dataset shape:", data.shape)
print("\nSamples per sign:")
print(data["label"].value_counts())

# 3. Separate coordinates and labels
X = data.drop("label", axis=1)
y = data["label"]

# 4. Check that every sample has 63 features
if X.shape[1] != 63:
    raise ValueError("Expected 63 coordinate features.")

if y.nunique() < 2:
    raise ValueError("Collect samples for at least two labels.")

if y.value_counts().min() < 2:
    raise ValueError(
        "Collect at least two samples for every label."
    )

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Create the Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Predict signs in the test set
predictions = model.predict(X_test)

# 9. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))

# 10. Save the trained model
joblib.dump(model, "isl_sign_model.pkl")

print("\nModel saved as isl_sign_model.pkl")
