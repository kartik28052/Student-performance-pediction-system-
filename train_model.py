"""Step 2: Train the AI model and save it.

The model learns from students.csv and predicts whether a student's
performance will be Low, Medium or High.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("students.csv")

FEATURES = ["attendance", "study_hours", "prev_marks", "assignments", "sleep_hours"]
X = df[FEATURES]            # inputs
y = df["performance"]       # what we want to predict

# 80% of data to learn from, 20% to test on (data the model has never seen)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 1), "%")
print(classification_report(y_test, pred))

print("Which factors matter most:")
for name, score in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name}: {score:.2f}")

joblib.dump(model, "model.pkl")
print("Saved model.pkl")