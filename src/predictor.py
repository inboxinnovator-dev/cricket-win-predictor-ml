import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load historical dataset
df = pd.read_csv("data/historical_match_data.csv")

# Feature selection
X = df[
    [
        "runs_left",
        "balls_left",
        "wickets_left",
        "current_run_rate",
        "required_run_rate"
    ]
]

# Target variable
y = df["result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save metrics
with open("output/model_metrics.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.2f}")

print("Model trained successfully")
print("Accuracy:", accuracy)

# Predict a new match situation
new_match = [[60, 36, 6, 7.5, 10.0]]
win_probability = model.predict_proba(new_match)

print("Win Probability:", win_probability)
