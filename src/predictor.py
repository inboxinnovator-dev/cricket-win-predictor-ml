import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/match_data.csv")

# Separate features and target
X = df[["Runs", "Wickets", "Overs", "Target"]]
y = df["Result"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)

# Save model performance
with open("output/model_metrics.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.2f}")

print("Model trained successfully")
print("Accuracy:", accuracy)

