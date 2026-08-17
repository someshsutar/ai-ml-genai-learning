import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# Step 1: Define Inputs (Dataset)
# -----------------------------
# Features: [study_hours, attendance(%), previous_grade]
X = np.array([
    [5, 90, 70],
    [3, 80, 65],
    [8, 95, 85],
    [2, 60, 50],
    [6, 85, 75]
])

# Target (Upcoming Score)
y = np.array([75, 68, 90, 55, 80])

# -----------------------------
# Step 2: Train Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Step 3: Inspect Parameters
# -----------------------------
weights = model.coef_   # learned weights for each feature
bias = model.intercept_ # learned bias

print("Weights (importance of each feature):", weights)
print("Bias (constant offset):", bias)

# -----------------------------
# Step 4: Make Prediction
# -----------------------------

new_student = np.array([[4, 85, 72]])
predicted_score = model.predict(new_student)

print("Predicted upcoming score:", predicted_score[0])
