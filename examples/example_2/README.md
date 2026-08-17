## Conceptual Breakdown

- **Inputs**: The raw data we feed into the model (study hours, attendance, previous grade).
- **Features**: Each measurable attribute of the input (e.g., study hours = 5, attendance = 90%).
- **Weights**: Numbers the model learns to scale each feature’s importance.
- **Bias**: A constant offset that shifts predictions up or down.
- **Parameters**: The combination of weights + bias that define the model.

Mathematically, prediction is:

$$
y = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b
$$

where $x_1, x_2, x_3$ are features, $w_i$ are weights, and $b$ is bias.

---

## Python Code Example

```python
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
# Example: student with 4 study hours, 85% attendance, previous grade 72
new_student = np.array([[4, 85, 72]])
predicted_score = model.predict(new_student)

print("Predicted upcoming score:", predicted_score[0])
```

---

## Explanation of Each Step

1. **Inputs**: The dataset `X` contains rows of student data. Each row = one student.
2. **Features**: Columns in `X` → study hours, attendance, previous grade.
3. **Weights**: Learned values in `model.coef_` show how strongly each feature affects the score.
4. **Bias**: `model.intercept_` adjusts predictions when features alone aren’t enough.
5. **Parameters**: Together, weights + bias define the linear function mapping inputs to predictions.
6. **Prediction**: For a new student, the model multiplies features by weights, adds bias, and outputs the score.

---

## Why This Matters
This simple example shows how **linear algebra (matrix multiplication)** underpins ML:
- Training = solving optimization problems to find best weights/bias.
- Prediction = dot product of features and weights + bias.

---