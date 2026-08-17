# ============================================================
# Demonstrating Input, Feature, Weight, Bias and Parameters
# using a simple Machine Learning model
#
# Example:
# Predict salary from years of experience
# ============================================================

# ------------------------------------------------------------
# 1. TRAINING DATA
# ------------------------------------------------------------
# Feature:
#   Years of experience
#
# Input:
#   The feature values provided to the model
#
# Target:
#   Actual salary corresponding to each input
# ------------------------------------------------------------

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

y = [
    35000,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    75000,
    80000
]


# ------------------------------------------------------------
# 2. INITIAL MODEL PARAMETERS
# ------------------------------------------------------------
# Weight and bias are initially unknown.
# We start with arbitrary values.
#
# Model:
#
# prediction = weight * input + bias
# ------------------------------------------------------------

weight = 0.0
bias = 0.0


# ------------------------------------------------------------
# 3. LEARNING RATE
# ------------------------------------------------------------
# Controls how much the model changes its parameters
# during each training iteration.
# ------------------------------------------------------------

learning_rate = 0.00001


# ------------------------------------------------------------
# 4. PREDICTION FUNCTION
# ------------------------------------------------------------

def predict(x, weight, bias):
    return weight * x + bias


# ------------------------------------------------------------
# 5. TRAINING
# ------------------------------------------------------------

epochs = 10000

for epoch in range(epochs):

    total_error = 0

    # Gradients for weight and bias
    dw = 0
    db = 0

    number_of_samples = len(X)

    # --------------------------------------------------------
    # Calculate prediction and error for every training sample
    # --------------------------------------------------------

    for i in range(number_of_samples):

        x = X[i]
        actual = y[i]

        # Prediction
        prediction = predict(x, weight, bias)

        # Error
        error = prediction - actual

        total_error += error ** 2

        # ----------------------------------------------------
        # Gradient calculations
        #
        # These tell us how the weight and bias should change.
        # ----------------------------------------------------

        dw += error * x
        db += error

    # Average gradients
    dw = (2 / number_of_samples) * dw
    db = (2 / number_of_samples) * db

    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    weight = weight - learning_rate * dw
    bias = bias - learning_rate * db

    # Print progress
    if epoch % 1000 == 0:
        mse = total_error / number_of_samples

        print(
            f"Epoch: {epoch:5d} | "
            f"Weight: {weight:10.2f} | "
            f"Bias: {bias:10.2f} | "
            f"MSE: {mse:12.2f}"
        )


# ------------------------------------------------------------
# 6. TRAINED MODEL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Learned Weight : {weight:.2f}")
print(f"Learned Bias   : {bias:.2f}")


# ------------------------------------------------------------
# 7. MAKE A PREDICTION
# ------------------------------------------------------------

while True:
    user_input = input("\nEnter year of experiance or press 'q' to exit: ").strip()

    if user_input == "q":
        break

    if not user_input.isdigit():
        print("Input value should be a positive intiger!")
        continue

    experience = int(user_input)

    predicted_salary = predict(
        experience,
        weight,
        bias
    )

    print("\nPrediction")
    print("-" * 60)

    print(f"Experience : {experience} years")
    print(f"Predicted Salary : ₹{predicted_salary:,.2f}")


    # ------------------------------------------------------------
    # 8. DISPLAY THE MODEL EQUATION
    # ------------------------------------------------------------

    print("\nLearned Model")
    print("-" * 60)

    print(
        f"Salary = {weight:.2f} × Experience + {bias:.2f}"
    )
    print("-" * 60)