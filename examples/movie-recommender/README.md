Yes. I’d turn the earlier example into a small **end-to-end ML recommendation project** rather than just one Python script.

The project will:

* Generate a reasonably large synthetic movie-rating dataset.
* Save the dataset as CSV.
* Train a **collaborative-filtering ML model** using matrix factorization.
* Use **NumPy + Pandas + scikit-learn**.
* Evaluate the model using RMSE.
* Save the trained model.
* Provide an interactive CLI where you can:

  * list users
  * show a user's ratings
  * get recommendations
  * rate a movie
  * retrain the model
  * inspect similar users
  * exit
* Demonstrate the underlying linear algebra rather than hiding everything behind a library.

One useful distinction: **matrix factorization is the ML model here**. The original cosine-similarity example is primarily a similarity-based recommendation algorithm; this expanded version actually learns latent user/movie representations from the data.

## Project structure

```text
movie-recommender/
│
├── data/
│   └── ratings.csv
│
├── models/
│   └── recommender.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── generate_data.py
│   ├── data_loader.py
│   ├── recommender.py
│   ├── train.py
│   └── cli.py
│
├── requirements.txt
├── README.md
└── main.py
```

### Dataset

We'll generate approximately:

* **1,000 users**
* **500 movies**
* **50,000+ ratings**

The data will look like:

```text
user_id,movie_id,rating
U0001,M0123,4
U0001,M0445,5
U0001,M0102,2
U0002,M0007,5
...
```

---

# 1. Create the project

```bash
mkdir movie-recommender
cd movie-recommender

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Create directories:

```bash
mkdir -p data models src
touch src/__init__.py
```

---

# 2. requirements.txt

```text
numpy>=2.0
pandas>=2.2
scikit-learn>=1.5
joblib>=1.4
```

Install:

```bash
pip install -r requirements.txt
```

---

# 3. Configuration

Create:

`src/config.py`

```python
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

RATINGS_FILE = DATA_DIR / "ratings.csv"
MODEL_FILE = MODEL_DIR / "recommender.pkl"

# Dataset size
NUM_USERS = 1000
NUM_MOVIES = 500

# Approximately this many ratings will be generated
NUM_RATINGS = 60000

# Rating scale
MIN_RATING = 1
MAX_RATING = 5

# Matrix factorization parameters
LATENT_FACTORS = 20
LEARNING_RATE = 0.01
REGULARIZATION = 0.02
EPOCHS = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

---

# 4. Generate a larger dataset

This is particularly interesting because we don't want completely random ratings.

If ratings were entirely random, there would be no meaningful pattern for the ML model to learn.

Instead, we'll generate hidden **latent preferences**.

For example:

```text
User
 ├── Action preference
 ├── Comedy preference
 ├── Drama preference
 └── Sci-Fi preference
```

Movies will also have latent characteristics.

The rating is generated approximately as:

[
r_{ui} = \mu + b_u + b_i + U_u \cdot V_i
]

where:

* (U_u) = user latent vector
* (V_i) = movie latent vector
* (b_u) = user bias
* (b_i) = movie bias
* (\mu) = global average rating

This is already very close to the model we will train.

Create:

`src/generate_data.py`

```python
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    RATINGS_FILE,
    NUM_USERS,
    NUM_MOVIES,
    NUM_RATINGS,
    MIN_RATING,
    MAX_RATING,
    LATENT_FACTORS,
    RANDOM_STATE,
)


def generate_dataset():
    """
    Generate a synthetic movie-rating dataset.

    The ratings are generated from hidden user/movie
    latent factors so that a machine-learning model
    has meaningful patterns to learn.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Hidden user and movie representations
    # -----------------------------------------------------

    user_factors = rng.normal(
        0,
        1,
        size=(NUM_USERS, LATENT_FACTORS),
    )

    movie_factors = rng.normal(
        0,
        1,
        size=(NUM_MOVIES, LATENT_FACTORS),
    )

    # User and movie biases
    user_bias = rng.normal(
        0,
        0.3,
        size=NUM_USERS,
    )

    movie_bias = rng.normal(
        0,
        0.3,
        size=NUM_MOVIES,
    )

    global_mean = 3.5

    # -----------------------------------------------------
    # Generate unique user/movie combinations
    # -----------------------------------------------------

    total_possible = NUM_USERS * NUM_MOVIES

    if NUM_RATINGS > total_possible:
        raise ValueError(
            "NUM_RATINGS cannot exceed "
            "NUM_USERS * NUM_MOVIES"
        )

    combinations = rng.choice(
        total_possible,
        size=NUM_RATINGS,
        replace=False,
    )

    user_indices = combinations // NUM_MOVIES
    movie_indices = combinations % NUM_MOVIES

    # -----------------------------------------------------
    # Generate ratings
    # -----------------------------------------------------

    interaction_scores = np.sum(
        user_factors[user_indices]
        * movie_factors[movie_indices],
        axis=1,
    )

    ratings = (
        global_mean
        + user_bias[user_indices]
        + movie_bias[movie_indices]
        + interaction_scores * 0.35
        + rng.normal(0, 0.35, size=NUM_RATINGS)
    )

    ratings = np.clip(
        np.rint(ratings),
        MIN_RATING,
        MAX_RATING,
    ).astype(int)

    # -----------------------------------------------------
    # Convert to DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame(
        {
            "user_id": [
                f"U{i + 1:04d}"
                for i in user_indices
            ],
            "movie_id": [
                f"M{i + 1:04d}"
                for i in movie_indices
            ],
            "rating": ratings,
        }
    )

    df.to_csv(
        RATINGS_FILE,
        index=False,
    )

    print("Dataset generated successfully.")
    print(f"Users:   {df['user_id'].nunique()}")
    print(f"Movies:  {df['movie_id'].nunique()}")
    print(f"Ratings: {len(df)}")
    print(f"Saved:   {RATINGS_FILE}")

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    return df


if __name__ == "__main__":
    generate_dataset()
```

Run:

```bash
python -m src.generate_data
```

You should get output similar to:

```text
Dataset generated successfully.
Users:   1000
Movies:  500
Ratings: 60000

Rating distribution:

1     ...
2     ...
3     ...
4     ...
5     ...
```

---

# 5. Data loader

Create:

`src/data_loader.py`


# 6. The actual ML model

Now we reach the important part.

We'll implement **Matrix Factorization using SGD** ourselves.

The model assumes:

$$
R \approx UV^T
$$

where:

$$
U =
\begin{bmatrix}
u_1\\
u_2\\
\vdots\\
u_m
\end{bmatrix}
$$

contains user embeddings, and:

$$
V =
\begin{bmatrix}
v_1\\
v_2\\
\vdots\\
v_n
\end{bmatrix}
$$

contains movie embeddings.

The predicted rating is:

$$
\hat r_{ui} = \mu + b_u + b_i + U_u \cdot V_i
$$

The error is:

$$
e_{ui}=r_{ui}-\hat r_{ui}
$$

Then we update the parameters using gradient descent.

This connects directly to the **vectors, dot product, gradients and optimization** topics you've been studying.

Create:

`src/recommender.py`


---

# 7. Training program

Create:

`src/train.py`

Run:

```bash
python -m src.train
```

You should see something like:

```text
Loading dataset...
Ratings loaded: 60,000

Training Matrix Factorization Model
============================================================

Epoch 01/30 | Train RMSE: 0.91 | Test RMSE: 0.88
Epoch 02/30 | Train RMSE: 0.82 | Test RMSE: 0.80
Epoch 03/30 | Train RMSE: 0.76 | Test RMSE: 0.75
...
Epoch 30/30 | Train RMSE: 0.45 | Test RMSE: 0.55

Training complete.

Model saved to:
models/recommender.pkl
```

The exact numbers will vary.

---

# 8. Interactive CLI

Now we make the project interactive.

The user will see:

```text
==================================================
      MOVIE RECOMMENDATION SYSTEM
==================================================

1. List users
2. Show user ratings
3. Recommend movies
4. Find similar users
5. Train model
6. Dataset statistics
7. Exit
```

Create:

`src/cli.py`


---

# 9. Main entry point

Create:

`main.py`

Now the entire application can be launched with:

```bash
python main.py
```

---

# 10. Running the complete project

The first time:

```bash
python -m src.generate_data
```

Then:

```bash
python -m src.train
```

Then:

```bash
python main.py
```

You should get:

```text
======================================================================
MOVIE RECOMMENDATION SYSTEM
======================================================================

1. List users
2. Show user ratings
3. Recommend movies
4. Find similar users
5. Train model
6. Dataset statistics
7. Exit

Select an option:
```

---

# 11. Try the recommendation

Choose:

```text
3
```

Then:

```text
Enter user ID (e.g. U0001): U0001

How many recommendations? [10]: 5
```

Example:

```text
Top 5 recommendations for U0001
------------------------------------------------------------

 1. M0321      Predicted Rating: 4.71
 2. M0187      Predicted Rating: 4.65
 3. M0412      Predicted Rating: 4.59
 4. M0098      Predicted Rating: 4.53
 5. M0274      Predicted Rating: 4.49
```

Because the dataset is synthetic, the movie IDs don't have meaningful names yet. That's intentional for the first version: we're concentrating on the **ML mechanics**.

---

# 12. Where is the linear algebra?

This is the most important part of the project.

The model contains:

```python
np.dot(
    self.user_factors[user_index],
    self.movie_factors[movie_index],
)
```

Mathematically:

$$
\hat r_{ui}
=
\mu+b_u+b_i+
U_u\cdot V_i
$$

The dot product is:

$$
U_u\cdot V_i
=
\sum_{k=1}^{K}
U_{uk}V_{ik}
$$

If we have 20 latent factors:

```text
User U0001

[0.31, -0.12, 0.71, ..., 0.22]
                    ↑
                20 values
```

and:

```text
Movie M0123

[0.42, 0.51, -0.13, ..., 0.17]
                    ↑
                20 values
```

the dot product gives us a measure of how well the user's learned preferences match the movie's learned characteristics.

---

# 13. Matrix representation

After training, our model has learned:

```text
User Factors

1000 × 20
```

and:

```text
Movie Factors

500 × 20
```

So:

$$
U \in R^{1000\times20}
$$

and:

$$
V \in R^{500\times20}
$$

The reconstructed rating matrix is:

$$
\hat R = UV^T
$$

Therefore:

$$
(1000\times20)
(20\times500)
=
1000\times500
$$

That is pure **matrix multiplication**.

This is the core linear algebra behind matrix-factorization recommendation.

---

# 14. Why do we call this machine learning?

Because we don't explicitly tell the model:

```text
Alice likes science fiction.
Bob likes action.
Charlie likes drama.
```

Instead, we give it observed ratings:

```text
Alice → Matrix → 5
Alice → Titanic → 2

Bob → Matrix → 5
Bob → Titanic → 1

...
```

The algorithm learns the hidden representations:

```text
Users
     ↓
Latent Factors
     ↓
Movies
```

through optimization.

The model minimizes:

$$
\sum_{(u,i)}
(r_{ui}-\hat r_{ui})^2
$$

plus regularization:

$$
\lambda
(
||U||^2 + ||V||^2
)
$$

This is an actual ML training objective.

---

# 15. Gradient descent

For each rating:

```text
Actual rating
      ↓
Model prediction
      ↓
Calculate error
      ↓
Calculate gradient
      ↓
Update user vector
      ↓
Update movie vector
```

For example:

```python
self.user_factors[user_index] += (
    self.learning_rate
    * (
        error * movie_vector
        - self.regularization
        * user_vector
    )
)
```

That's gradient descent.

So this one project demonstrates:

```text
Linear Algebra
      │
      ├── Vectors
      ├── Matrices
      ├── Dot Product
      ├── Matrix Multiplication
      ├── Norm
      └── Cosine Similarity
             │
             ▼
       Machine Learning
             │
             ├── Loss
             ├── Gradient
             ├── Gradient Descent
             ├── Regularization
             └── Model Evaluation
                     │
                     ▼
             Recommendation System
```

---
