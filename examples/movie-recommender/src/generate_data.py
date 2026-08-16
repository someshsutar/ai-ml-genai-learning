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
