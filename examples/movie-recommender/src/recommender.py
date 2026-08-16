from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.config import (
    LATENT_FACTORS,
    LEARNING_RATE,
    REGULARIZATION,
    EPOCHS,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_FILE,
)


@dataclass
class RecommendationModel:

    latent_factors: int = LATENT_FACTORS
    learning_rate: float = LEARNING_RATE
    regularization: float = REGULARIZATION
    epochs: int = EPOCHS

    def __post_init__(self):

        self.user_to_index = {}
        self.movie_to_index = {}

        self.index_to_user = {}
        self.index_to_movie = {}

        self.user_factors = None
        self.movie_factors = None

        self.user_bias = None
        self.movie_bias = None

        self.global_mean = 0.0

        self.trained = False

    # -----------------------------------------------------
    # Build ID mappings
    # -----------------------------------------------------

    def _create_mappings(self, df):

        users = sorted(
            df["user_id"].unique()
        )

        movies = sorted(
            df["movie_id"].unique()
        )

        self.user_to_index = {
            user: index
            for index, user in enumerate(users)
        }

        self.movie_to_index = {
            movie: index
            for index, movie in enumerate(movies)
        }

        self.index_to_user = {
            index: user
            for user, index
            in self.user_to_index.items()
        }

        self.index_to_movie = {
            index: movie
            for movie, index
            in self.movie_to_index.items()
        }

    # -----------------------------------------------------
    # Initialize model
    # -----------------------------------------------------

    def _initialize_parameters(self):

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        num_users = len(
            self.user_to_index
        )

        num_movies = len(
            self.movie_to_index
        )

        self.user_factors = rng.normal(
            0,
            0.1,
            size=(
                num_users,
                self.latent_factors,
            ),
        )

        self.movie_factors = rng.normal(
            0,
            0.1,
            size=(
                num_movies,
                self.latent_factors,
            ),
        )

        self.user_bias = np.zeros(
            num_users
        )

        self.movie_bias = np.zeros(
            num_movies
        )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    def predict_single(
        self,
        user_index,
        movie_index,
    ):

        prediction = (
            self.global_mean
            + self.user_bias[user_index]
            + self.movie_bias[movie_index]
            + np.dot(
                self.user_factors[user_index],
                self.movie_factors[movie_index],
            )
        )

        return float(
            np.clip(prediction, 1, 5)
        )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    def fit(self, df):

        self._create_mappings(df)

        self._initialize_parameters()

        self.global_mean = df["rating"].mean()

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        train_records = list(
            train_df.itertuples(
                index=False
            )
        )

        test_records = list(
            test_df.itertuples(
                index=False
            )
        )

        print("\nTraining Matrix Factorization Model")
        print("=" * 60)

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            np.random.shuffle(
                train_records
            )

            total_error = 0.0

            for record in train_records:

                user_id = record.user_id
                movie_id = record.movie_id
                rating = float(record.rating)

                user_index = (
                    self.user_to_index[user_id]
                )

                movie_index = (
                    self.movie_to_index[movie_id]
                )

                prediction = self.predict_single(
                    user_index,
                    movie_index,
                )

                error = rating - prediction

                total_error += error ** 2

                user_vector = (
                    self.user_factors[user_index]
                    .copy()
                )

                movie_vector = (
                    self.movie_factors[movie_index]
                    .copy()
                )

                # -----------------------------------------
                # Gradient descent updates
                # -----------------------------------------

                self.user_bias[user_index] += (
                    self.learning_rate
                    * (
                        error
                        - self.regularization
                        * self.user_bias[user_index]
                    )
                )

                self.movie_bias[movie_index] += (
                    self.learning_rate
                    * (
                        error
                        - self.regularization
                        * self.movie_bias[movie_index]
                    )
                )

                self.user_factors[user_index] += (
                    self.learning_rate
                    * (
                        error * movie_vector
                        - self.regularization
                        * user_vector
                    )
                )

                self.movie_factors[movie_index] += (
                    self.learning_rate
                    * (
                        error * user_vector
                        - self.regularization
                        * movie_vector
                    )
                )

            train_rmse = np.sqrt(
                total_error
                / len(train_records)
            )

            test_rmse = self._calculate_rmse(
                test_records
            )

            print(
                f"Epoch {epoch:02d}/{self.epochs} "
                f"| Train RMSE: {train_rmse:.4f} "
                f"| Test RMSE: {test_rmse:.4f}"
            )

        self.trained = True

        print("\nTraining complete.")

        return self

    # -----------------------------------------------------
    # RMSE
    # -----------------------------------------------------

    def _calculate_rmse(
        self,
        records,
    ):

        actual = []
        predicted = []

        for record in records:

            user_id = record.user_id
            movie_id = record.movie_id

            if (
                user_id not in self.user_to_index
                or movie_id not in self.movie_to_index
            ):
                continue

            user_index = (
                self.user_to_index[user_id]
            )

            movie_index = (
                self.movie_to_index[movie_id]
            )

            prediction = self.predict_single(
                user_index,
                movie_index,
            )

            actual.append(
                float(record.rating)
            )

            predicted.append(
                prediction
            )

        if not actual:
            return 0.0

        return np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )

    # -----------------------------------------------------
    # Recommend movies
    # -----------------------------------------------------

    def recommend(
        self,
        user_id,
        ratings_df,
        top_n=10,
    ):

        if user_id not in self.user_to_index:
            raise ValueError(
                f"Unknown user: {user_id}"
            )

        user_index = (
            self.user_to_index[user_id]
        )

        watched_movies = set(
            ratings_df.loc[
                ratings_df["user_id"] == user_id,
                "movie_id",
            ]
        )

        recommendations = []

        for movie_id, movie_index in (
            self.movie_to_index.items()
        ):

            if movie_id in watched_movies:
                continue

            score = self.predict_single(
                user_index,
                movie_index,
            )

            recommendations.append(
                (
                    movie_id,
                    score,
                )
            )

        recommendations.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return recommendations[:top_n]

    # -----------------------------------------------------
    # Similar users
    # -----------------------------------------------------

    def similar_users(
        self,
        user_id,
        top_n=5,
    ):

        if user_id not in self.user_to_index:
            raise ValueError(
                f"Unknown user: {user_id}"
            )

        user_index = (
            self.user_to_index[user_id]
        )

        user_vector = (
            self.user_factors[user_index]
        )

        user_norm = np.linalg.norm(
            user_vector
        )

        similarities = []

        for index, other_vector in enumerate(
            self.user_factors
        ):

            if index == user_index:
                continue

            other_norm = np.linalg.norm(
                other_vector
            )

            if (
                user_norm == 0
                or other_norm == 0
            ):
                similarity = 0
            else:
                similarity = (
                    np.dot(
                        user_vector,
                        other_vector,
                    )
                    / (
                        user_norm
                        * other_norm
                    )
                )

            similarities.append(
                (
                    self.index_to_user[index],
                    similarity,
                )
            )

        similarities.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return similarities[:top_n]

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    def save(self):

        MODEL_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self,
            MODEL_FILE,
        )

        print(
            f"Model saved to: {MODEL_FILE}"
        )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    @staticmethod
    def load():

        return joblib.load(
            MODEL_FILE
        )
