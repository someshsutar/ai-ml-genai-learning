import os

from src.config import MODEL_FILE
from src.data_loader import load_ratings
from src.recommender import RecommendationModel
from src.train import main as train_model


def clear_screen():

    os.system("cls" if os.name == "nt" else "clear")


def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def list_users(df):

    print_header("USERS")

    users = sorted(
        df["user_id"].unique()
    )

    for i in range(0, len(users), 10):

        print(
            " | ".join(
                users[i:i + 10]
            )
        )


def show_user_ratings(df):

    print_header("USER RATINGS")

    user_id = input(
        "Enter user ID (e.g. U0001): "
    ).strip().upper()

    user_ratings = df[
        df["user_id"] == user_id
    ]

    if user_ratings.empty:

        print(
            f"No ratings found for {user_id}"
        )

        return

    user_ratings = user_ratings.sort_values(
        "rating",
        ascending=False,
    )

    print(
        f"\nRatings for {user_id}"
    )

    print("-" * 50)

    for row in user_ratings.itertuples():

        print(
            f"{row.movie_id:<10} "
            f"Rating: {row.rating}"
        )


def recommend_movies(
    model,
    df,
):

    print_header("MOVIE RECOMMENDATIONS")

    user_id = input(
        "Enter user ID (e.g. U0001): "
    ).strip().upper()

    try:

        top_n_input = input(
            "How many recommendations? [10]: "
        ).strip()

        top_n = (
            int(top_n_input)
            if top_n_input
            else 10
        )

        recommendations = model.recommend(
            user_id,
            df,
            top_n,
        )

    except ValueError as exc:

        print(f"\nError: {exc}")

        return

    print(
        f"\nTop {len(recommendations)} "
        f"recommendations for {user_id}"
    )

    print("-" * 60)

    for rank, (movie, score) in enumerate(
        recommendations,
        start=1,
    ):

        print(
            f"{rank:2}. "
            f"{movie:<10} "
            f"Predicted Rating: {score:.2f}"
        )


def similar_users(model):

    print_header("SIMILAR USERS")

    user_id = input(
        "Enter user ID: "
    ).strip().upper()

    try:

        similar = model.similar_users(
            user_id
        )

    except ValueError as exc:

        print(f"\nError: {exc}")

        return

    print(
        f"\nUsers similar to {user_id}"
    )

    print("-" * 50)

    for rank, (user, score) in enumerate(
        similar,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{user:<10} "
            f"Similarity: {score:.4f}"
        )


def dataset_statistics(df):

    print_header(
        "DATASET STATISTICS"
    )

    print(
        f"Users:        "
        f"{df['user_id'].nunique():,}"
    )

    print(
        f"Movies:       "
        f"{df['movie_id'].nunique():,}"
    )

    print(
        f"Ratings:      "
        f"{len(df):,}"
    )

    print(
        f"Average:      "
        f"{df['rating'].mean():.2f}"
    )

    print(
        f"Minimum:      "
        f"{df['rating'].min()}"
    )

    print(
        f"Maximum:      "
        f"{df['rating'].max()}"
    )


def load_model():

    if not MODEL_FILE.exists():

        print(
            "\nNo trained model found."
        )

        print(
            "Please choose option 5 "
            "to train the model."
        )

        return None

    return RecommendationModel.load()


def main():

    df = load_ratings()

    model = None

    while True:

        print_header(
            "MOVIE RECOMMENDATION SYSTEM"
        )

        print("1. List users")
        print("2. Show user ratings")
        print("3. Recommend movies")
        print("4. Find similar users")
        print("5. Train model")
        print("6. Dataset statistics")
        print("7. Exit")

        print()

        choice = input(
            "Select an option: "
        ).strip()

        if choice == "1":

            list_users(df)

        elif choice == "2":

            show_user_ratings(df)

        elif choice == "3":

            if model is None:
                model = load_model()

            if model:
                recommend_movies(
                    model,
                    df,
                )

        elif choice == "4":

            if model is None:
                model = load_model()

            if model:
                similar_users(model)

        elif choice == "5":

            train_model()

            model = RecommendationModel.load()

        elif choice == "6":

            dataset_statistics(df)

        elif choice == "7":

            print(
                "\nThank you for using "
                "the Movie Recommendation System."
            )

            break

        else:

            print(
                "\nInvalid option."
            )

        input(
            "\nPress ENTER to continue..."
        )


if __name__ == "__main__":
    main()