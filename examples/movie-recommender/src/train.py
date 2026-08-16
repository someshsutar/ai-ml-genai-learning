from src.data_loader import load_ratings
from src.recommender import RecommendationModel


def main():

    print("Loading dataset...")

    df = load_ratings()

    print(f"Ratings loaded: {len(df):,}")

    model = RecommendationModel()

    model.fit(df)

    model.save()


if __name__ == "__main__":
    main()