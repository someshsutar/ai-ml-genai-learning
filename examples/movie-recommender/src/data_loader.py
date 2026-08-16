import pandas as pd

from src.config import RATINGS_FILE


def load_ratings():
    """
    Load ratings from CSV.
    """

    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RATINGS_FILE}\n"
            "Run: python -m src.generate_data"
        )

    df = pd.read_csv(RATINGS_FILE)

    required_columns = {
        "user_id",
        "movie_id",
        "rating",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df
