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