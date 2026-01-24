import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


# -----------------------------
# PATH CONFIGURATION
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_rec.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

RATING_SCALE = (1, 5)
TEST_SIZE = 0.2
RANDOM_STATE = 42


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    # Create processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Rename columns to Surprise format
    df = df.rename(columns={
        "UserId": "user_id",
        "ProductId": "product_id",
        "ratings": "rating"
    })

    # Keep required columns only
    df = df[["user_id", "product_id", "rating"]]

    # Clean data
    df = df.dropna()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # Prepare Surprise dataset
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(df, reader)

    # Train-test split using Surprise
    trainset, testset = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Convert trainset to DataFrame
    train_df = pd.DataFrame(
        [
            (
                trainset.to_raw_uid(u),
                trainset.to_raw_iid(i),
                r
            )
            for (u, i, r) in trainset.all_ratings()
        ],
        columns=["user_id", "product_id", "rating"]
    )

    # Convert testset to DataFrame
    test_df = pd.DataFrame(
        testset,
        columns=["user_id", "product_id", "rating"]
    )

    # Save files
    train_path = os.path.join(PROCESSED_DIR, TRAIN_FILE)
    test_path = os.path.join(PROCESSED_DIR, TEST_FILE)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("✅ Train–Test Split Completed Using Surprise")
    print(f"📁 Train file saved at: {train_path}")
    print(f"📁 Test file saved at : {test_path}")
    print(f"🔢 Train rows: {len(train_df)}")
    print(f"🔢 Test rows : {len(test_df)}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
