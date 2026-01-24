import os
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, accuracy


# -----------------------------
# PATH HANDLING
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

RATING_SCALE = (1, 5)


# -----------------------------
# LOAD DATA
# -----------------------------
def load_train_test():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    reader = Reader(rating_scale=RATING_SCALE)

    train_data = Dataset.load_from_df(
        train_df[["user_id", "product_id", "rating"]],
        reader
    )

    trainset = train_data.build_full_trainset()

    testset = list(
        zip(
            test_df["user_id"],
            test_df["product_id"],
            test_df["rating"]
        )
    )

    return trainset, testset


# -----------------------------
# TRAIN & EVALUATE KNN MODEL
# -----------------------------
def run_knn():
    print("🔹 Training KNNBasic Model...")

    trainset, testset = load_train_test()

    sim_options = {
        "name": "cosine",
        "user_based": True   # User-based CF
    }

    knn_model = KNNBasic(sim_options=sim_options)
    knn_model.fit(trainset)

    predictions = knn_model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    return rmse, mae


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_knn()
