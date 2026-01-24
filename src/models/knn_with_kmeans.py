import os
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, accuracy

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(
        train_df[["user_id", "product_id", "rating"]],
        reader
    )
    trainset = train_data.build_full_trainset()

    testset = list(zip(
        test_df["user_id"],
        test_df["product_id"],
        test_df["rating"]
    ))

    return trainset, testset


def run_knn_with_means():
    print("🔹 Training KNNWithMeans Model...")

    trainset, testset = load_data()

    sim_options = {
        "name": "cosine",
        "user_based": True
    }

    model = KNNWithMeans(sim_options=sim_options)
    model.fit(trainset)

    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)

    return rmse, mae

if __name__ == "__main__":
    run_knn_with_means()
