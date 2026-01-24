import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, accuracy

# -----------------------------
# PATH CONFIG
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
TEST_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "knn_basic_model.pkl")

RATING_SCALE = (1, 5)

# -----------------------------
# LOAD TRAIN & TEST DATA
# -----------------------------
def load_train_test():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

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
# TRAIN FINAL MODEL
# -----------------------------
def train_final_model():
    print("🔹 Training FINAL KNNBasic Model")

    trainset, testset = load_train_test()

    sim_options = {
        "name": "cosine",
        "user_based": False   # ITEM-BASED (best from GridSearch)
    }

    model = KNNBasic(
        k=20,
        min_k=1,
        sim_options=sim_options
    )

    model.fit(trainset)

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    mae  = accuracy.mae(predictions)

    print("\n✅ Final Model Performance")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

    return model

# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Model saved at:\n{MODEL_PATH}")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    final_model = train_final_model()
    save_model(final_model)
