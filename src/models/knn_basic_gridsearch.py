import os
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import GridSearchCV


# -----------------------------
# PATH CONFIG
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["user_id", "product_id", "rating"]]

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df, reader)

    return data


# -----------------------------
# GRID SEARCH
# -----------------------------
def run_knn_gridsearch():
    print("🔹 Running GridSearchCV for KNNBasic...")

    data = load_data()

    param_grid = {
        "k": [20, 40],
        "min_k": [1, 3],
        "sim_options": {
            "name": ["cosine", "pearson"],
            "user_based": [False]
        }
    }

    gs = GridSearchCV(
        KNNBasic,
        param_grid,
        measures=["rmse", "mae"],
        cv=3,
        n_jobs=1,       # Sequential to fix MemoryError
        joblib_verbose=1 # Show progress
    )

    gs.fit(data)

    best_rmse = gs.best_score['rmse']
    best_mae = gs.best_score['mae']
    best_params = gs.best_params['rmse']

    return best_rmse, best_mae, best_params


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    rmse, mae, params = run_knn_gridsearch()
    print(f"\n✅ GridSearch Results (KNNBasic)")
    print(f"   Best RMSE: {rmse:.4f}")
    print(f"   Best MAE : {mae:.4f}")
    print(f"   Best Params: {params}")
