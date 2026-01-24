"""
A/B Testing Framework for Recommendation Models
Compares: BaselineOnly, KNNBasic, SVD, KNNWithMeans
"""

import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, BaselineOnly, KNNBasic, KNNWithMeans, SVD, accuracy
from surprise.model_selection import train_test_split
import json
from datetime import datetime


# -----------------------------
# PATH CONFIG
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "ab_test_results.json")


def load_data():
    """Load and prepare data for testing"""
    df = pd.read_csv(DATA_PATH)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["user_id", "product_id", "rating"]], 
        reader
    )
    return data


def get_models():
    """Return dictionary of all models to test"""
    return {
        "BaselineOnly": BaselineOnly(),
        "KNNBasic": KNNBasic(
            sim_options={"name": "cosine", "user_based": True}
        ),
        "KNNWithMeans": KNNWithMeans(
            sim_options={"name": "cosine", "user_based": True}
        ),
        "SVD": SVD(n_factors=100, n_epochs=20, random_state=42)
    }


def run_ab_test(n_splits=5):
    """
    Run A/B testing across all models
    Uses multiple train/test splits for statistical significance
    """
    print("\n" + "=" * 60)
    print(" 🔬 A/B TESTING - COMPARING 4 MODELS ")
    print("=" * 60)
    
    data = load_data()
    models = get_models()
    
    # Store results for each model
    results = {name: {"rmse": [], "mae": []} for name in models.keys()}
    
    print(f"\n📊 Running {n_splits} test splits per model...")
    print("-" * 60)
    
    for split_num in range(n_splits):
        print(f"\n🔄 Split {split_num + 1}/{n_splits}")
        
        # Random train/test split
        trainset, testset = train_test_split(data, test_size=0.2, random_state=split_num)
        
        for model_name, model in models.items():
            # Train model
            model.fit(trainset)
            
            # Test model
            predictions = model.test(testset)
            
            # Calculate metrics
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            
            results[model_name]["rmse"].append(rmse)
            results[model_name]["mae"].append(mae)
            
            print(f"   {model_name:15} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return results


def analyze_results(results):
    """Analyze and compare model performance"""
    print("\n" + "=" * 60)
    print(" 📈 A/B TEST RESULTS ANALYSIS ")
    print("=" * 60)
    
    summary = []
    
    for model_name, metrics in results.items():
        rmse_mean = np.mean(metrics["rmse"])
        rmse_std = np.std(metrics["rmse"])
        mae_mean = np.mean(metrics["mae"])
        mae_std = np.std(metrics["mae"])
        
        summary.append({
            "Model": model_name,
            "RMSE (mean)": rmse_mean,
            "RMSE (std)": rmse_std,
            "MAE (mean)": mae_mean,
            "MAE (std)": mae_std
        })
    
    # Create DataFrame and sort by RMSE
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("RMSE (mean)").reset_index(drop=True)
    
    print("\n📊 Performance Summary (sorted by RMSE):")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    
    # Find winner
    winner = summary_df.iloc[0]["Model"]
    best_rmse = summary_df.iloc[0]["RMSE (mean)"]
    
    # Statistical significance check
    print("\n📏 Statistical Comparison:")
    print("-" * 60)
    
    for i, row in summary_df.iterrows():
        if row["Model"] != winner:
            diff = row["RMSE (mean)"] - best_rmse
            pct_diff = (diff / best_rmse) * 100
            print(f"   {winner} vs {row['Model']}: {pct_diff:.2f}% better")
    
    print("\n" + "=" * 60)
    print(f" 🏆 WINNER: {winner} ")
    print(f"    Average RMSE: {best_rmse:.4f}")
    print("=" * 60)
    
    return summary_df, winner


def save_results(results, summary_df, winner):
    """Save A/B test results to JSON"""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "winner": winner,
        "summary": summary_df.to_dict(orient="records"),
        "detailed_results": {
            model: {
                "rmse_scores": metrics["rmse"],
                "mae_scores": metrics["mae"]
            }
            for model, metrics in results.items()
        }
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"\n✅ Results saved to: {RESULTS_PATH}")


def run_ab_testing():
    """Main function to run complete A/B test"""
    # Run tests
    results = run_ab_test(n_splits=5)
    
    # Analyze
    summary_df, winner = analyze_results(results)
    
    # Save
    save_results(results, summary_df, winner)
    
    return winner, summary_df


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    winner, summary = run_ab_testing()
