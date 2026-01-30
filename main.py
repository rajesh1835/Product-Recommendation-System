# main.py
# =========================================================
# E-COMMERCE RECOMMENDATION SYSTEM (PIPELINE ENTRY POINT)
# =========================================================

from src.components.data_loader import DataLoader
from src.components.data_combiner import DataCombiner
from src.components.data_cleaner import DataCleaner
from src.components.sampler import DataSampler
from src.components.data_summary import DataSummary
from src.components.data_quality_report import DataQualityReport
from src.components.eda import EDA
from warnings import filterwarnings
filterwarnings('ignore')

from src.models.baseline_model import run_baseline
from src.models.knn_model import run_knn
from src.models.svd_model import run_svd
from src.models.knn_with_kmeans import run_knn_with_means
from src.models.knn_basic_gridsearch import run_knn_gridsearch
from src.testing.ab_test import run_ab_testing
from src.inference.predictor import train_and_save_model, demo_predictions

import os
import pandas as pd


def main():
    print("\n===================================================")
    print(" PROJECT STARTED : E-COMMERCE RECOMMENDER SYSTEM ")
    print("===================================================\n")

    # -------------------------------------------------
    # STEP 1: Load Raw Data
    # -------------------------------------------------
    print("[STEP 1] Loading raw datasets...\n")

    loader = DataLoader()
    amazon_df = loader.load_amazon_data()
    reviews = loader.load_reviews()

    summary = DataSummary()

    summary.add_stage(
        "Raw Amazon Products Data",
        amazon_df,
        "Original Amazon product dataset"
    )

    summary.add_text_stage(
        "Raw Reviews Data",
        len(reviews),
        "Unstructured reviews text file"
    )

    # -------------------------------------------------
    # STEP 2: Combine Datasets
    # -------------------------------------------------
    print("[STEP 2] Combining Amazon data with reviews...\n")

    combiner = DataCombiner(amazon_df, reviews)
    combined_df = (
        combiner
        .generate_user_product_ids()
        .assign_reviews()
        .get_combined_data()
    )

    summary.add_stage(
        "Combined Dataset",
        combined_df,
        "Amazon data + synthetic UserId/ProductId"
    )

    print("✔ Dataset combination completed\n")

    # -------------------------------------------------
    # STEP 3: Data Cleaning
    # -------------------------------------------------
    print("[STEP 3] Cleaning dataset...\n")

    cleaner = DataCleaner(combined_df)
    cleaned_df = (
        cleaner
        .drop_unused_columns()
        .convert_numeric_columns()
        .handle_missing_values()
        .fix_price_logic()
        .get_clean_data()
    )

    print("✔ Data cleaning completed\n")

    print("[DATA QUALITY REPORT] Before vs After Cleaning\n")
    DataQualityReport.compare(
        before_df=combined_df,
        after_df=cleaned_df
    )

    summary.add_stage(
        "Cleaned Dataset",
        cleaned_df,
        "After datatype correction and missing-value handling"
    )

    # -------------------------------------------------
    # EXPORT: Save Cleaned Data
    # -------------------------------------------------
    export_path = "data/processed/amazon_products_with_reviews.csv"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    print(f"[EXPORT] Saving cleaned dataset to {export_path}...")
    cleaned_df.to_csv(export_path, index=False)
    print("✔ Export completed\n")

    # -------------------------------------------------
    # STEP 4: Sampling
    # -------------------------------------------------
    print("[STEP 4] Sampling dataset to 25,000 rows...\n")

    sampler = DataSampler(cleaned_df)
    sampled_df = sampler.sample(n=25000)

    summary.add_stage(
        "Sampled Dataset",
        sampled_df,
        "Random 25k sample for scalable analysis"
    )

    print("✔ Sampling completed\n")

    # -------------------------------------------------
    # STEP 5: Dataset Summary
    # -------------------------------------------------
    print("[STEP 5] Dataset summary across pipeline stages:\n")
    summary.print_summary()
    summary.save_summary()

    # -------------------------------------------------
    # STEP 6: Exploratory Data Analysis (EDA)
    # -------------------------------------------------
    print("[STEP 6] Starting Exploratory Data Analysis (EDA)...\n")

    eda = EDA(sampled_df)
    
    # -----------------------------
    # 6.1 Univariate Analysis
    # -----------------------------
    print("   [6.1] Univariate Analysis (Individual Distributions)...")
    eda.category_distribution()
    eda.ratings_distribution()
    eda.popularity_distribution()

    # -----------------------------
    # 6.2 Bivariate Analysis
    # -----------------------------
    print("\n   [6.2] Bivariate Analysis (Relationships)...")
    eda.price_analysis() 
    eda.top_product_per_category()
    eda.ratings_vs_popularity()

    # -----------------------------
    # 6.3 Multivariate Analysis
    # -----------------------------
    print("\n   [6.3] Multivariate Analysis (Correlations)...")
    eda.correlation_matrix()
    
    # Storytelling
    eda.generate_insights()


    print("\n✔ EDA completed - All visualizations saved to artifacts/eda/\n")

    # -------------------------------------------------
    # STEP 7: Model Training & Evaluation (4 Models)
    # -------------------------------------------------
    print("[STEP 7] Training and evaluating recommendation models...\n")

    results = []

    # Model 1: Baseline
    rmse, mae = run_baseline()
    print(f"   ✓ BaselineOnly   - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    results.append(["BaselineOnly", rmse, mae])

    # Model 2: KNN Basic
    rmse, mae = run_knn()
    print(f"   ✓ KNNBasic       - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    results.append(["KNNBasic", rmse, mae])

    # Model 3: SVD
    rmse, mae = run_svd()
    print(f"   ✓ SVD            - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    results.append(["SVD", rmse, mae])

    # Model 4: KNN With Means
    rmse, mae = run_knn_with_means()
    print(f"   ✓ KNNWithMeans   - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    results.append(["KNNWithMeans", rmse, mae])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "RMSE", "MAE"]
    )

    # Sort by RMSE and find best model
    results_df = results_df.sort_values("RMSE").reset_index(drop=True)
    best_model = results_df.iloc[0]["Model"]
    best_rmse = results_df.iloc[0]["RMSE"]
    best_mae = results_df.iloc[0]["MAE"]

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv(
        "data/processed/model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 50)
    print(" MODEL COMPARISON RESULTS (Sorted by RMSE) ")
    print("=" * 50)
    print(results_df.to_string(index=False))
    print("=" * 50)
    
    print(f"\n🏆 BEST MODEL SELECTED: {best_model}")
    print(f"   RMSE: {best_rmse:.4f}")
    print(f"   MAE : {best_mae:.4f}")
    print("\n✅ Model comparison saved to data/processed/model_comparison.csv\n")

    # -------------------------------------------------
    # STEP 8: A/B Testing (Compare All 4 Models)
    # -------------------------------------------------
    print("[STEP 8] Running A/B Testing to compare all models...\n")

    ab_winner, ab_summary = run_ab_testing()
    print(f"\n🏆 A/B Test Winner: {ab_winner}\n")

    # -------------------------------------------------
    # STEP 9: KNN Grid Search (Hyperparameter Tuning)
    # -------------------------------------------------
    print("[STEP 9] Hyperparameter Tuning with GridSearchCV...\n")

    gs_rmse, gs_mae, best_params = run_knn_gridsearch()
    print(f"   ✓ Best RMSE: {gs_rmse:.4f}")
    print(f"   ✓ Best MAE : {gs_mae:.4f}")
    print(f"   ✓ Best Params: {best_params}\n")

    # -------------------------------------------------
    # STEP 10: Save Final Model Configuration
    # -------------------------------------------------
    print("[STEP 10] Saving final model configuration...\n")

    import json
    model_config = {
        "best_model": ab_winner,  # Use A/B test winner
        "best_params": best_params,
        "gs_rmse": float(gs_rmse),
        "gs_mae": float(gs_mae)
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/final_model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)

    print("✅ Final model configuration saved to artifacts/final_model_config.json\n")

    # -------------------------------------------------
    # STEP 11: Train, Save & Demo Predictions
    # -------------------------------------------------
    print("[STEP 11] Training final model and generating predictions...\n")

    # Train and save the winning model
    train_and_save_model(ab_winner)

    # Show sample recommendations
    demo_predictions(n_users=3, n_recs=5)

    print("\n===================================================")
    print(" PROJECT COMPLETED SUCCESSFULLY ")
    print("===================================================")


if __name__ == "__main__":
    main()
