# ==========================================
# 🚀 FIXED GRID SEARCH SCRIPT FOR COLAB
# ==========================================

import os
import sys

def install_dependencies():
    print("📦 Installing dependencies...")
    # Surprise has issues with newer numpy versions (>=1.24/1.25) sometimes.
    # We force a reinstall to ensure compatibility.
    os.system('pip install numpy<2.0.0 scikit-surprise pandas')

try:
    import surprise
except ImportError:
    install_dependencies()
    # Force re-import after install
    import site
    site.main() 
    try:
        import surprise
    except ImportError:
        print("⚠ Restarting runtime may be required if import fails.")

# IMPORTS
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import GridSearchCV
import random

def run_colab_gridsearch():
    print("⏳ Loading Data...")
    
    # Check if files exist
    if not os.path.exists('Amazon-Products.csv'):
        print("❌ Error: Please upload 'Amazon-Products.csv' to Colab!")
        return

    # Load Products
    amazon_df = pd.read_csv("Amazon-Products.csv")
    # Clean currency symbols
    if 'actual_price' in amazon_df.columns:
        amazon_df['actual_price'] = amazon_df['actual_price'].astype(str).str.replace('[₹,]', '', regex=True)
        amazon_df['actual_price'] = pd.to_numeric(amazon_df['actual_price'], errors='coerce')
    
    if 'discount_price' in amazon_df.columns:
        amazon_df['discount_price'] = amazon_df['discount_price'].astype(str).str.replace('[₹,]', '', regex=True)
        amazon_df['discount_price'] = pd.to_numeric(amazon_df['discount_price'], errors='coerce')

    print("⚙️ Processing Data...")
    
    # ----------------------------------------------------
    # SYNTHETIC DATA GENERATION (Since raw data lacks UserIDs)
    # ----------------------------------------------------
    # Ensure product_id exists
    if 'product_id' not in amazon_df.columns:
        # Create synthetic product IDs if not present
        amazon_df['product_id'] = [f"B{str(i).zfill(5)}" for i in range(len(amazon_df))]

    # Create Interactions
    data_list = []
    random.seed(42)
    
    # Create sample user IDs
    user_ids = [f"A{str(i).zfill(5)}" for i in range(1000)]
    
    # Using a subset of products for speed
    product_ids = amazon_df['product_id'].head(5000).values
    
    sample_size = 20000 
    print(f"   Creating synthetic interactions: {sample_size} rows...")
    
    for _ in range(sample_size):
        u_id = random.choice(user_ids)
        p_id = random.choice(product_ids)
        rating = random.randint(1, 5) # Synthetic rating
        data_list.append({'user_id': u_id, 'product_id': p_id, 'rating': rating})
        
    interaction_df = pd.DataFrame(data_list)
    
    # ----------------------------------------------------
    # SUPRRise SETUP
    # ----------------------------------------------------
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interaction_df[['user_id', 'product_id', 'rating']], reader)
    
    # ----------------------------------------------------
    # GRID SEARCH
    # ----------------------------------------------------
    print("\n🔍 Starting GridSearchCV...")
    param_grid = {
        "k": [20, 40], # Reduced search space for stability
        "min_k": [1, 3],
        "sim_options": {
            "name": ["cosine", "pearson"],
            "user_based": [False]
        }
    }

    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
    gs.fit(data)

    print("\n" + "="*40)
    print("🎉 GRID SEARCH RESULTS")
    print("="*40)
    print(f"🏆 Best RMSE: {gs.best_score['rmse']:.4f}")
    print(f"👉 Best Params (RMSE): {gs.best_params['rmse']}")
    print("-" * 40)
    print(f"🏆 Best MAE: {gs.best_score['mae']:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_colab_gridsearch()
