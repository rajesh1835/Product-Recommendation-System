"""
Model Predictor - Train, Save, Load, and Make Predictions
"""

import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, SVD, BaselineOnly
from src.models.content_based import ContentBasedRecommender


# -----------------------------
# PATH CONFIG
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
PRODUCTS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_rec.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "trained_model.pkl")


# -----------------------------
# MODEL FACTORY
# -----------------------------
def get_model(model_name):
    """Return model instance by name"""
    models = {
        "BaselineOnly": BaselineOnly(),
        "KNNBasic": KNNBasic(sim_options={"name": "cosine", "user_based": True}),
        "KNNWithMeans": KNNWithMeans(sim_options={"name": "cosine", "user_based": True}),
        "SVD": SVD(n_factors=100, n_epochs=20, random_state=42)
    }
    return models.get(model_name, KNNBasic())


# -----------------------------
# TRAIN & SAVE MODEL
# -----------------------------
def train_and_save_model(model_name="KNNBasic"):
    """Train and save the best model"""
    print(f"🔹 Training {model_name} model for predictions...")
    
    # Load data
    df = pd.read_csv(TRAIN_PATH)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["user_id", "product_id", "rating"]], 
        reader
    )
    trainset = data.build_full_trainset()
    
    # Train model
    model = get_model(model_name)
    model.fit(trainset)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to: {MODEL_PATH}")
    return model


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    """Load saved model from pickle"""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ No saved model found. Training new model...")
        return train_and_save_model()
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded from: {MODEL_PATH}")
    return model


# -----------------------------
# GET RECOMMENDATIONS
# -----------------------------
def get_recommendations(user_id, n=5, model=None):
    """
    Get top N product recommendations for a user
    
    Args:
        user_id: User ID to get recommendations for
        n: Number of recommendations (default 5)
        model: Pre-loaded model (optional)
    
    Returns:
        DataFrame with top N recommended products
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load product data
    products_df = pd.read_csv(PRODUCTS_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    
    # Get products user hasn't rated
    user_products = train_df[train_df["user_id"] == user_id]["product_id"].tolist()
    all_products = train_df["product_id"].unique()
    products_to_predict = [p for p in all_products if p not in user_products]
    
    # Predict ratings for unseen products
    predictions = []
    for product_id in products_to_predict:
        pred = model.predict(user_id, product_id)
        predictions.append({
            "product_id": product_id,
            "predicted_rating": pred.est
        })
    
    # Sort by predicted rating
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values("predicted_rating", ascending=False)
    top_n = predictions_df.head(n)
    
    # Merge with product details
    result = top_n.merge(
        products_df[["ProductId", "name", "main_category", "ratings", "actual_price"]],
        left_on="product_id",
        right_on="ProductId",
        how="left"
    )
    
    return result[["product_id", "name", "main_category", "predicted_rating", "actual_price"]]


# -----------------------------
# GET SIMILAR PRODUCTS (CONTENT)
# -----------------------------
def get_similar_products(product_id, n=5):
    """
    Get products similar to a given product_id using Content-Based Filtering
    """
    products_df = pd.read_csv(PRODUCTS_PATH)
    
    # Initialize Content-Based Model
    cb_model = ContentBasedRecommender(products_df)
    
    return cb_model.get_recommendations(product_id, n=n)


# -----------------------------
# DEMO PREDICTIONS
# -----------------------------
def demo_predictions(n_users=3, n_recs=5):
    """Show sample predictions for random users"""
    print("\n" + "=" * 60)
    print(" 🎯 SAMPLE RECOMMENDATIONS ")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Get sample users
    train_df = pd.read_csv(TRAIN_PATH)
    sample_users = train_df["user_id"].sample(n_users, random_state=42).tolist()
    
    for user_id in sample_users:
        print(f"\n👤 User: {user_id}")
        print("-" * 50)
        
        recs = get_recommendations(user_id, n=n_recs, model=model)
        
        for i, row in recs.iterrows():
            name = str(row["name"])[:40] if pd.notna(row["name"]) else "Unknown"
            rating = row["predicted_rating"]
            print(f"   {i+1}. {name}... (predicted: {rating:.2f}⭐)")
    
    print("\n" + "=" * 60)
    
    # ------------------------------------------------
    # PART 2: Content-Based Recommendations
    # ------------------------------------------------
    print(" 🔄 SIMILAR PRODUCT RECOMMENDATIONS (Content-Based) ")
    print("=" * 60)
    
    # Pick random products to find similarities for
    products_df = pd.read_csv(PRODUCTS_PATH)
    sample_products = products_df["ProductId"].sample(n_users, random_state=42).tolist()
    
    for product_id in sample_products:
        product_name = products_df[products_df["ProductId"] == product_id]["name"].values[0]
        display_name = str(product_name)[:50]
        
        print(f"\n📦 Product: {display_name}...")
        print(f"   (ID: {product_id})")
        print("-" * 50)
        
        try:
            similar_recs = get_similar_products(product_id, n=n_recs)
            
            if similar_recs.empty:
                print("   No similar products found.")
                continue

            for i, row in similar_recs.iterrows():
                name = str(row["name"])[:40]
                score = row["similarity_score"]
                print(f"   {i+1}. {name}... (similarity: {score:.2f})")
                
        except Exception as e:
            print(f"   Error generating recommendations: {e}")

    print("\n" + "=" * 60)

    return sample_users


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    # Train and save model
    train_and_save_model("KNNBasic")
    
    # Show demo predictions
    demo_predictions()
