"""
Hybrid Recommendation System
Combines Collaborative Filtering (CF) and Content-Based Filtering (CBF)

Formula: Hybrid Score = α × CF_Score + (1-α) × Content_Score
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path configuration
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PRODUCTS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_rec.csv")
PRODUCTS_PATH_ALT = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_products_with_reviews.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "trained_model.pkl")
TFIDF_CACHE_PATH = os.path.join(PROJECT_ROOT, "artifacts", "tfidf_matrix.pkl")

# Import config
from src.inference.hybrid_config import HYBRID_CONFIG


class HybridRecommender:
    """
    Hybrid Recommendation Engine combining:
    - Content-Based Filtering (TF-IDF on product names/descriptions)
    - Collaborative Filtering (KNN from trained model)
    """
    
    def __init__(self, alpha=None):
        """
        Initialize the hybrid recommender
        
        Args:
            alpha: Weight for collaborative filtering (0-1)
                   If None, uses value from config
        """
        self.alpha = alpha if alpha is not None else HYBRID_CONFIG["alpha"]
        self.products = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self.cf_model = None
        
        # Load data and build matrices
        self._load_products()
        self._build_tfidf_matrix()
        self._load_cf_model()
    
    def _load_products(self):
        """Load products from CSV"""
        try:
            if os.path.exists(PRODUCTS_PATH):
                self.products = pd.read_csv(PRODUCTS_PATH)
            elif os.path.exists(PRODUCTS_PATH_ALT):
                self.products = pd.read_csv(PRODUCTS_PATH_ALT)
            else:
                raise FileNotFoundError("No product data found")
            
            # Clean data
            self.products = self.products.dropna(subset=['name'])
            
            # Identify product ID column
            if 'ProductId' in self.products.columns:
                id_col = 'ProductId'
            elif 'product_id' in self.products.columns:
                id_col = 'product_id'
            else:
                id_col = self.products.columns[0]
            
            # Build index mappings
            for idx, row in self.products.iterrows():
                pid = row[id_col]
                self.product_id_to_idx[pid] = idx
                self.idx_to_product_id[idx] = pid
            
            print(f"✅ Loaded {len(self.products)} products for hybrid recommendations")
            
        except Exception as e:
            print(f"❌ Error loading products: {e}")
            self.products = pd.DataFrame()
    
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix for content-based filtering"""
        if self.products is None or len(self.products) == 0:
            print("⚠️ No products to build TF-IDF matrix")
            return
        
        # Check cache
        if HYBRID_CONFIG["cache_enabled"] and os.path.exists(TFIDF_CACHE_PATH):
            try:
                with open(TFIDF_CACHE_PATH, 'rb') as f:
                    cache = pickle.load(f)
                    self.tfidf_matrix = cache['matrix']
                    self.tfidf_vectorizer = cache['vectorizer']
                    print("✅ Loaded TF-IDF matrix from cache")
                    return
            except:
                pass
        
        try:
            # Combine text features
            text_features = []
            for _, row in self.products.iterrows():
                parts = []
                for feature in HYBRID_CONFIG["content_features"]:
                    if feature in row and pd.notna(row[feature]):
                        parts.append(str(row[feature]))
                text_features.append(" ".join(parts))
            
            # Build TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=HYBRID_CONFIG["tfidf_max_features"],
                ngram_range=HYBRID_CONFIG["tfidf_ngram_range"],
                stop_words=HYBRID_CONFIG["tfidf_stop_words"]
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
            
            # Cache the matrix
            if HYBRID_CONFIG["cache_enabled"]:
                os.makedirs(os.path.dirname(TFIDF_CACHE_PATH), exist_ok=True)
                with open(TFIDF_CACHE_PATH, 'wb') as f:
                    pickle.dump({
                        'matrix': self.tfidf_matrix,
                        'vectorizer': self.tfidf_vectorizer
                    }, f)
            
            print(f"✅ Built TF-IDF matrix: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            print(f"❌ Error building TF-IDF matrix: {e}")
    
    def _load_cf_model(self):
        """Load collaborative filtering model"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.cf_model = pickle.load(f)
                print("✅ Loaded collaborative filtering model")
            else:
                print("⚠️ No CF model found - using content-based only")
        except Exception as e:
            print(f"⚠️ Error loading CF model: {e}")
    
    def get_content_based_recommendations(self, product_id, n=6):
        """
        Get content-based recommendations using TF-IDF similarity
        
        Args:
            product_id: Product ID to find similar products for
            n: Number of recommendations
            
        Returns:
            List of dicts with product info and similarity scores
        """
        if self.tfidf_matrix is None or product_id not in self.product_id_to_idx:
            return []
        
        try:
            idx = self.product_id_to_idx[product_id]
            
            # Compute cosine similarity
            product_vector = self.tfidf_matrix[idx]
            similarities = cosine_similarity(product_vector, self.tfidf_matrix).flatten()
            
            # Get top N similar products (excluding itself)
            similar_indices = similarities.argsort()[::-1][1:n+1]
            
            results = []
            for sim_idx in similar_indices:
                sim_score = similarities[sim_idx]
                if sim_score < HYBRID_CONFIG["min_similarity_threshold"]:
                    continue
                    
                pid = self.idx_to_product_id.get(sim_idx)
                if pid is None:
                    continue
                
                row = self.products.iloc[sim_idx]
                results.append({
                    'ProductId': pid,
                    'name': row.get('name', 'Unknown'),
                    'main_category': row.get('main_category', ''),
                    'sub_category': row.get('sub_category', ''),
                    'ratings': row.get('ratings', 0),
                    'discount_price': row.get('discount_price', 0),
                    'actual_price': row.get('actual_price', 0),
                    'image': row.get('image', ''),
                    'similarity_score': float(sim_score),
                    'rec_type': 'content'
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error in content-based recommendations: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, product_id=None, n=6):
        """
        Get collaborative filtering recommendations
        
        Args:
            user_id: User ID for personalized recommendations
            product_id: Optional product ID to exclude
            n: Number of recommendations
            
        Returns:
            List of dicts with product info and predicted ratings
        """
        if self.cf_model is None:
            return []
        
        try:
            # Get all products
            all_products = list(self.product_id_to_idx.keys())
            
            # Predict ratings for all products
            predictions = []
            for pid in all_products:
                if pid == product_id:
                    continue
                try:
                    pred = self.cf_model.predict(user_id, pid)
                    predictions.append((pid, pred.est))
                except:
                    pass
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_n = predictions[:n]
            
            results = []
            for pid, pred_rating in top_n:
                idx = self.product_id_to_idx.get(pid)
                if idx is None:
                    continue
                    
                row = self.products.iloc[idx]
                results.append({
                    'ProductId': pid,
                    'name': row.get('name', 'Unknown'),
                    'main_category': row.get('main_category', ''),
                    'sub_category': row.get('sub_category', ''),
                    'ratings': row.get('ratings', 0),
                    'discount_price': row.get('discount_price', 0),
                    'actual_price': row.get('actual_price', 0),
                    'image': row.get('image', ''),
                    'predicted_rating': float(pred_rating),
                    'similarity_score': float(pred_rating / 5.0),  # Normalize to 0-1
                    'rec_type': 'collaborative'
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ CF recommendations failed: {e}")
            return []
    
    def get_hybrid_recommendations(self, product_id, user_id=None, n=6):
        """
        Get hybrid recommendations combining CF and content-based
        
        Args:
            product_id: Product ID to find similar products for
            user_id: Optional user ID for personalized CF
            n: Number of recommendations
            
        Returns:
            List of dicts with product info and hybrid scores
        """
        # Get content-based recommendations
        content_recs = self.get_content_based_recommendations(product_id, n=n*2)
        
        # Get collaborative recommendations if user_id provided
        cf_recs = []
        if user_id and self.cf_model:
            cf_recs = self.get_collaborative_recommendations(user_id, product_id, n=n*2)
        
        # If no CF available, return content-based
        if not cf_recs:
            for rec in content_recs[:n]:
                rec['rec_type'] = 'hybrid'  # Mark as hybrid even if only content
            return content_recs[:n]
        
        # Combine recommendations
        combined = {}
        
        # Add content-based scores
        for rec in content_recs:
            pid = rec['ProductId']
            combined[pid] = {
                **rec,
                'content_score': rec['similarity_score'],
                'cf_score': 0.0
            }
        
        # Add CF scores
        for rec in cf_recs:
            pid = rec['ProductId']
            if pid in combined:
                combined[pid]['cf_score'] = rec['similarity_score']
            else:
                combined[pid] = {
                    **rec,
                    'content_score': 0.0,
                    'cf_score': rec['similarity_score']
                }
        
        # Calculate hybrid scores
        for pid in combined:
            content_score = combined[pid]['content_score']
            cf_score = combined[pid]['cf_score']
            hybrid_score = self.alpha * cf_score + (1 - self.alpha) * content_score
            combined[pid]['similarity_score'] = hybrid_score
            combined[pid]['rec_type'] = 'hybrid'
        
        # Sort by hybrid score
        sorted_recs = sorted(
            combined.values(),
            key=lambda x: x['similarity_score'],
            reverse=True
        )
        
        return sorted_recs[:n]
    
    def get_recommendations(self, product_id, user_id=None, n=6, method='hybrid'):
        """
        Get recommendations using specified method
        
        Args:
            product_id: Product ID
            user_id: Optional user ID
            n: Number of recommendations
            method: 'hybrid', 'content', or 'collaborative'
        """
        if method == 'content':
            return self.get_content_based_recommendations(product_id, n)
        elif method == 'collaborative' and user_id:
            return self.get_collaborative_recommendations(user_id, product_id, n)
        else:
            return self.get_hybrid_recommendations(product_id, user_id, n)


# -----------------------------
# DEMO / TEST
# -----------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🤖 HYBRID RECOMMENDATION SYSTEM ")
    print("=" * 60)
    
    hr = HybridRecommender(alpha=0.5)
    
    # Test with a sample product
    if hr.products is not None and len(hr.products) > 0:
        sample_id = list(hr.product_id_to_idx.keys())[0]
        print(f"\n📦 Testing recommendations for product: {sample_id}")
        
        print("\n--- Content-Based ---")
        content_recs = hr.get_content_based_recommendations(sample_id, n=3)
        for rec in content_recs:
            name = str(rec['name'])[:40]
            score = rec['similarity_score']
            print(f"  • {name}... (score: {score:.3f})")
        
        print("\n--- Hybrid ---")
        hybrid_recs = hr.get_hybrid_recommendations(sample_id, n=3)
        for rec in hybrid_recs:
            name = str(rec['name'])[:40]
            score = rec['similarity_score']
            rec_type = rec['rec_type']
            print(f"  • {name}... (score: {score:.3f}, type: {rec_type})")
    
    print("\n" + "=" * 60)
