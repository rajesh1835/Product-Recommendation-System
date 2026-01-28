"""
Hybrid Recommendation Configuration
"""

HYBRID_CONFIG = {
    # Weight for collaborative filtering (0-1)
    # Higher value = more weight to collaborative filtering
    # Lower value = more weight to content-based filtering
    "alpha": 0.5,
    
    # Features to use for content-based filtering
    "content_features": ["name", "main_category", "sub_category"],
    
    # TF-IDF vectorizer settings
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),  # Unigrams and bigrams
    "tfidf_stop_words": "english",
    
    # Similarity threshold
    "min_similarity_threshold": 0.05,
    
    # Cache settings
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,
    
    # Default number of recommendations
    "default_n_recommendations": 6
}
