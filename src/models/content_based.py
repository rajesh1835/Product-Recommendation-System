
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    """
    Content-Based Filtering Recommendation System
    Uses TF-IDF on product names and categories to find similar products.
    """
    
    def __init__(self, df):
        """
        Initialize with product dataframe
        Args:
            df (pd.DataFrame): Dataframe containing 'ProductId', 'name', 'main_category', 'sub_category'
        """
        self.df = df.copy().reset_index(drop=True)
        self.tfidf_matrix = None
        self.indices = None
        
        print("   Computing TF-IDF matrix for Content-Based Filtering...")
        self._prepare_vectorizer()

    def _prepare_vectorizer(self):
        # Combine text features for similarity
        # We allow Description and Review if available, otherwise just Name/Category
        text_cols = ['name', 'main_category', 'sub_category']
        
        # Ensure columns exist and fill NaNs
        for col in text_cols:
            if col not in self.df.columns:
                self.df[col] = ''
            else:
                self.df[col] = self.df[col].fillna('')

        # Create a "soup" of metadata
        self.df['combined_features'] = (
            self.df['name'] + " " + 
            self.df['main_category'] + " " + 
            self.df['sub_category']
        )
        
        # Reduce dimensionality if dataset is too large (optimization)
        # For 25k rows, distinct vocabulary might be large, but usually okay for modern memory
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])
        
        # Quick lookup for product index
        self.indices = pd.Series(self.df.index, index=self.df['ProductId']).drop_duplicates()

    def get_recommendations(self, product_id, n=5):
        """
        Get top N similar products based on content similarity
        """
        if product_id not in self.indices:
            print(f"   ⚠️ ProductId {product_id} not found in content database.")
            return pd.DataFrame()

        idx = self.indices[product_id]
        
        # Compute Cosine Similarity
        # linear_kernel is faster than cosine_similarity for sparse matrices
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix)
        
        # Get pairwise scores
        sim_scores = list(enumerate(cosine_sim[0]))
        
        # Sort based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N (ignoring index 0 because that is the product itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get product indices
        product_indices = [i[0] for i in sim_scores]
        
        # Return results
        results = self.df.iloc[product_indices][['ProductId', 'name', 'main_category', 'actual_price', 'ratings']].copy()
        results['similarity_score'] = [i[1] for i in sim_scores]
        
        return results
