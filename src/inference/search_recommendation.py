"""
Product Search-Based Recommendation System
Uses SQLite database for fast queries - No TF-IDF (faster startup)
"""

import os
import sqlite3
import pandas as pd


# -----------------------------
# PATH HANDLING
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DB_PATH = os.path.join(PROJECT_ROOT, "app", "database.db")


class ProductRecommender:
    def __init__(self):
        """Initialize the recommender with database connection"""
        # Connect to database
        self.db_path = DB_PATH
        self.conn = None
        self._connect()
        
        # For compatibility - load empty products dataframe
        self.products = pd.DataFrame()
    
    def _connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
    
    # -----------------------------
    # SEARCH PRODUCTS (DATABASE)
    # -----------------------------
    def search_products(self, query, top_n=20):
        """Search products by name using database"""
        cursor = self.conn.execute("""
            SELECT product_id as ProductId, name, main_category, ratings, 
                   actual_price, image, no_of_ratings, discount_price
            FROM products 
            WHERE name LIKE ? 
            ORDER BY no_of_ratings DESC
            LIMIT ?
        """, (f'%{query}%', top_n))
        
        results = cursor.fetchall()
        df = pd.DataFrame([dict(row) for row in results])
        return df
    
    # -----------------------------
    # GET SIMILAR PRODUCTS (Category-based)
    # -----------------------------
    def get_similar_products(self, product_id, top_n=6):
        """Get similar products from same sub-category"""
        # Get product sub-category
        cursor = self.conn.execute(
            "SELECT sub_category FROM products WHERE product_id = ?",
            (product_id,)
        )
        row = cursor.fetchone()
        if not row:
            return pd.DataFrame()
        
        sub_category = row['sub_category']
        
        # Get top products from same sub-category
        cursor = self.conn.execute("""
            SELECT product_id as ProductId, name, main_category, ratings, 
                   actual_price, image
            FROM products 
            WHERE sub_category = ? AND product_id != ?
            ORDER BY no_of_ratings DESC
            LIMIT ?
        """, (sub_category, product_id, top_n))
        
        df = pd.DataFrame([dict(row) for row in cursor.fetchall()])
        if len(df) > 0:
            df['similarity_score'] = 0.9  # Default similarity for same category
        return df
    
    # -----------------------------
    # GET RELATED PRODUCTS (CATEGORY)
    # -----------------------------
    def get_related_products(self, product_id, top_n=6):
        """Get related products from same main category"""
        # Get product category
        cursor = self.conn.execute(
            "SELECT main_category FROM products WHERE product_id = ?",
            (product_id,)
        )
        row = cursor.fetchone()
        if not row:
            return pd.DataFrame()
        
        category = row['main_category']
        
        # Get top products from same category
        cursor = self.conn.execute("""
            SELECT product_id as ProductId, name, main_category, ratings, 
                   actual_price, image
            FROM products 
            WHERE main_category = ? AND product_id != ?
            ORDER BY no_of_ratings DESC
            LIMIT ?
        """, (category, product_id, top_n))
        
        return pd.DataFrame([dict(row) for row in cursor.fetchall()])
    
    # -----------------------------
    # GET POPULAR PRODUCTS
    # -----------------------------
    def get_popular_products(self, top_n=10):
        """Get most popular products from database"""
        cursor = self.conn.execute("""
            SELECT product_id as ProductId, name, main_category, ratings, 
                   actual_price, image, no_of_ratings
            FROM products 
            ORDER BY no_of_ratings DESC 
            LIMIT ?
        """, (top_n,))
        
        return pd.DataFrame([dict(row) for row in cursor.fetchall()])
    
    # -----------------------------
    # GET KPI STATS
    # -----------------------------
    def get_kpi_stats(self):
        """Get KPI statistics from database"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_products,
                COUNT(DISTINCT main_category) as total_categories,
                ROUND(AVG(ratings), 2) as avg_rating,
                ROUND(AVG(actual_price), 2) as avg_price
            FROM products
        """)
        row = cursor.fetchone()
        return dict(row)


    # -----------------------------
    # GET CHART DATA
    # -----------------------------
    def get_chart_data(self):
        """Get data for dashboard charts"""
        # Top 5 categories
        cursor = self.conn.execute("""
            SELECT main_category, COUNT(*) as count 
            FROM products 
            WHERE main_category IS NOT NULL 
            GROUP BY main_category 
            ORDER BY count DESC 
            LIMIT 5
        """)
        categories = cursor.fetchall()
        
        # Rating distribution
        cursor = self.conn.execute("""
            SELECT CAST(ratings AS INTEGER) as rating_bin, COUNT(*) as count
            FROM products
            WHERE ratings > 0
            GROUP BY rating_bin
            ORDER BY rating_bin
        """)
        ratings = cursor.fetchall()
        
        return {
            "categories": [dict(r) for r in categories],
            "ratings": [dict(r) for r in ratings]
        }


# -----------------------------
# DEMO
# -----------------------------
if __name__ == "__main__":
    rec = ProductRecommender()
    
    # Test KPI
    kpi = rec.get_kpi_stats()
    print(f"\nKPI Stats:")
    print(f"  Total Products: {kpi['total_products']:,}")
    print(f"  Categories: {kpi['total_categories']}")
    print(f"  Avg Rating: {kpi['avg_rating']}")
    
    # Test search
    print("\nSearching for 'laptop'...")
    results = rec.search_products("laptop", top_n=3)
    print(f"  Found {len(results)} results")
    
    # Test popular
    print("\nPopular products:")
    popular = rec.get_popular_products(top_n=3)
    for i, row in popular.iterrows():
        name = str(row['name'])[:50] if row['name'] else 'Unknown'
        print(f"  - {name}...")
