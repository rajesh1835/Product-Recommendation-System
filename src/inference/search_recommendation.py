"""
Product Search-Based Recommendation System
Uses CSV files for data - No database required
"""

import os
import pandas as pd
import numpy as np


# -----------------------------
# PATH HANDLING
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# CSV file paths
PRODUCTS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_products_with_reviews.csv")
PRODUCTS_CSV_ALT = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_rec.csv")


class ProductRecommender:
    def __init__(self):
        """Initialize the recommender with CSV data"""
        self.products = None
        self._load_data()
    
    def _load_data(self):
        """Load products data from CSV file"""
        # Try primary CSV first, then alternative
        if os.path.exists(PRODUCTS_CSV):
            csv_path = PRODUCTS_CSV
        elif os.path.exists(PRODUCTS_CSV_ALT):
            csv_path = PRODUCTS_CSV_ALT
        else:
            print(f"⚠️ No CSV file found. Please ensure data exists in data/processed/")
            self.products = pd.DataFrame()
            return
        
        print(f"📂 Loading products from: {csv_path}")
        self.products = pd.read_csv(csv_path, low_memory=False)
        
        # Standardize column names (lowercase)
        self.products.columns = self.products.columns.str.lower().str.strip()
        
        # Rename columns for consistency
        column_mapping = {
            'productid': 'product_id',
            'userid': 'user_id',
        }
        self.products = self.products.rename(columns=column_mapping)
        
        # Generate product_id if not exists
        if 'product_id' not in self.products.columns:
            self.products['product_id'] = ['P' + str(i).zfill(6) for i in range(len(self.products))]
        
        # Clean price columns
        self._clean_price_columns()
        
        print(f"✅ Loaded {len(self.products):,} products")
    
    def _clean_price_columns(self):
        """Clean price and rating columns"""
        def clean_price(val):
            if pd.isna(val):
                return 0.0
            val = str(val).replace('₹', '').replace(',', '').replace('$', '').strip()
            try:
                return float(val)
            except:
                return 0.0
        
        def clean_rating(val):
            if pd.isna(val):
                return 0.0
            try:
                return float(str(val).strip())
            except:
                return 0.0
        
        def clean_count(val):
            if pd.isna(val):
                return 0
            val = str(val).replace(',', '').strip()
            try:
                return int(float(val))
            except:
                return 0
        
        if 'discount_price' in self.products.columns:
            self.products['discount_price'] = self.products['discount_price'].apply(clean_price)
        if 'actual_price' in self.products.columns:
            self.products['actual_price'] = self.products['actual_price'].apply(clean_price)
        if 'ratings' in self.products.columns:
            self.products['ratings'] = self.products['ratings'].apply(clean_rating)
        if 'no_of_ratings' in self.products.columns:
            self.products['no_of_ratings'] = self.products['no_of_ratings'].apply(clean_count)
    
    # -----------------------------
    # SEARCH PRODUCTS (CSV)
    # -----------------------------
    def search_products(self, query, top_n=20):
        """Search products by name using CSV data"""
        if self.products.empty:
            return pd.DataFrame()
        
        query = query.lower().strip()
        
        # Filter products where name contains the query
        mask = self.products['name'].str.lower().str.contains(query, na=False)
        results = self.products[mask].copy()
        
        # Sort by popularity (no_of_ratings)
        if 'no_of_ratings' in results.columns:
            results = results.sort_values('no_of_ratings', ascending=False)
        
        # Select columns and limit results
        columns = ['product_id', 'name', 'main_category', 'ratings', 
                   'actual_price', 'image', 'no_of_ratings', 'discount_price']
        available_cols = [c for c in columns if c in results.columns]
        
        results = results[available_cols].head(top_n)
        
        # Rename for consistency
        results = results.rename(columns={'product_id': 'ProductId'})
        
        return results
    
    # -----------------------------
    # GET SIMILAR PRODUCTS (Category-based)
    # -----------------------------
    def get_similar_products(self, product_id, top_n=6):
        """Get similar products from same sub-category"""
        if self.products.empty:
            return pd.DataFrame()
        
        # Find the product
        product = self.products[self.products['product_id'] == product_id]
        if product.empty:
            return pd.DataFrame()
        
        sub_category = product.iloc[0].get('sub_category', None)
        if pd.isna(sub_category):
            return pd.DataFrame()
        
        # Get products from same sub-category (excluding current product)
        mask = (self.products['sub_category'] == sub_category) & (self.products['product_id'] != product_id)
        similar = self.products[mask].copy()
        
        # Sort by popularity
        if 'no_of_ratings' in similar.columns:
            similar = similar.sort_values('no_of_ratings', ascending=False)
        
        # Select columns
        columns = ['product_id', 'name', 'main_category', 'ratings', 'actual_price', 'image', 'discount_price']
        available_cols = [c for c in columns if c in similar.columns]
        similar = similar[available_cols].head(top_n)
        
        # Add similarity score
        similar['similarity_score'] = 0.9
        
        # Rename for consistency
        similar = similar.rename(columns={'product_id': 'ProductId'})
        
        return similar
    
    # -----------------------------
    # GET RELATED PRODUCTS (CATEGORY)
    # -----------------------------
    def get_related_products(self, product_id, top_n=6):
        """Get related products from same main category"""
        if self.products.empty:
            return pd.DataFrame()
        
        # Find the product
        product = self.products[self.products['product_id'] == product_id]
        if product.empty:
            return pd.DataFrame()
        
        main_category = product.iloc[0].get('main_category', None)
        if pd.isna(main_category):
            return pd.DataFrame()
        
        # Get products from same main category (excluding current product)
        mask = (self.products['main_category'] == main_category) & (self.products['product_id'] != product_id)
        related = self.products[mask].copy()
        
        # Sort by popularity
        if 'no_of_ratings' in related.columns:
            related = related.sort_values('no_of_ratings', ascending=False)
        
        # Select columns
        columns = ['product_id', 'name', 'main_category', 'ratings', 'actual_price', 'image', 'discount_price']
        available_cols = [c for c in columns if c in related.columns]
        related = related[available_cols].head(top_n)
        
        # Rename for consistency
        related = related.rename(columns={'product_id': 'ProductId'})
        
        return related
    
    # -----------------------------
    # GET POPULAR PRODUCTS
    # -----------------------------
    def get_popular_products(self, top_n=10):
        """Get most popular products from CSV data"""
        if self.products.empty:
            return pd.DataFrame()
        
        # Sort by popularity
        popular = self.products.copy()
        if 'no_of_ratings' in popular.columns:
            popular = popular.sort_values('no_of_ratings', ascending=False)
        
        # Select columns
        columns = ['product_id', 'name', 'main_category', 'ratings', 
                   'actual_price', 'image', 'no_of_ratings', 'discount_price']
        available_cols = [c for c in columns if c in popular.columns]
        popular = popular[available_cols].head(top_n)
        
        # Rename for consistency
        popular = popular.rename(columns={'product_id': 'ProductId'})
        
        return popular
    
    # -----------------------------
    # GET PRODUCT BY ID
    # -----------------------------
    def get_product_by_id(self, product_id):
        """Get a single product by ID"""
        if self.products.empty:
            return None
        
        product = self.products[self.products['product_id'] == product_id]
        if product.empty:
            return None
        
        # Return as dictionary
        result = product.iloc[0].to_dict()
        result['ProductId'] = result.pop('product_id', product_id)
        return result
    
    # -----------------------------
    # GET KPI STATS
    # -----------------------------
    def get_kpi_stats(self):
        """Get KPI statistics from CSV data"""
        if self.products.empty:
            return {
                'total_products': 0,
                'total_categories': 0,
                'avg_rating': 0,
                'avg_price': 0
            }
        
        total_products = len(self.products)
        total_categories = self.products['main_category'].nunique() if 'main_category' in self.products.columns else 0
        avg_rating = round(self.products['ratings'].mean(), 2) if 'ratings' in self.products.columns else 0
        avg_price = round(self.products['actual_price'].mean(), 2) if 'actual_price' in self.products.columns else 0
        
        return {
            'total_products': total_products,
            'total_categories': total_categories,
            'avg_rating': avg_rating,
            'avg_price': avg_price
        }

    # -----------------------------
    # GET CHART DATA
    # -----------------------------
    def get_chart_data(self):
        """Get data for dashboard charts"""
        if self.products.empty:
            return {"categories": [], "ratings": []}
        
        # Top 5 categories
        categories = []
        if 'main_category' in self.products.columns:
            cat_counts = self.products['main_category'].value_counts().head(5)
            categories = [{"main_category": cat, "count": count} for cat, count in cat_counts.items()]
        
        # Rating distribution
        ratings = []
        if 'ratings' in self.products.columns:
            rating_bins = self.products['ratings'].dropna().astype(int).value_counts().sort_index()
            ratings = [{"rating_bin": rating, "count": count} for rating, count in rating_bins.items()]
        
        return {
            "categories": categories,
            "ratings": ratings
        }


# -----------------------------
# DEMO
# -----------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" Product Recommender - CSV Mode ")
    print("=" * 50)
    
    rec = ProductRecommender()
    
    # Test KPI
    kpi = rec.get_kpi_stats()
    print(f"\nKPI Stats:")
    print(f"  Total Products: {kpi['total_products']:,}")
    print(f"  Categories: {kpi['total_categories']}")
    print(f"  Avg Rating: {kpi['avg_rating']}")
    print(f"  Avg Price: ₹{kpi['avg_price']:,.2f}")
    
    # Test search
    print("\nSearching for 'laptop'...")
    results = rec.search_products("laptop", top_n=3)
    print(f"  Found {len(results)} results")
    if not results.empty:
        for _, row in results.iterrows():
            name = str(row.get('name', 'Unknown'))[:50]
            print(f"    - {name}...")
    
    # Test popular
    print("\nPopular products:")
    popular = rec.get_popular_products(top_n=3)
    for _, row in popular.iterrows():
        name = str(row.get('name', 'Unknown'))[:50]
        print(f"  - {name}...")
    
    print("\n" + "=" * 50)
