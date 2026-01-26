import pandas as pd
import sys
sys.path.insert(0, 'd:\\Product Recommendation System')

from flask import Flask
from src.components.database import db, Product
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

def migrate_products():
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check if data already exists
        existing_count = Product.query.count()
        if existing_count > 0:
            print(f"⚠️  Database already has {existing_count} products. Clearing and re-migrating...")
            Product.query.delete()
            db.session.commit()
        
        # Load CSV
        try:
            df = pd.read_csv('data/processed/amazon_products_with_reviews.csv')
            print(f"📊 Loaded {len(df)} products from CSV")
        except FileNotFoundError:
            print("❌ CSV file not found. Try: data/processed/amazon_products_with_reviews.csv")
            return
        
        # Insert into database
        count = 0
        for _, row in df.iterrows():
            try:
                product = Product(
                    name=str(row.get('name', '')),
                    main_category=str(row.get('main_category', '')),
                    sub_category=str(row.get('sub_category', '')),
                    image=str(row.get('image', '')) if pd.notna(row.get('image')) else None,
                    link=str(row.get('link', '')) if pd.notna(row.get('link')) else None,
                    ratings=float(row.get('ratings', 0)) if pd.notna(row.get('ratings')) else 0,
                    no_of_ratings=int(row.get('no_of_ratings', 0)) if pd.notna(row.get('no_of_ratings')) else 0,
                    discount_price=float(row.get('discount_price', 0)) if pd.notna(row.get('discount_price')) else 0,
                    actual_price=float(row.get('actual_price', 0)) if pd.notna(row.get('actual_price')) else 0,
                    product_id=str(row.get('ProductId', ''))
                )
                db.session.add(product)
                count += 1
                
                if count % 100 == 0:
                    db.session.commit()
                    print(f"  ✓ Inserted {count} products...")
            except Exception as e:
                print(f"⚠️  Error inserting row: {e}")
                continue
        
        db.session.commit()
        print(f"✅ Successfully migrated {count} products to MySQL")

if __name__ == '__main__':
    migrate_products()
