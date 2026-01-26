import sys
sys.path.insert(0, 'd:\\Product Recommendation System')

from flask import Flask
from src.components.database import db, Product
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    print("🗑️  Dropping old products table...")
    db.drop_all()
    print("✅ Old tables dropped")
    
    print("📝 Creating new tables...")
    db.create_all()
    print("✅ New tables created with correct columns")
