from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(500), nullable=False)
    main_category = db.Column(db.String(100), index=True)
    sub_category = db.Column(db.String(100), index=True)
    image = db.Column(db.String(500))
    link = db.Column(db.String(500))
    ratings = db.Column(db.Float, default=0, index=True)
    no_of_ratings = db.Column(db.Integer, default=0)
    discount_price = db.Column(db.Float, default=0, index=True)
    actual_price = db.Column(db.Float, default=0)
    product_id = db.Column(db.String(50), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __setattr__(self, name, value):
        # Validate ratings to be between 0 and 5
        if name == 'ratings' and value is not None:
            value = max(0, min(5.0, float(value)))
        super().__setattr__(name, value)

class Rating(db.Model):
    __tablename__ = 'ratings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    product_id = db.Column(db.String(50), db.ForeignKey('products.product_id'))
    rating = db.Column(db.Float)
    review = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    product_id = db.Column(db.String(50), db.ForeignKey('products.product_id'))
    score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PageView(db.Model):
    __tablename__ = 'page_views'
    id = db.Column(db.Integer, primary_key=True)
    page_name = db.Column(db.String(100), nullable=False, index=True)
    visit_count = db.Column(db.Integer, default=1)
    date = db.Column(db.Date, default=datetime.utcnow, index=True)

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    id = db.Column(db.Integer, primary_key=True)
    recommendation_count = db.Column(db.Integer, default=0)
    date = db.Column(db.Date, default=datetime.utcnow, index=True)

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
