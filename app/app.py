"""
Flask Web Application for Product Recommendation System
Features: Login/Signup, Product Search, Recommendations, KPI Dashboard
"""

import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.inference.search_recommendation import ProductRecommender

# -----------------------------
# APP CONFIGURATION
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize recommender
recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        recommender = ProductRecommender()
    return recommender

# -----------------------------
# USER MODEL
# -----------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@app.template_filter('clean_price')
def clean_price_filter(val):
    """Jinja filter to clean currency strings and return float"""
    if val is None: return 0.0
    s_val = str(val)
    s_val = s_val.replace('₹', '').replace(',', '').replace('$', '').strip()
    try:
        return float(s_val)
    except ValueError:
        return 0.0

@app.template_filter('discount_pct')
def discount_pct_filter(product):
    """Calculate discount percentage from product dict"""
    try:
        actual = clean_price_filter(product.get('actual_price'))
        discount = clean_price_filter(product.get('discount_price'))
        if actual > 0 and discount < actual:
            return round(((actual - discount) / actual) * 100)
    except:
        pass
    return 0

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -----------------------------
# KPI DATA
# -----------------------------
def get_kpi_data():
    """Get KPI metrics from database"""
    rec = get_recommender()
    return rec.get_kpi_stats()

# -----------------------------
# ROUTES - HOME
# -----------------------------
@app.route('/')
def index():
    rec = get_recommender()
    popular = rec.get_popular_products(top_n=8)
    kpi = get_kpi_data()
    return render_template('index.html', popular=popular.to_dict('records'), kpi=kpi)

# -----------------------------
# ROUTES - AUTH
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# -----------------------------
# ROUTES - DASHBOARD
# -----------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    rec = get_recommender()
    popular = rec.get_popular_products(top_n=6)
    kpi = get_kpi_data()
    return render_template('dashboard.html', popular=popular.to_dict('records'), kpi=kpi)

# -----------------------------
# ROUTES - SEARCH
# -----------------------------
@app.route('/search')
def search():
    query = request.args.get('q', '')
    
    if not query:
        return render_template('search.html', results=[], query='')
    
    rec = get_recommender()
    results = rec.search_products(query, top_n=20)
    
    return render_template('search.html', results=results.to_dict('records'), query=query)

# -----------------------------
# ROUTES - PRODUCT DETAIL
# -----------------------------
@app.route('/product/<product_id>')
def product_detail(product_id):
    rec = get_recommender()
    
    # Get product details from database
    cursor = rec.conn.execute("""
        SELECT product_id as ProductId, name, main_category, sub_category,
               ratings, no_of_ratings, discount_price, actual_price, image, link
        FROM products WHERE product_id = ?
    """, (product_id,))
    row = cursor.fetchone()
    
    if not row:
        flash('Product not found', 'error')
        return redirect(url_for('index'))
    
    product = dict(row)
    
    # Get recommendations
    similar = rec.get_similar_products(product_id, top_n=6)
    related = rec.get_related_products(product_id, top_n=6)
    
    return render_template('product.html', 
                         product=product,
                         similar=similar.to_dict('records'),
                         related=related.to_dict('records'))

# -----------------------------
# API ROUTES
# -----------------------------
@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json()
    query = data.get('query', '')
    
    rec = get_recommender()
    results = rec.search_products(query, top_n=10)
    
    return jsonify(results.to_dict('records'))

@app.route('/api/recommend/<product_id>')
def api_recommend(product_id):
    rec = get_recommender()
    similar = rec.get_similar_products(product_id, top_n=6)
    related = rec.get_related_products(product_id, top_n=6)
    
    return jsonify({
        'similar': similar.to_dict('records'),
        'related': related.to_dict('records')
    })

@app.route('/api/stats')
def api_stats():
    rec = get_recommender()
    data = rec.get_chart_data()
    return jsonify(data)

# -----------------------------
# INITIALIZE
# -----------------------------
def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database initialized!")

# -----------------------------
# RUN
# -----------------------------
if __name__ == '__main__':
    init_db()
    print("\n🚀 Starting Flask server...")
    print("📍 Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
