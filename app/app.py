"""
SmartPic: Bridging User Interest and Product Content with Hybrid Machine Learning
Flask Web Application for Advanced Product Recommendations
Features: Hybrid AI Suggestions, Search, KPI Dashboard, MySQL
"""

import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config import Config
from src.components.database import db, Product, User, init_db
from src.inference.search_recommendation import ProductRecommender
from src.inference.hybrid_recommender import HybridRecommender

# Decorator for login required
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# APP CONFIGURATION
# -----------------------------
app = Flask(__name__)
app.config.from_object(Config)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-for-session')

# Initialize database
init_db(app)

# Initialize recommender
recommender = None
hybrid_recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        recommender = ProductRecommender()
    return recommender

def get_hybrid_recommender():
    """Get or initialize the hybrid recommender"""
    global hybrid_recommender
    if hybrid_recommender is None:
        try:
            hybrid_recommender = HybridRecommender(alpha=0.5)
        except Exception as e:
            print(f"⚠️ Error initializing hybrid recommender: {e}")
            return None
    return hybrid_recommender

# Get products from MySQL database
def get_products_from_db(limit=None):
    """Get products from MySQL database"""
    query = Product.query
    if limit:
        query = query.limit(limit)
    products = query.all()
    return [{'id': p.id, 'name': p.name, 'category': p.category, 'price': p.price, 'rating': p.rating} for p in products]

def search_products_db(query_text, limit=20):
    """Search products in MySQL database"""
    results = Product.query.filter(
        Product.name.ilike(f'%{query_text}%') | 
        Product.main_category.ilike(f'%{query_text}%')
    ).limit(limit).all()
    return results

# -----------------------------
# TEMPLATE FILTERS
# -----------------------------
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

# -----------------------------
# AUTHENTICATION ROUTES
# -----------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        if not username or not email or not password:
            flash('All fields required', 'error')
            return redirect(url_for('signup'))
        
        if password != password_confirm:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('signup'))
        
        with app.app_context():
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Username already exists', 'error')
                return redirect(url_for('signup'))
            
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                flash('Email already registered', 'error')
                return redirect(url_for('signup'))
            
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password required', 'error')
            return redirect(url_for('login'))
        
        with app.app_context():
            user = User.query.filter_by(username=username).first()
            
            if not user or not user.check_password(password):
                flash('Invalid username or password', 'error')
                return redirect(url_for('login'))
            
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

# -----------------------------
# KPI DATA
# -----------------------------
def get_kpi_data():
    """Get KPI metrics from database"""
    with app.app_context():
        total_products = Product.query.count()
        avg_rating = db.session.query(db.func.avg(Product.ratings)).scalar() or 0
        avg_price = db.session.query(db.func.avg(Product.discount_price)).scalar() or 0
        categories = db.session.query(db.func.count(db.distinct(Product.main_category))).scalar() or 0
        
        return {
            'total_products': total_products,
            'avg_rating': round(avg_rating, 2),
            'avg_price': round(avg_price, 2),
            'total_categories': categories,
            'categories': categories
        }

# -----------------------------
# ROUTES - HOME
# -----------------------------
@app.route('/')
def index():
    with app.app_context():
        popular = Product.query.order_by(Product.ratings.desc()).limit(8).all()
        kpi = get_kpi_data()
        popular_data = [{
            'ProductId': p.product_id, 
            'name': p.name, 
            'price': p.discount_price, 
            'rating': p.ratings,
            'main_category': p.main_category,
            'ratings': p.ratings,
            'discount_price': p.discount_price,
            'image': p.image
        } for p in popular]
        return render_template('index.html', popular=popular_data, kpi=kpi)

# -----------------------------
# ROUTES - DASHBOARD (Improved with Analytics)
# -----------------------------
@app.route('/dashboard')
def dashboard():
    from src.components.database import PageView, Recommendation
    from datetime import date, timedelta
    import threading
    
    with app.app_context():
        # Track page view asynchronously (don't wait for it)
        def track_visit():
            try:
                today = date.today()
                page_view = PageView.query.filter_by(page_name='dashboard', date=today).first()
                if page_view:
                    page_view.visit_count += 1
                else:
                    page_view = PageView(page_name='dashboard', visit_count=1, date=today)
                db.session.add(page_view)
                db.session.commit()
            except:
                pass
        
        threading.Thread(target=track_visit, daemon=True).start()
        
        # Execute all product queries together for efficiency
        popular = Product.query.order_by(Product.ratings.desc()).limit(6).all()
        top_categories = db.session.query(
            Product.main_category,
            db.func.count(Product.id).label('count'),
            db.func.avg(Product.ratings).label('avg_rating')
        ).group_by(Product.main_category).order_by(db.func.count(Product.id).desc()).limit(5).all()
        
        top_rated = Product.query.filter(Product.ratings > 0).order_by(Product.ratings.desc()).limit(5).all()
        best_deals = Product.query.filter(
            (Product.actual_price - Product.discount_price) > 0
        ).order_by((Product.actual_price - Product.discount_price).desc()).limit(5).all()
        
        # Get KPI data
        kpi = get_kpi_data()
        
        # Cache analytics or use simpler queries
        last_week = date.today() - timedelta(days=7)
        
        # Get aggregated totals with separate queries to avoid cartesian product
        total_visitors = db.session.query(db.func.sum(PageView.visit_count)).scalar() or 0
        total_recommendations = db.session.query(db.func.sum(Recommendation.recommendation_count)).scalar() or 0
        
        # Get last 7 days with simpler queries
        daily_visitors = db.session.query(
            PageView.date,
            db.func.sum(PageView.visit_count).label('count')
        ).filter(PageView.date >= last_week).group_by(PageView.date).order_by(PageView.date).all()
        
        daily_recommendations = db.session.query(
            Recommendation.date,
            db.func.sum(Recommendation.recommendation_count).label('count')
        ).filter(Recommendation.date >= last_week).group_by(Recommendation.date).order_by(Recommendation.date).all()
        
        # Convert to chart data efficiently
        visitors_chart_data = {
            'dates': [str(v[0]) for v in daily_visitors],
            'counts': [v[1] or 0 for v in daily_visitors]
        }
        
        recommendations_chart_data = {
            'dates': [str(r[0]) for r in daily_recommendations],
            'counts': [r[1] or 0 for r in daily_recommendations]
        }
        
        # Build response data
        popular_data = [{
            'ProductId': p.product_id, 
            'name': p.name, 
            'price': p.discount_price, 
            'rating': p.ratings,
            'main_category': p.main_category,
            'ratings': p.ratings,
            'discount_price': p.discount_price,
            'image': p.image
        } for p in popular]
        
        categories_data = [{
            'name': cat[0],
            'count': cat[1],
            'avg_rating': round(cat[2], 2) if cat[2] else 0
        } for cat in top_categories]
        
        top_rated_data = [{
            'ProductId': p.product_id,
            'name': p.name,
            'rating': p.ratings,
            'no_of_ratings': p.no_of_ratings,
            'image': p.image
        } for p in top_rated]
        
        best_deals_data = [{
            'ProductId': p.product_id,
            'name': p.name,
            'original_price': p.actual_price,
            'discount_price': p.discount_price,
            'savings': round(p.actual_price - p.discount_price, 2),
            'discount_pct': round(((p.actual_price - p.discount_price) / p.actual_price * 100)) if p.actual_price > 0 else 0,
            'image': p.image
        } for p in best_deals]
        
        return render_template('dashboard.html', 
                             popular=popular_data, 
                             kpi=kpi,
                             categories=categories_data,
                             top_rated=top_rated_data,
                             best_deals=best_deals_data,
                             username=session.get('username', 'Guest'),
                             total_visitors=total_visitors,
                             total_recommendations=total_recommendations,
                             visitors_chart_data=visitors_chart_data,
                             recommendations_chart_data=recommendations_chart_data)

# -----------------------------
# ROUTES - SEARCH WITH FILTERS
# -----------------------------
@app.route('/search')
def search():
    query = request.args.get('q', '')
    main_category = request.args.get('main_category', '')
    sub_category = request.args.get('sub_category', '')
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    min_rating = request.args.get('min_rating', type=float)
    
    with app.app_context():
        # Get all categories for filter dropdowns
        all_categories = db.session.query(db.distinct(Product.main_category)).filter(
            Product.main_category.isnot(None)
        ).all()
        all_categories = [cat[0] for cat in all_categories if cat[0]]
        
        all_subcategories = db.session.query(db.distinct(Product.sub_category)).filter(
            Product.sub_category.isnot(None)
        ).all()
        all_subcategories = [subcat[0] for subcat in all_subcategories if subcat[0]]
        
        # Build query
        results_query = Product.query
        
        if query:
            results_query = results_query.filter(
                Product.name.ilike(f'%{query}%') | 
                Product.main_category.ilike(f'%{query}%') |
                Product.sub_category.ilike(f'%{query}%')
            )
        
        if main_category:
            results_query = results_query.filter(Product.main_category == main_category)
        
        if sub_category:
            results_query = results_query.filter(Product.sub_category == sub_category)
        
        if min_price:
            results_query = results_query.filter(Product.discount_price >= min_price)
        
        if max_price:
            results_query = results_query.filter(Product.discount_price <= max_price)
        
        if min_rating:
            results_query = results_query.filter(Product.ratings >= min_rating)
        
        results = results_query.order_by(Product.ratings.desc()).limit(50).all()
        
        # Get suggestions based on top categories in results
        suggestions = []
        if results:
            top_cats = set([p.main_category for p in results[:5]])
            for cat in top_cats:
                suggestions.extend(
                    Product.query.filter(
                        Product.main_category == cat,
                        ~Product.id.in_([p.id for p in results])
                    ).order_by(Product.ratings.desc()).limit(3).all()
                )
        
        results_list = [{
            'ProductId': p.product_id, 
            'name': p.name, 
            'main_category': p.main_category,
            'sub_category': p.sub_category or '',
            'price': p.discount_price, 
            'ratings': p.ratings,
            'no_of_ratings': p.no_of_ratings,
            'discount_price': p.discount_price,
            'actual_price': p.actual_price or 0,
            'image': p.image
        } for p in results]
        
        suggestions_list = [{
            'product_id': p.product_id,
            'name': p.name, 
            'ratings': p.ratings,
            'discount_price': p.discount_price,
            'image': p.image
        } for p in suggestions[:6]]
        
        # Track recommendations if suggestions were shown
        if suggestions_list:
            from src.components.database import Recommendation
            from datetime import date
            today = date.today()
            rec = Recommendation.query.filter_by(date=today).first()
            if rec:
                rec.recommendation_count += len(suggestions_list)
            else:
                rec = Recommendation(recommendation_count=len(suggestions_list), date=today)
            db.session.add(rec)
            db.session.commit()
        
        return render_template('search.html', 
                             results=results_list, 
                             query=query,
                             all_categories=all_categories,
                             all_subcategories=all_subcategories,
                             suggestions=suggestions_list)

# -----------------------------
# ROUTES - PRODUCT DETAIL
# -----------------------------
@app.route('/product/<product_id>')
def product_detail(product_id):
    with app.app_context():
        product = Product.query.filter_by(product_id=product_id).first()
        
        if not product:
            flash('Product not found', 'error')
            return redirect(url_for('index'))
        
        # Try to use hybrid recommender for similar products
        hybrid_recs = []
        hr = get_hybrid_recommender()
        if hr:
            user_id = session.get('user_id')  # Get logged-in user if available
            hybrid_recs = hr.get_hybrid_recommendations(product_id, user_id=user_id, n=6)
        
        # Fallback to category-based if hybrid fails
        if not hybrid_recs:
            similar = Product.query.filter(
                Product.main_category == product.main_category,
                Product.id != product_id
            ).order_by(Product.ratings.desc()).limit(6).all()
            
            hybrid_recs = [{
                'ProductId': p.product_id,
                'name': p.name, 
                'main_category': p.main_category,
                'discount_price': p.discount_price,
                'actual_price': p.actual_price or 0,
                'ratings': p.ratings,
                'image': p.image,
                'similarity_score': 0,
                'rec_type': 'category'
            } for p in similar]
        
        # Get related products from same category (different from hybrid recs)
        hybrid_pids = [r.get('ProductId') for r in hybrid_recs]
        related = Product.query.filter(
            Product.main_category == product.main_category,
            Product.product_id != product_id,
            ~Product.product_id.in_(hybrid_pids) if hybrid_pids else True
        ).order_by(Product.ratings.desc()).limit(6).all()
        
        related_list = [{
            'ProductId': p.product_id,
            'name': p.name, 
            'main_category': p.main_category,
            'discount_price': p.discount_price,
            'actual_price': p.actual_price or 0,
            'ratings': p.ratings,
            'image': p.image,
            'rec_type': 'related'
        } for p in related]
        
        # Track recommendation
        from src.components.database import Recommendation
        from datetime import date
        today = date.today()
        rec = Recommendation.query.filter_by(date=today).first()
        if rec:
            rec.recommendation_count += len(hybrid_recs) + len(related_list)
        else:
            rec = Recommendation(recommendation_count=len(hybrid_recs) + len(related_list), date=today)
        db.session.add(rec)
        db.session.commit()
        
        # Build product dict with all needed fields
        product_dict = {
            'ProductId': product.product_id, 
            'name': product.name, 
            'main_category': product.main_category,
            'sub_category': product.sub_category or '',
            'discount_price': product.discount_price,
            'actual_price': product.actual_price or 0,
            'ratings': product.ratings,
            'no_of_ratings': product.no_of_ratings or 0,
            'image': product.image
        }
        
        return render_template('product.html', 
                             product=product_dict,
                             similar=hybrid_recs,
                             related=related_list,
                             recommendation_type='hybrid' if hr else 'category')

# -----------------------------
# API ROUTES
# -----------------------------
@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json()
    query = data.get('query', '')
    
    with app.app_context():
        results = search_products_db(query, limit=10)
        results_list = [{
            'ProductId': p.product_id,
            'name': p.name, 
            'main_category': p.main_category, 
            'price': p.discount_price, 
            'ratings': p.ratings,
            'image': p.image
        } for p in results]
    
    return jsonify(results_list)

@app.route('/api/recommend/<product_id>')
def api_recommend(product_id):
    with app.app_context():
        product = Product.query.filter_by(product_id=product_id).first()
        if not product:
            return jsonify({'error': 'Product not found'}), 404
        
        # Try hybrid recommendations first
        hr = get_hybrid_recommender()
        recommendation_type = 'category'
        
        if hr:
            user_id = session.get('user_id')
            hybrid_recs = hr.get_hybrid_recommendations(product_id, user_id=user_id, n=6)
            if hybrid_recs:
                recommendation_type = 'hybrid'
                return jsonify({
                    'similar': hybrid_recs,
                    'related': hybrid_recs,
                    'recommendation_type': recommendation_type
                })
        
        # Fallback to category-based
        similar = Product.query.filter(
            Product.main_category == product.main_category,
            Product.id != product_id
        ).order_by(Product.ratings.desc()).limit(6).all()
        
        similar_list = [{
            'ProductId': p.product_id,
            'name': p.name, 
            'discount_price': p.discount_price, 
            'ratings': p.ratings,
            'image': p.image,
            'rec_type': 'category'
        } for p in similar]
        
        return jsonify({
            'similar': similar_list,
            'related': similar_list,
            'recommendation_type': recommendation_type
        })

@app.route('/api/stats')
def api_stats():
    rec = get_recommender()
    data = rec.get_chart_data()
    return jsonify(data)

# -----------------------------
# RUN
# -----------------------------
if __name__ == '__main__':
    print("\n🚀 Starting Flask server...")
    print("📍 Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)