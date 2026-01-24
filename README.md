# 🛍️ E-Commerce Product Recommendation System

An AI-powered product recommendation system built with Python, Flask, and Machine Learning algorithms. This system provides personalized product suggestions using collaborative filtering techniques.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![scikit-surprise](https://img.shields.io/badge/Surprise-1.1.4-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [ML Pipeline](#-ml-pipeline)
- [Models](#-models)
- [API Endpoints](#-api-endpoints)
- [Screenshots](#-screenshots)
- [Author](#-author)

## ✨ Features

- 🔐 **User Authentication** - Secure login/signup with Flask-Login
- 🔍 **Product Search** - Fast search with SQLite database
- 🎯 **Similar Products** - Category-based product recommendations
- 📊 **KPI Dashboard** - Real-time statistics and charts
- 🌙 **Dark/Light Theme** - Modern UI with theme toggle
- 📱 **Responsive Design** - Works on desktop and mobile
- 🤖 **ML-Powered** - Collaborative filtering using Surprise library

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python, Flask, Flask-Login, Flask-SQLAlchemy |
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **Database** | SQLite |
| **ML/Data** | Pandas, NumPy, Scikit-learn, Scikit-Surprise |
| **Visualization** | Matplotlib, Seaborn |

## 📁 Project Structure

```
Product Recommendation System/
├── app/                          # Flask Web Application
│   ├── app.py                    # Main Flask application
│   ├── static/
│   │   ├── css/style.css         # Stylesheet
│   │   └── js/main.js            # JavaScript
│   └── templates/                # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── login.html
│       ├── signup.html
│       ├── dashboard.html
│       ├── search.html
│       └── product.html
│
├── src/                          # Source Code
│   ├── components/               # Data processing modules
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   ├── data_combiner.py
│   │   ├── eda.py
│   │   └── database.py
│   ├── models/                   # ML models
│   │   ├── baseline_model.py
│   │   ├── knn_model.py
│   │   ├── svd_model.py
│   │   └── knn_with_kmeans.py
│   ├── inference/                # Prediction modules
│   │   ├── predictor.py
│   │   └── search_recommendation.py
│   └── testing/                  # A/B testing
│       └── ab_test.py
│
├── artifacts/                    # Generated outputs
│   ├── eda/                      # EDA visualizations
│   └── final_model_config.json
│
├── presentation/                 # Project presentations
│
├── main.py                       # ML Pipeline entry point
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/rajesh1835/Product-Recommendation-System.git
   cd Product-Recommendation-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download Amazon Products dataset
   - Place it in `data/raw/Amazon-Products.csv`

5. **Run the ML pipeline** (optional - trains models)
   ```bash
   python main.py
   ```

6. **Initialize the database**
   ```bash
   python -c "from src.components.database import init_database; init_database()"
   ```

7. **Start the web application**
   ```bash
   cd app
   python app.py
   ```

8. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

## 💻 Usage

### Running the ML Pipeline
```bash
python main.py
```
This will:
1. Load and combine raw data
2. Clean and preprocess data
3. Generate EDA visualizations
4. Train and compare 4 ML models
5. Run A/B testing
6. Save the best model

### Running the Web App
```bash
cd app
python app.py
```

## 🔄 ML Pipeline

The pipeline consists of 11 steps:

```
Step 1:  Load Raw Data
Step 2:  Combine Datasets
Step 3:  Data Cleaning
Step 4:  Sampling (25,000 rows)
Step 5:  Dataset Summary
Step 6:  Exploratory Data Analysis (EDA)
Step 7:  Model Training & Evaluation
Step 8:  A/B Testing
Step 9:  Hyperparameter Tuning (GridSearch)
Step 10: Save Final Model Config
Step 11: Generate Predictions Demo
```

## 🤖 Models

Four collaborative filtering models are compared:

| Model | Description | RMSE |
|-------|-------------|------|
| **KNNBasic** | K-Nearest Neighbors (User-based) | 0.611 ✅ |
| **KNNWithMeans** | KNN with mean ratings | 0.611 |
| **BaselineOnly** | Baseline predictor | 0.615 |
| **SVD** | Singular Value Decomposition | 0.619 |

**Best Model:** KNNBasic with cosine similarity

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with popular products |
| `/login` | GET/POST | User login |
| `/signup` | GET/POST | User registration |
| `/logout` | GET | User logout |
| `/dashboard` | GET | User dashboard |
| `/search` | GET | Product search |
| `/product/<id>` | GET | Product detail with recommendations |
| `/api/search` | POST | Search API (JSON) |
| `/api/recommend/<id>` | GET | Get recommendations for product |
| `/api/stats` | GET | Dashboard statistics |

## 📸 Screenshots

### Home Page
- Hero section with search
- KPI cards showing total products, categories, avg rating
- Popular products grid

### Product Detail
- Product image and details
- Price with discount
- Similar products recommendations
- Related products from same category

### Dashboard
- User-specific recommendations
- Category distribution chart
- Rating distribution chart

## 📊 EDA Visualizations

The system generates comprehensive EDA reports:
- Category distribution
- Ratings distribution
- Correlation matrix
- Price analysis
- Popularity distribution
- Top products per category

All visualizations are saved in `artifacts/eda/`

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file:
```
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///database.db
DEBUG=True
```

## 📝 Requirements

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.4.2
scipy==1.11.4
scikit-surprise==1.1.4
nltk==3.8.1
flask==3.0.0
flask-login==0.6.3
flask-sqlalchemy==3.1.1
werkzeug==3.0.1
```

## 👨‍💻 Author

**Tarigonda Rajesh**
- GitHub: [@rajesh1835](https://github.com/rajesh1835)

## 📄 License

This project is licensed under the MIT License.

---

⭐ **Star this repository if you found it helpful!**
