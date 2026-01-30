# 🤖 SmartPic: Bridging User Interest and Product Content with Hybrid Machine Learning

An industrial-grade E-Commerce recommendation engine built with Python and Flask. This system delivers high-precision suggestions by merging **Collaborative Filtering** (User Behavior) and **Content-Based Filtering** (Product Metadata) into a powerful **Hybrid AI** model.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![scikit-surprise](https://img.shields.io/badge/Surprise-1.1.4-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Dataset](#-dataset)
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
- 🔍 **Product Search** - Fast search with MySQL and full-text matching
- 🤖 **Hybrid AI Recommendations** - Combination of KNN collaborative filtering and TF-IDF content similarity
- 🎯 **Smart Similarity** - Item-to-item suggestions based on both feature similarity and user preference history
- 📊 **KPI Dashboard** - Real-time statistics and charts
- 🌙 **Dark/Light Theme** - Modern UI with theme toggle
- 📱 **Responsive Design** - Works on desktop and mobile
- 🤖 **ML-Powered** - Multi-model comparison (KNN, SVD, Baseline) using Scikit-Surprise library
- ⚡ **Real-time Engine** - Dynamic hybrid scoring with α-weighted combination logic

## 🎥 Demo

> **Watch the system in action:** [Click here to watch demo video](https://1drv.ms/v/c/7a629c80013a5764/IQDtFaKGi78PR4KGZENdx6-hAXND8Lhn-MJMlOmcd2C8ZHI)

## 📂 Dataset

The system utilizes a rich dataset of Amazon products and user reviews to generate recommendations.

- **Amazon Products**: [Download Here](https://drive.google.com/file/d/1HvrCPVh_BWykakKFcBdWgyznDwUfOtv1/view?usp=sharing)
- **User Reviews**: [Download Here](https://drive.google.com/file/d/1HvrCPVh_BWykakKFcBdWgyznDwUfOtv1/view?usp=drive_link)

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python, Flask, Flask-SQLAlchemy, MySQL |
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **Database** | MySQL (with SQLALchemy) |
| **ML/Data** | Pandas, NumPy, Scikit-learn, Scikit-Surprise |
| **Visualization** | Matplotlib, Seaborn |

## 📁 Project Structure

```
Product Recommendation System/
├── app/                          # Flask Web Application
│   ├── app.py                    # Main Flask application
│   ├── static/                   # Static assets (CSS, JS, Images)
│   └── templates/                # HTML templates (Dashboard, Search, etc.)
│
├── src/                          # Source Code
│   ├── components/               # Data processing modules
│   │   └── database.py           # SQLALchemy Models & DB Init
│   ├── models/                   # ML models (SVD, KNN, etc.)
│   ├── inference/                # Prediction & Recommendation logic
│   └── testing/                  # A/B testing modules
│
├── scripts/                      # Utility Scripts
│   ├── migrate_csv_to_db.py      # Import CSV data to MySQL
│   └── reset_db.py               # Reset and Reinitialize Database
│
├── artifacts/                    # Generated ML models & plots
├── data/                         # Datasets (Raw & Processed)
├── main.py                       # ML Pipeline entry point
├── config.py                     # Flask Configuration
└── requirements.txt              # Project dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- MySQL Server
- Optional: Virtual Environment (recommended)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/rajesh1835/Product-Recommendation-System.git
   cd Product-Recommendation-System
   ```

2. **Setup Environment (Choose one: venv or Conda)**

   **Option A: Virtual Environment (venv)**
   ```bash
   python -m venv venv
   # Windows
   source venv/Scripts/activate
   ```

   **Option B: Conda Environment**
   ```bash
   conda create -n product_rec python=3.10 -y
   conda activate product_rec
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your_secret_key
   DB_USER=root
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_NAME=product_rec_db
   ```

5. **Initialize Database**
   Run the utility script to create tables and import data:
   ```bash
   python scripts/migrate_csv_to_db.py
   ```

6. **Start the web application**
   ```bash
   cd app
   python app.py
   ```

7. **Open in browser**
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

## 📄 Research Paper

A comprehensive IEEE-standard research paper documenting the system architecture, methodology, and experimental results is available:

- [View Research Paper (HTML)](presentation/Research/IEEE_Research_Paper.html)
- [View Research Paper (PDF Source)](presentation/Research/IEEE_Research_Paper.tex)
- [View Research Paper (Markdown)](presentation/Research/IEEE_Research_Paper.md)

## 🎤 Presentation

A professional slide deck summarizing the project, methodology, and results is available:

- [View Final Presentation (Slides)](presentation/Final_Presentation.html)
- [View System Diagrams](presentation/diagrams/system_diagrams.html)

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
