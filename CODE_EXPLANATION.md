# 📘 Product Recommendation System - Code Walkthrough Guide

This guide explains the entire project code "story," from data loading to the final recommendation. Use this to explain the system to instructors and teammates.

---

## 🚀 1. The Entry Point (`main.py`)
**"Where it all starts."**

The `main.py` file controls the entire pipeline. It calls different modules step-by-step.
- **Why?** Keeps the code organized. We don't write all logic in one file; `main.py` just orchestrates the work.
- **Key Flow:**
  1.  **Load Data**: Reads the raw CSV and text files.
  2.  **Combine Data**: Merges reviews into product data and creates dummy User/Product IDs.
  3.  **Clean Data**: Fixes missing values and invalid prices.
  4.  **Sample Data**: Takes a random 25k sample so training is fast.
  5.  **EDA**: Generates charts to understand the data.
  6.  **Model Training**: We train 4 models (Baseline, KNN, SVD, KNNWithMeans).
  7.  **Comparison**: The code automatically picks the best model (lowest RMSE).
  8.  **Save Artifacts**: The model and config are saved for later use.

---

## 🛠️ 2. Data Processing Layer (`src/components/`)
**"Preparing the ingredients before cooking."**

### A. Data Loading (`data_loader.py`)
- **Function:** `load_amazon_data()`, `load_reviews()`
- **Explanation:**
  - Safely opens the file paths.
  - Returns a Pandas DataFrame for the products and a list for reviews.
  - **Key Detail:** "We decoupled loading logic so if the file path changes, we only update this one file."

### B. Data Combining (`data_combiner.py`)
- **Function:** `generate_user_product_ids()`, `assign_reviews()`
- **Explanation:**
  - Our raw data didn't have user IDs, so we simulated them.
  - `generate_user_product_ids`: Randomly assigns `A00...` user IDs and `B00...` product IDs to create a "User-Item Interaction Matrix" structure.
  - `assign_reviews`: Randomly attaches the text reviews to products.

### C. Data Cleaning (`data_cleaner.py`)
- **Function:** `handle_missing_values()`, `fix_price_logic()`
- **Explanation:**
  - **Data Type Conversion:** Prices were strings (e.g., "₹499"); we strip the currency symbol and convert to float.
  - **Missing Values:** We fill missing ratings with the *median* (not mean, to avoid outliers).
  - **Logic Fix:** We remove rows where `Discount Price > Actual Price` because that's impossible.

### D. Sampling (`sampler.py`)
- **Function:** `sample(n=25000)`
- **Explanation:**
  - Training ML models on huge datasets takes too long for a prototype.
  - We take a random sample of 25,000 rows to prove the concept efficiently.

---

## 📊 3. Analysis Layer (`src/components/eda.py`)
**"Understanding our data."**

This class generates all the visualizations you see in the `artifacts/eda/` folder.
- **Key Methods:**
  - `ratings_distribution()`: Shows if users are generally happy or unhappy.
  - `price_analysis()`: Checks if our products are budget-friendly or expensive.
  - `correlation_matrix()`: Checks if price correlates with ratings (e.g., "Do expensive items get better ratings?").
- **Tip for presentation:** "We used automated EDA to generate insights instantly every time we run the pipeline."

---

## 🤖 4. Model Layer (`src/models/`)
**"The Brain of the system."**

We used the **Surprise Library** for collaborative filtering.

### A. The Setup
- We transform the Pandas DataFrame into a Surprise `Dataset` object (User, Item, Rating).
- We use **Cross-Validation (CV)** to ensure our results are robust.

### B. The Models
1.  **BaselineOnly (`baseline_model.py`)**:
    - A simple statistical model.
    - Used as a benchmark. "If our complex model can't beat this, it's useless."
2.  **KNNBasic (`knn_model.py`)**:
    - **Concept:** "Users who liked similar items in the past will like similar items in the future."
    - Uses **Cosine Similarity**.
3.  **SVD (`svd_model.py`)**:
    - **Concept:** Matrix Factorization. It finds hidden "latent factors" (features) that connect users and items.
    - Usually the industry standard, but...
4.  **KNNWithMeans (`knn_with_kmeans.py`) — THE WINNER 🏆**:
    - **Concept:** Same as KNNBasic, but it accounts for **User Bias**.
    - **Explanation:** "Some users are tough critics (never give 5 stars), some are generous (always give 5). `KNNWithMeans` subtracts the user's average rating to normalize this bias."
    - This is why it had the best RMSE (0.728).

### C. Hyperparameter Tuning (`knn_basic_gridsearch.py`)
- We didn't just guess the settings.
- We used `GridSearchCV` to try different combinations of neighbors (`k=20`, `k=40`) and similarity metrics (`cosine`, `pearson`).
- The system automatically found that `k=20` + `cosine` was the best combination.

---

## 🔍 5. Inference Layer (`src/inference/`)
**"Delivering the results."**

### Search & Recommendation (`search_recommendation.py`)
This file is how the web app (in future) would get answers.
1.  **KNN Recommender**:
    - Takes a `ProductId`.
    - Looks up the nearest neighbors in the trained model.
    - Returns the top 6 similar products.
2.  **TF-IDF Recommender (Content-Based Fallback)**:
    - **Problem:** "What if a new product has no ratings?" (The Cold Start Problem).
    - **Solution:** We analyze the text (Name + Category). If KNN fails, we recommend items with similar *text words* using TF-IDF.
3.  **Hybrid Approach**:
    - We combine both methods for robustness.

---

## 📝 Summary for Intro
"Our project is an end-to-end recommendation system pipeline. We prioritized **Modularity** (breaking code into small components) and **Automation** (the pipeline runs start-to-finish with one command). We compared 4 distinct algorithms and found that **KNNWithMeans** performed best because it effectively handles user rating bias."
