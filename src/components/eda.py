import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


class EDA:
    def __init__(self, df, output_dir="artifacts/eda"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # 1. Missing Values Analysis
    # -------------------------------------------------
    def missing_values_analysis(self, save=True):
        """Visualize missing values pattern"""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            print("✅ No missing values in the dataset!")
            return self

        plt.figure(figsize=(10, 6))
        missing.plot(kind="bar", color="coral", edgecolor="black")
        plt.title("Missing Values by Column", fontsize=14, fontweight="bold")
        plt.xlabel("Columns")
        plt.ylabel("Number of Missing Values")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/missing_values.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/missing_values.png")
        plt.show()
        return self

    # -------------------------------------------------
    # 2. Category Distribution (Enhanced Bar Chart)
    # -------------------------------------------------
    def category_distribution(self, column="main_category", top_n=10, save=True):
        """Bar chart for category distribution"""
        if column not in self.df.columns:
            print(f"⚠ Column '{column}' not found!")
            return self

        plt.figure(figsize=(12, 6))

        value_counts = self.df[column].value_counts().head(top_n)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
        bars = plt.barh(value_counts.index, value_counts.values, color=colors, edgecolor="black")

        # Add value labels
        for bar, val in zip(bars, value_counts.values):
            plt.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{val:,}', va='center', fontsize=10)

        plt.xlabel("Count", fontsize=12)
        plt.ylabel(column.replace("_", " ").title(), fontsize=12)
        plt.title(f"Top {top_n} {column.replace('_', ' ').title()} Distribution",
                  fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/category_distribution.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/category_distribution.png")
        plt.show()
        return self

    # -------------------------------------------------
    # 3. Ratings Distribution (Histogram + KDE)
    # -------------------------------------------------
    def ratings_distribution(self, column="ratings", save=True):
        """Histogram with KDE for ratings"""
        if column not in self.df.columns:
            print(f"⚠ Column '{column}' not found!")
            return self

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram with KDE
        sns.histplot(self.df[column].dropna(), bins=20, kde=True,
                     color="steelblue", edgecolor="black", ax=axes[0])
        axes[0].set_title(f"{column.title()} Distribution", fontsize=14, fontweight="bold")
        axes[0].set_xlabel(column.title())
        axes[0].set_ylabel("Frequency")

        # Add mean & median lines
        mean_val = self.df[column].mean()
        median_val = self.df[column].median()
        axes[0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[0].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        axes[0].legend()

        # Box plot
        sns.boxplot(x=self.df[column].dropna(), color="lightblue", ax=axes[1])
        axes[1].set_title(f"{column.title()} Box Plot", fontsize=14, fontweight="bold")
        axes[1].set_xlabel(column.title())

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/ratings_distribution.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/ratings_distribution.png")
        plt.show()
        return self

    # -------------------------------------------------
    # 4. Correlation Matrix Heatmap
    # -------------------------------------------------
    def correlation_matrix(self, save=True):
        """Generate correlation heatmap for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            print("⚠ Not enough numeric columns for correlation!")
            return self

        corr = numeric_df.corr()

        plt.figure(figsize=(10, 8))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/correlation_matrix.png")
        plt.show()

        # Print high correlations
        print("\n📊 High Correlations (|r| > 0.5):")
        print("-" * 40)
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.5:
                    print(f"   {corr.columns[i]} ↔ {corr.columns[j]}: {corr.iloc[i, j]:.3f}")

        return self

    # -------------------------------------------------
    # 5. Price Analysis
    # -------------------------------------------------
    def price_analysis(self, save=True):
        """Analyze price columns"""
        if "actual_price" not in self.df.columns or "discount_price" not in self.df.columns:
            print("⚠ Price columns not found!")
            return self

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Actual vs Discount Price
        axes[0].scatter(
            self.df["actual_price"],
            self.df["discount_price"],
            alpha=0.3,
            c="steelblue",
            s=10
        )
        axes[0].plot([0, self.df["actual_price"].max()],
                     [0, self.df["actual_price"].max()],
                     'r--', label="No Discount Line")
        axes[0].set_xlabel("Actual Price")
        axes[0].set_ylabel("Discount Price")
        axes[0].set_title("Actual vs Discount Price", fontweight="bold")
        axes[0].legend()

        # Discount percentage distribution
        discount_pct = ((self.df["actual_price"] - self.df["discount_price"]) / 
                        self.df["actual_price"] * 100).clip(0, 100)
        sns.histplot(discount_pct.dropna(), bins=30, kde=True, color="coral", ax=axes[1])
        axes[1].set_xlabel("Discount %")
        axes[1].set_title("Discount Percentage Distribution", fontweight="bold")

        # Price by category
        if "main_category" in self.df.columns:
            top_cats = self.df["main_category"].value_counts().head(8).index
            cat_prices = self.df[self.df["main_category"].isin(top_cats)]
            sns.boxplot(data=cat_prices, x="main_category", y="actual_price", ax=axes[2])
            axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
            axes[2].set_title("Price by Category", fontweight="bold")

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/price_analysis.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/price_analysis.png")
        plt.show()
        return self

    # -------------------------------------------------
    # 6. Popularity Distribution (Log Scale)
    # -------------------------------------------------
    def popularity_distribution(self, column="no_of_ratings", save=True):
        """Analyze popularity/number of ratings"""
        if column not in self.df.columns:
            print(f"⚠ Column '{column}' not found!")
            return self

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Log-transformed histogram
        log_values = np.log1p(self.df[column].dropna())
        sns.histplot(log_values, bins=30, kde=True, color="teal", ax=axes[0])
        axes[0].set_xlabel(f"Log({column})")
        axes[0].set_title(f"Log-Transformed {column.replace('_', ' ').title()}",
                          fontweight="bold")

        # Top 10 most popular
        if "name" in self.df.columns:
            top_products = self.df.nlargest(10, column)[["name", column]]
            top_products["name"] = top_products["name"].str[:40] + "..."
            axes[1].barh(top_products["name"], top_products[column], color="coral")
            axes[1].set_xlabel(column.replace("_", " ").title())
            axes[1].set_title("Top 10 Most Popular Products", fontweight="bold")
            axes[1].invert_yaxis()

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/popularity_distribution.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/popularity_distribution.png")
        plt.show()
        return self

    # -------------------------------------------------
    # 7. Top Product per Category
    # -------------------------------------------------
    def top_product_per_category(self, metric="no_of_ratings", save=True):
        """Find top product in each category"""
        if "main_category" not in self.df.columns:
            print("⚠ 'main_category' column not found!")
            return self

        top = (
            self.df[["main_category", "name", metric]]
            .sort_values(metric, ascending=False)
            .groupby("main_category")
            .head(1)
            .head(15)
        )

        plt.figure(figsize=(14, 8))
        labels = top["main_category"].str[:20] + " | " + top["name"].str[:25]

        colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(top)))
        plt.barh(labels, top[metric], color=colors, edgecolor="black")

        plt.xlabel(metric.replace("_", " ").title())
        plt.title("Top Product per Category", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/top_per_category.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/top_per_category.png")
        plt.show()
        return self

    # -------------------------------------------------
    # 8. Ratings vs Popularity Scatter
    # -------------------------------------------------
    def ratings_vs_popularity(self, save=True):
        """Scatter plot of ratings vs number of ratings"""
        if "ratings" not in self.df.columns or "no_of_ratings" not in self.df.columns:
            print("⚠ Required columns not found!")
            return self

        plt.figure(figsize=(10, 6))

        scatter = plt.scatter(
            self.df["ratings"],
            np.log1p(self.df["no_of_ratings"]),
            alpha=0.3,
            c=self.df["ratings"],
            cmap="RdYlGn",
            s=20
        )

        plt.colorbar(scatter, label="Rating")
        plt.xlabel("Rating (1-5)", fontsize=12)
        plt.ylabel("Log(Number of Ratings)", fontsize=12)
        plt.title("Ratings vs Popularity", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/ratings_vs_popularity.png", dpi=150)
            print(f"✔ Saved: {self.output_dir}/ratings_vs_popularity.png")
        plt.show()
        return self

    # -------------------------------------------------
    # 9. Storytelling Insights
    # -------------------------------------------------
    def generate_insights(self):
        """Generate key business insights from data"""
        print("\n" + "=" * 70)
        print(" 📖 DATA STORYTELLING - KEY INSIGHTS ")
        print("=" * 70)

        # Insight 1: Dataset Overview
        print("\n🔹 INSIGHT 1: Dataset Overview")
        print("-" * 50)
        print(f"   • Total Products: {len(self.df):,}")
        print(f"   • Categories: {self.df['main_category'].nunique() if 'main_category' in self.df.columns else 'N/A'}")
        print(f"   • Sub-categories: {self.df['sub_category'].nunique() if 'sub_category' in self.df.columns else 'N/A'}")

        # Insight 2: Rating Distribution Story
        if "ratings" in self.df.columns:
            print("\n🔹 INSIGHT 2: Customer Satisfaction Story")
            print("-" * 50)
            avg_rating = self.df["ratings"].mean()
            high_rated = (self.df["ratings"] >= 4).sum()
            low_rated = (self.df["ratings"] <= 2).sum()
            print(f"   • Average Rating: {avg_rating:.2f}/5 ⭐")
            print(f"   • High-rated products (≥4): {high_rated:,} ({high_rated/len(self.df)*100:.1f}%)")
            print(f"   • Low-rated products (≤2): {low_rated:,} ({low_rated/len(self.df)*100:.1f}%)")
            
            if avg_rating >= 4:
                print("   → 📈 STORY: Most products have high ratings, indicating good quality")
            else:
                print("   → ⚠️ STORY: Average ratings suggest room for quality improvement")

        # Insight 3: Price Story
        if "actual_price" in self.df.columns and "discount_price" in self.df.columns:
            print("\n🔹 INSIGHT 3: Pricing Strategy Story")
            print("-" * 50)
            avg_discount = ((self.df["actual_price"] - self.df["discount_price"]) / 
                           self.df["actual_price"] * 100).mean()
            max_price = self.df["actual_price"].max()
            min_price = self.df["actual_price"].min()
            print(f"   • Price Range: ₹{min_price:.0f} - ₹{max_price:.0f}")
            print(f"   • Average Discount: {avg_discount:.1f}%")
            
            if avg_discount > 30:
                print("   → 💰 STORY: Heavy discounting strategy - price-sensitive market")
            else:
                print("   → 💎 STORY: Moderate discounts - focus on value perception")

        # Insight 4: Popularity Story
        if "no_of_ratings" in self.df.columns:
            print("\n🔹 INSIGHT 4: Product Popularity Story")
            print("-" * 50)
            popular_threshold = self.df["no_of_ratings"].quantile(0.9)
            popular_products = (self.df["no_of_ratings"] >= popular_threshold).sum()
            unpopular = (self.df["no_of_ratings"] == 0).sum()
            print(f"   • Top 10% popular products: {popular_products:,} (>{popular_threshold:.0f} ratings)")
            print(f"   • Zero-rating products: {unpopular:,}")
            print("   → 📊 STORY: Long-tail distribution - few products dominate sales")

        # Insight 5: Category Story
        if "main_category" in self.df.columns:
            print("\n🔹 INSIGHT 5: Category Performance Story")
            print("-" * 50)
            top_cat = self.df["main_category"].value_counts().head(3)
            for i, (cat, count) in enumerate(top_cat.items(), 1):
                print(f"   {i}. {cat}: {count:,} products ({count/len(self.df)*100:.1f}%)")
            print("   → 🏆 STORY: Focus on top categories for maximum impact")

        print("\n" + "=" * 70 + "\n")
        return self

    # -------------------------------------------------
    # 10. Feature Selection for Model Building
    # -------------------------------------------------
    def feature_selection_analysis(self, target="ratings", save=True):
        """Analyze and recommend features for model building"""
        print("\n" + "=" * 70)
        print(" 🎯 FEATURE SELECTION FOR MODEL BUILDING ")
        print("=" * 70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        print("\n📋 AVAILABLE FEATURES:")
        print("-" * 50)
        print(f"   Numeric ({len(numeric_cols)}): {numeric_cols}")
        print(f"   Categorical ({len(cat_cols)}): {cat_cols[:5]}...")

        # Correlation with target
        if target in self.df.columns:
            print(f"\n📊 CORRELATION WITH TARGET ({target}):")
            print("-" * 50)
            correlations = self.df[numeric_cols].corr()[target].drop(target).abs().sort_values(ascending=False)
            for col, corr in correlations.head(5).items():
                strength = "Strong" if corr > 0.5 else "Moderate" if corr > 0.3 else "Weak"
                print(f"   • {col}: {corr:.3f} ({strength})")

            # Plot feature importance
            plt.figure(figsize=(10, 6))
            colors = ['green' if c > 0.3 else 'orange' if c > 0.1 else 'red' for c in correlations.values]
            correlations.plot(kind='barh', color=colors, edgecolor='black')
            plt.xlabel("Absolute Correlation")
            plt.title(f"Feature Correlation with {target}", fontweight="bold")
            plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Strong (>0.3)')
            plt.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate (>0.1)')
            plt.legend()
            plt.tight_layout()
            
            if save:
                plt.savefig(f"{self.output_dir}/feature_correlation.png", dpi=150)
                print(f"\n✔ Saved: {self.output_dir}/feature_correlation.png")
            plt.show()

        # Recommended features for recommendation system
        print("\n🎯 RECOMMENDED FEATURES FOR RECOMMENDATION SYSTEM:")
        print("-" * 50)
        print("   📌 Essential Features (Must Have):")
        essential = ["UserId", "ProductId", "ratings"]
        for feat in essential:
            status = "✅" if feat in self.df.columns else "❌"
            print(f"      {status} {feat}")

        print("\n   📌 Content-Based Features:")
        content_features = ["name", "main_category", "sub_category", "Review"]
        for feat in content_features:
            status = "✅" if feat in self.df.columns else "❌"
            print(f"      {status} {feat}")

        print("\n   📌 Collaborative Filtering Features:")
        cf_features = ["ratings", "no_of_ratings"]
        for feat in cf_features:
            status = "✅" if feat in self.df.columns else "❌"
            print(f"      {status} {feat}")

        print("\n   📌 Price-Based Features (for business rules):")
        price_features = ["actual_price", "discount_price"]
        for feat in price_features:
            status = "✅" if feat in self.df.columns else "❌"
            print(f"      {status} {feat}")

        # Feature engineering suggestions
        print("\n💡 FEATURE ENGINEERING SUGGESTIONS:")
        print("-" * 50)
        print("   1. discount_percentage = (actual_price - discount_price) / actual_price")
        print("   2. popularity_score = log(no_of_ratings + 1)")
        print("   3. price_category = binned(actual_price) → Low/Medium/High")
        print("   4. rating_category = binned(ratings) → Poor/Average/Good/Excellent")
        print("   5. review_length = len(Review)")

        print("\n" + "=" * 70 + "\n")
        return self

    # -------------------------------------------------
    # 11. Feature Engineering (Create New Features)
    # -------------------------------------------------
    def create_features(self):
        """Create new engineered features"""
        print("\n🔧 CREATING ENGINEERED FEATURES...")
        print("-" * 50)

        new_features = []

        # 1. Discount Percentage
        if "actual_price" in self.df.columns and "discount_price" in self.df.columns:
            self.df["discount_pct"] = ((self.df["actual_price"] - self.df["discount_price"]) / 
                                        self.df["actual_price"] * 100).clip(0, 100)
            new_features.append("discount_pct")

        # 2. Popularity Score (log-transformed)
        if "no_of_ratings" in self.df.columns:
            self.df["popularity_score"] = np.log1p(self.df["no_of_ratings"])
            new_features.append("popularity_score")

        # 3. Price Category
        if "actual_price" in self.df.columns:
            self.df["price_category"] = pd.cut(
                self.df["actual_price"],
                bins=[0, 500, 2000, 10000, float('inf')],
                labels=["Budget", "Mid-Range", "Premium", "Luxury"]
            )
            new_features.append("price_category")

        # 4. Rating Category
        if "ratings" in self.df.columns:
            self.df["rating_category"] = pd.cut(
                self.df["ratings"],
                bins=[0, 2, 3, 4, 5],
                labels=["Poor", "Average", "Good", "Excellent"]
            )
            new_features.append("rating_category")

        # 5. Review Length
        if "Review" in self.df.columns:
            self.df["review_length"] = self.df["Review"].astype(str).str.len()
            new_features.append("review_length")

        print(f"   ✅ Created {len(new_features)} new features: {new_features}")
        print("-" * 50 + "\n")
        return self

    # -------------------------------------------------
    # 12. Run All EDA (with storytelling)
    # -------------------------------------------------
    def run_all(self, save=True):
        """Run complete EDA analysis with storytelling"""
        print("\n" + "=" * 60)
        print(" 🔍 RUNNING COMPLETE EDA ANALYSIS ")
        print("=" * 60 + "\n")

        self.missing_values_analysis(save=save)
        self.category_distribution(save=save)
        self.ratings_distribution(save=save)
        self.correlation_matrix(save=save)
        self.price_analysis(save=save)
        self.popularity_distribution(save=save)
        self.top_product_per_category(save=save)
        self.ratings_vs_popularity(save=save)
        
        # Storytelling & Feature Selection
        self.generate_insights()
        self.feature_selection_analysis(save=save)

        print("\n" + "=" * 60)
        print(" ✅ EDA COMPLETE! ")
        print(f" 📁 All visualizations saved to: {self.output_dir}")
        print("=" * 60 + "\n")

        return self


# -------------------------------------------------
# STANDALONE USAGE
# -------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/processed/amazon_rec.csv")
    eda = EDA(df)
    eda.run_all()
