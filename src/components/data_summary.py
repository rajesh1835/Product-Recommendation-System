import pandas as pd
import numpy as np


class DataSummary:
    def __init__(self):
        self.records = []

    def add_stage(self, stage_name, df, description):
        """Add a DataFrame stage to the summary"""
        self.records.append({
            "Stage": stage_name,
            "Rows": f"{len(df):,}",
            "Columns": df.shape[1],
            "Memory (MB)": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
            "Missing %": f"{(df.isnull().sum().sum() / df.size * 100):.1f}%",
            "Description": description
        })

    def add_text_stage(self, stage_name, count, description):
        """Add a non-DataFrame stage (like text files)"""
        self.records.append({
            "Stage": stage_name,
            "Rows": f"{count:,}",
            "Columns": "-",
            "Memory (MB)": "-",
            "Missing %": "-",
            "Description": description
        })

    def get_summary_table(self):
        """Return summary as DataFrame"""
        return pd.DataFrame(self.records)

    def print_summary(self):
        """Print formatted summary table"""
        print("\n" + "=" * 80)
        print(" 📊 PIPELINE SUMMARY ")
        print("=" * 80)
        print(self.get_summary_table().to_string(index=False))
        print("=" * 80 + "\n")

    def get_detailed_stats(self, df, stage_name):
        """Get detailed statistics for a DataFrame"""
        print(f"\n📋 Detailed Stats for: {stage_name}")
        print("-" * 50)

        # Numeric columns stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n🔢 Numeric Columns ({len(numeric_cols)}):")
            for col in numeric_cols[:5]:  # Show first 5
                print(f"   • {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

        # Categorical columns stats
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print(f"\n🔤 Categorical Columns ({len(cat_cols)}):")
            for col in cat_cols[:5]:  # Show first 5
                print(f"   • {col}: {df[col].nunique()} unique values")

        print("-" * 50)
        return self

    def save_summary(self, filepath="artifacts/pipeline_summary.csv"):
        """Save summary to CSV"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.get_summary_table().to_csv(filepath, index=False)
        print(f"✔ Summary saved to: {filepath}")
        return self
