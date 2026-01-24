import pandas as pd


class DataQualityReport:
    @staticmethod
    def compare(before_df: pd.DataFrame, after_df: pd.DataFrame):
        print("\n================ DATA QUALITY REPORT ================\n")

        # -----------------------------
        # 1. Data Types Comparison
        # -----------------------------
        print("▶ DATA TYPES (Before → After)\n")

        dtype_comparison = pd.DataFrame({
            "Before Cleaning": before_df.dtypes,
            "After Cleaning": after_df.dtypes
        })

        print(dtype_comparison)
        print()

        # -----------------------------
        # 2. Null Values Comparison
        # -----------------------------
        print("▶ NULL VALUES (Before → After)\n")

        null_comparison = pd.DataFrame({
            "Before Cleaning": before_df.isnull().sum(),
            "After Cleaning": after_df.isnull().sum()
        })

        print(null_comparison)
        print()

        # -----------------------------
        # 3. Total Null Reduction
        # -----------------------------
        total_before = before_df.isnull().sum().sum()
        total_after = after_df.isnull().sum().sum()

        print("▶ TOTAL NULL VALUES")
        print(f"Before Cleaning : {total_before}")
        print(f"After Cleaning  : {total_after}")
        print(f"Reduced         : {total_before - total_after}")

        print("\n====================================================\n")
