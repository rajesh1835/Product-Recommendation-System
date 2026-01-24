import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, base_path="data/raw"):
        self.base_path = Path(base_path)

    def load_amazon_data(
        self,
        filename="Amazon-Products.csv",
        verbose=True,
        show_head=True,
        top_n_categories=5
    ):
        path = self.base_path / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        print("\n[DATA LOADER] Loading Amazon Products Dataset...")
        df = pd.read_csv(path)

        if verbose:
            self._print_dataframe_summary(
                df,
                dataset_name="Amazon Products Dataset",
                show_head=show_head,
                top_n_categories=top_n_categories
            )

        print("[DATA LOADER] Amazon Products Dataset loaded successfully\n")
        return df

    def load_reviews(self, filename="reviews.txt", verbose=True):
        path = self.base_path / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        print("\n[DATA LOADER] Loading Reviews Dataset...")
        with open(path, "r", encoding="utf-8") as f:
            reviews = [line.strip() for line in f if line.strip()]

        if verbose:
            print("\n--- Reviews Dataset Summary ---")
            print(f"Total Reviews: {len(reviews)}")
            print("Sample Reviews:")
            for i, review in enumerate(reviews[:3], start=1):
                print(f"{i}. {review[:120]}...")
            print("-------------------------------\n")

        print("[DATA LOADER] Reviews Dataset loaded successfully\n")
        return reviews

    @staticmethod
    def _print_dataframe_summary(
        df: pd.DataFrame,
        dataset_name: str,
        show_head: bool,
        top_n_categories: int
    ):
        print(f"\n--- {dataset_name} Summary ---")
        print(f"Shape: {df.shape}")

        print("\nColumns:")
        print(list(df.columns))

        print("\nInfo:")
        df.info()

        print("\nStatistical Summary (numeric columns):")
        print(df.describe())

        if show_head:
            print("\nFirst 5 Rows (Head):")
            print(df.head())

        # Category summaries (if present)
        if "main_category" in df.columns:
            print(f"\nTop {top_n_categories} Main Categories:")
            print(
                df["main_category"]
                .value_counts()
                .head(top_n_categories)
            )

        if "sub_category" in df.columns:
            print(f"\nTop {top_n_categories} Sub-Categories:")
            print(
                df["sub_category"]
                .value_counts()
                .head(top_n_categories)
            )

        print("\n-------------------------------\n")
