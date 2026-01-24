import pandas as pd


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        # Always work on a deep copy
        self.df = df.copy()

    def drop_unused_columns(self):
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])
        return self

    def convert_numeric_columns(self):
        numeric_cols = [
            "ratings",
            "no_of_ratings",
            "discount_price",
            "actual_price"
        ]

        for col in numeric_cols:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
            )
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        return self

    def handle_missing_values(self):
        # ✅ NO inplace=True anywhere

        self.df["ratings"] = self.df["ratings"].fillna(
            self.df["ratings"].median()
        )

        self.df["discount_price"] = self.df["discount_price"].fillna(
            self.df["discount_price"].median()
        )

        self.df["actual_price"] = self.df["actual_price"].fillna(
            self.df["actual_price"].median()
        )

        self.df["no_of_ratings"] = (
            self.df["no_of_ratings"]
            .fillna(0)
            .astype(int)
        )

        return self

    def fix_price_logic(self):
        self.df = self.df[
            (self.df["discount_price"] <= self.df["actual_price"]) &
            (self.df["actual_price"] > 0)
        ]
        return self

    def get_clean_data(self):
        return self.df
