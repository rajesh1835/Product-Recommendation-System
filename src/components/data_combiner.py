import numpy as np
import pandas as pd


class DataCombiner:
    def __init__(self, amazon_df, reviews):
        self.amazon_df = amazon_df.copy()
        self.reviews = reviews

    def generate_user_product_ids(self, num_users=1000, seed=42):
        np.random.seed(seed)
        n = len(self.amazon_df)

        self.amazon_df["UserId"] = [
            f"A{str(i).zfill(9)}"
            for i in np.random.randint(1, num_users + 1, size=n)
        ]

        self.amazon_df["ProductId"] = [
            f"B{str(i).zfill(9)}" for i in range(n)
        ]

        return self

    def assign_reviews(self):
        self.amazon_df["Review"] = np.random.choice(
            self.reviews, size=len(self.amazon_df)
        )
        return self

    def get_combined_data(self):
        return self.amazon_df
