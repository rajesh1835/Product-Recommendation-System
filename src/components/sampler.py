class DataSampler:
    def __init__(self, df):
        self.df = df

    def sample(self, n=25000, seed=42):
        return (
            self.df
            .sample(n=n, random_state=seed)
            .reset_index(drop=True)
        )
