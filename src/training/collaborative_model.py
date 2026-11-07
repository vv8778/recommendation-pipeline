import numpy as np
import pandas as pd

class MatrixFactorizationRecommender:
    """
    Personalized Collaborative Filtering Recommender using Latent Factor Matrix Factorization (FunkSVD).
    Model: r_hat(u, i) = mu + b_u + b_i + p_u . q_i
    Trained via Stochastic Gradient Descent (SGD) with L2 regularization.
    """
    def __init__(self, n_factors: int = 20, learning_rate: float = 0.005,
                 regularization: float = 0.02, n_epochs: int = 20, random_seed: int = 42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_seed = random_seed

        self.mu = 0.0
        self.user_bias = {}
        self.item_bias = {}
        self.user_factors = {}
        self.item_factors = {}
        self.user_training_items = {}
        self.catalog_items = []

    def fit(self, train_df: pd.DataFrame):
        """Fits Matrix Factorization parameters using Stochastic Gradient Descent."""
        np.random.seed(self.random_seed)

        self.mu = float(train_df["rating"].mean())
        users = train_df["user_id"].unique()
        items = train_df["item_id"].unique()
        self.catalog_items = list(items)

        # Track items seen in training per user
        self.user_training_items = (
            train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        # Initialize biases to 0
        self.user_bias = {u: 0.0 for u in users}
        self.item_bias = {i: 0.0 for i in items}

        # Initialize latent factor vectors with small random normal values
        scale = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = {u: np.random.normal(0, scale, self.n_factors) for u in users}
        self.item_factors = {i: np.random.normal(0, scale, self.n_factors) for i in items}

        # Pre-extract arrays for fast iteration
        u_arr = train_df["user_id"].values
        i_arr = train_df["item_id"].values
        r_arr = train_df["rating"].values
        n_samples = len(train_df)

        lr = self.learning_rate
        reg = self.regularization

        print(f"[Training] Fitting Matrix Factorization (factors={self.n_factors}, epochs={self.n_epochs}, lr={lr}, reg={reg})...")

        for epoch in range(1, self.n_epochs + 1):
            indices = np.random.permutation(n_samples)
            loss = 0.0

            for idx in indices:
                u = u_arr[idx]
                i = i_arr[idx]
                r = r_arr[idx]

                pu = self.user_factors[u]
                qi = self.item_factors[i]
                bu = self.user_bias[u]
                bi = self.item_bias[i]

                pred = self.mu + bu + bi + np.dot(pu, qi)
                err = r - pred
                loss += err ** 2

                # Gradient updates with L2 regularization
                self.user_bias[u] += lr * (err - reg * bu)
                self.item_bias[i] += lr * (err - reg * bi)
                self.user_factors[u] += lr * (err * qi - reg * pu)
                self.item_factors[i] += lr * (err * pu - reg * qi)

            rmse_epoch = np.sqrt(loss / n_samples)
            if epoch % 5 == 0 or epoch == self.n_epochs:
                print(f"  - Epoch {epoch:02d}/{self.n_epochs:02d} | Train RMSE: {rmse_epoch:.4f}")

        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predicts explicit rating r_hat(u, i) with graceful cold-start fallback.
        Clamped to [1.0, 5.0].
        """
        has_user = user_id in self.user_factors
        has_item = item_id in self.item_factors

        if has_user and has_item:
            pred = (
                self.mu +
                self.user_bias[user_id] +
                self.item_bias[item_id] +
                np.dot(self.user_factors[user_id], self.item_factors[item_id])
            )
        elif has_user:
            pred = self.mu + self.user_bias[user_id]
        elif has_item:
            pred = self.mu + self.item_bias[item_id]
        else:
            pred = self.mu

        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, top_k: int = 10, all_candidate_items: list = None) -> list:
        """
        Generates personalized Top-K recommendations for user_id.
        Excludes items already seen in training.
        """
        candidate_pool = all_candidate_items if all_candidate_items is not None else self.catalog_items
        seen_items = self.user_training_items.get(user_id, set())

        # Unseen candidate items
        unseen = [i for i in candidate_pool if i not in seen_items]
        if not unseen:
            unseen = candidate_pool

        has_user = user_id in self.user_factors
        if not has_user:
            # Cold-start user: rank items by item bias / popularity
            scored = [(i, self.mu + self.item_bias.get(i, 0.0)) for i in unseen]
        else:
            pu = self.user_factors[user_id]
            bu = self.user_bias[user_id]
            scored = [
                (i, self.mu + bu + self.item_bias.get(i, 0.0) + np.dot(pu, self.item_factors[i]))
                if i in self.item_factors else (i, self.mu + bu)
                for i in unseen
            ]

        # Sort descending by predicted score
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, score in scored[:top_k]]
