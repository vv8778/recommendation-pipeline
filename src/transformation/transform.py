import os
import json
import yaml
import pandas as pd
import numpy as np

class DataTransformer:
    """
    Computes derived analytical datasets:
    1. User Activity Summary (interaction volume, rating distribution, activity segment)
    2. Item Popularity Summary (Bayesian weighted average, popularity rank, tiers)
    3. User-Item Interaction Matrix (sparse user-item rating representation)
    """
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = self.config["paths"]["processed_data_dir"]
        os.makedirs(self.processed_dir, exist_ok=True)

        self.user_bins = self.config["features"].get("user_activity_bins", [0, 10, 50, 100, 1000])
        self.user_labels = self.config["features"].get("user_activity_labels", ["casual", "moderate", "active", "power"])
        self.item_bins = self.config["features"].get("item_popularity_bins", [0, 10, 50, 150, 10000])
        self.item_labels = self.config["features"].get("item_popularity_labels", ["niche", "developing", "popular", "blockbuster"])

    def transform(self, train_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame):
        """
        Generates derived datasets strictly using training interactions to avoid data leakage.
        """
        print("[Transformation] Computing derived analytical datasets...")

        # -----------------------------------------------------------------
        # 1. User Activity Summary
        # -----------------------------------------------------------------
        user_grp = train_df.groupby("user_id")
        user_summary = user_grp.agg(
            total_ratings_count=("rating", "count"),
            user_mean_rating=("rating", "mean"),
            user_rating_std=("rating", lambda x: x.std(ddof=0)),
            user_min_rating=("rating", "min"),
            user_max_rating=("rating", "max"),
            first_interaction_ts=("timestamp", "min"),
            last_interaction_ts=("timestamp", "max")
        ).reset_index()

        # Calculate tenure in days
        user_summary["user_tenure_days"] = np.maximum(
            0, (user_summary["last_interaction_ts"] - user_summary["first_interaction_ts"]) / 86400.0
        ).round(2)

        # Activity segment
        user_summary["activity_segment"] = pd.cut(
            user_summary["total_ratings_count"],
            bins=self.user_bins,
            labels=self.user_labels,
            include_lowest=True
        ).astype(str)

        # Merge with user demographics
        user_summary = user_summary.merge(users_df, on="user_id", how="left")

        # -----------------------------------------------------------------
        # 2. Item Popularity Summary with Bayesian Weighted Rating
        # -----------------------------------------------------------------
        item_grp = train_df.groupby("item_id")
        item_summary = item_grp.agg(
            total_ratings_count=("rating", "count"),
            item_mean_rating=("rating", "mean"),
            item_rating_std=("rating", lambda x: x.std(ddof=0))
        ).reset_index()

        # Bayesian weighted rating: WR = (v / (v + m)) * R + (m / (v + m)) * C
        C = float(train_df["rating"].mean()) # global mean
        m = 10.0 # minimum weight constant
        v = item_summary["total_ratings_count"]
        R = item_summary["item_mean_rating"]
        item_summary["bayesian_weighted_rating"] = ((v / (v + m)) * R + (m / (v + m)) * C).round(4)

        # Popularity Rank (1 = most popular)
        item_summary["popularity_rank"] = item_summary["total_ratings_count"].rank(ascending=False, method="min").astype(int)

        # Popularity Tier
        item_summary["popularity_tier"] = pd.cut(
            item_summary["total_ratings_count"],
            bins=self.item_bins,
            labels=self.item_labels,
            include_lowest=True
        ).astype(str)

        # Merge with item details
        item_summary = item_summary.merge(items_df, on="item_id", how="left")

        # -----------------------------------------------------------------
        # 3. User-Item Interaction Matrix
        # -----------------------------------------------------------------
        # Create user & item index maps
        unique_users = sorted(train_df["user_id"].unique())
        unique_items = sorted(train_df["item_id"].unique())

        user2idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        item2idx = {int(iid): idx for idx, iid in enumerate(unique_items)}

        indexed_interactions = train_df[["user_id", "item_id", "rating"]].copy()
        indexed_interactions["user_idx"] = indexed_interactions["user_id"].map(user2idx)
        indexed_interactions["item_idx"] = indexed_interactions["item_id"].map(item2idx)

        # Pivot sparse-style representation
        user_item_matrix = indexed_interactions[["user_id", "item_id", "rating", "user_idx", "item_idx"]].copy()

        # Save derived datasets
        user_summary_path = os.path.join(self.processed_dir, "user_activity_summary.parquet")
        item_summary_path = os.path.join(self.processed_dir, "item_popularity_summary.parquet")
        matrix_path = os.path.join(self.processed_dir, "user_item_matrix.parquet")
        mappings_path = os.path.join(self.processed_dir, "index_mappings.json")

        user_summary.to_parquet(user_summary_path, index=False)
        item_summary.to_parquet(item_summary_path, index=False)
        user_item_matrix.to_parquet(matrix_path, index=False)

        with open(mappings_path, "w") as f:
            json.dump({
                "user2idx": user2idx,
                "item2idx": item2idx,
                "num_users": len(unique_users),
                "num_items": len(unique_items),
                "matrix_sparsity_pct": round(100.0 * (1.0 - len(train_df) / (len(unique_users) * len(unique_items))), 2)
            }, f, indent=2)

        print(f"[Transformation] Derived datasets successfully generated:")
        print(f"  - User Activity Summary: {len(user_summary):,} users -> {user_summary_path}")
        print(f"  - Item Popularity Summary: {len(item_summary):,} items -> {item_summary_path}")
        print(f"  - User-Item Matrix: {len(user_item_matrix):,} entries -> {matrix_path}")
        print(f"  - Index Mappings saved to: {mappings_path}")

        return {
            "user_activity_summary": user_summary,
            "item_popularity_summary": item_summary,
            "user_item_matrix": user_item_matrix,
            "user2idx": user2idx,
            "item2idx": item2idx
        }

if __name__ == "__main__":
    from src.preparation.prepare import DataPreparator
    prep = DataPreparator()
    train_df = pd.read_parquet("data/processed/train.parquet")
    users_df = pd.read_parquet("data/processed/users_clean.parquet")
    items_df = pd.read_parquet("data/processed/items_clean.parquet")
    transformer = DataTransformer()
    transformer.transform(train_df, users_df, items_df)
