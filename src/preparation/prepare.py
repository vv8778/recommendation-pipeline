import os
import yaml
import pandas as pd
import numpy as np

class DataPreparator:
    """
    Cleans, standardizes, filters, and performs chronological time-based train/test splitting.
    """
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = self.config["paths"]["processed_data_dir"]
        os.makedirs(self.processed_dir, exist_ok=True)

        self.time_split_ratio = float(self.config["preparation"].get("time_split_ratio", 0.80))
        self.min_user_interactions = int(self.config["preparation"].get("min_user_interactions", 3))
        self.min_item_interactions = int(self.config["preparation"].get("min_item_interactions", 2))

    def prepare(self, valid_users_df: pd.DataFrame, valid_items_df: pd.DataFrame, valid_interactions_df: pd.DataFrame):
        """
        Executes data preparation and time-based train/test split.
        """
        print("[Preparation] Standardizing data formats and performing chronological split...")

        # 1. Format Standardization
        interactions_clean = valid_interactions_df.copy()
        interactions_clean["timestamp"] = interactions_clean["timestamp"].astype(int)
        interactions_clean["interaction_datetime"] = pd.to_datetime(interactions_clean["timestamp"], unit="s")
        interactions_clean["rating"] = interactions_clean["rating"].astype(float)
        interactions_clean["user_id"] = interactions_clean["user_id"].astype(int)
        interactions_clean["item_id"] = interactions_clean["item_id"].astype(int)

        # Standardize Users
        users_clean = valid_users_df.copy()
        users_clean["user_id"] = users_clean["user_id"].astype(int)
        users_clean["gender"] = users_clean["gender"].astype(str).str.upper().str.strip()
        users_clean["occupation"] = users_clean["occupation"].astype(str).str.lower().str.strip()

        # Standardize Items
        items_clean = valid_items_df.copy()
        items_clean["item_id"] = items_clean["item_id"].astype(int)
        items_clean["title"] = items_clean["title"].astype(str).str.strip()
        items_clean["genres"] = items_clean["genres"].astype(str).str.strip()

        # 2. Activity Filtering (ensure sufficient interaction density)
        user_counts = interactions_clean["user_id"].value_counts()
        item_counts = interactions_clean["item_id"].value_counts()

        dense_users = user_counts[user_counts >= self.min_user_interactions].index
        dense_items = item_counts[item_counts >= self.min_item_interactions].index

        filtered_interactions = interactions_clean[
            interactions_clean["user_id"].isin(dense_users) &
            interactions_clean["item_id"].isin(dense_items)
        ].copy()

        # 3. Chronological Time-based Train/Test Split
        # Recommendation systems require temporal splitting to prevent future data leakage
        filtered_interactions = filtered_interactions.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

        train_records = []
        test_records = []

        # Split per user based on interaction timestamp
        for user_id, group in filtered_interactions.groupby("user_id"):
            n = len(group)
            if n <= 1:
                train_records.append(group)
            else:
                split_idx = int(np.ceil(n * self.time_split_ratio))
                # Ensure at least 1 record in train and 1 record in test if n >= 2
                split_idx = max(1, min(split_idx, n - 1))
                train_records.append(group.iloc[:split_idx])
                test_records.append(group.iloc[split_idx:])

        train_df = pd.concat(train_records, ignore_index=True)
        test_df = pd.concat(test_records, ignore_index=True)

        # 4. Save processed parquet datasets
        train_path = os.path.join(self.processed_dir, "train.parquet")
        test_path = os.path.join(self.processed_dir, "test.parquet")
        users_path = os.path.join(self.processed_dir, "users_clean.parquet")
        items_path = os.path.join(self.processed_dir, "items_clean.parquet")

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        users_clean.to_parquet(users_path, index=False)
        items_clean.to_parquet(items_path, index=False)

        print(f"[Preparation] Preparation and Time-Based Split complete:")
        print(f"  - Train Set: {len(train_df):,} interactions ({len(train_df)/len(filtered_interactions)*100:.1f}%) -> {train_path}")
        print(f"  - Test Set: {len(test_df):,} interactions ({len(test_df)/len(filtered_interactions)*100:.1f}%) -> {test_path}")
        print(f"  - Clean Users: {len(users_clean):,} -> {users_path}")
        print(f"  - Clean Items: {len(items_clean):,} -> {items_path}")

        return {
            "train_df": train_df,
            "test_df": test_df,
            "users_clean": users_clean,
            "items_clean": items_clean
        }

if __name__ == "__main__":
    from src.validation.validate import DataValidator
    validator = DataValidator()
    val_out = validator.validate()
    preparator = DataPreparator()
    preparator.prepare(val_out["valid_users_df"], val_out["valid_items_df"], val_out["valid_interactions_df"])
