import os
import random
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml

class DataIngestor:
    """
    Ingests user, item, and interaction data from either public MovieLens or
    generates a realistic high-fidelity MovieLens-compatible benchmark dataset.
    """
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = self.config["paths"]["raw_data_dir"]
        os.makedirs(self.raw_dir, exist_ok=True)
        self.random_seed = self.config["pipeline"].get("random_seed", 42)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def generate_synthetic_movielens(self, num_users: int = 1000, num_items: int = 500,
                                     num_interactions: int = 50000, inject_noise: bool = True):
        """
        Generates realistic synthetic MovieLens-style users, items, and interactions.
        Emulates power-law user activity, zipfian item popularity, and realistic genre distributions.
        """
        print(f"[Ingestion] Generating synthetic data: {num_users} users, {num_items} items, {num_interactions} interactions...")

        # 1. Users
        genders = ["M", "F"]
        occupations = [
            "engineer", "educator", "student", "artist", "writer",
            "executive", "healthcare", "scientist", "programmer", "other"
        ]
        user_ids = list(range(1, num_users + 1))
        users_df = pd.DataFrame({
            "user_id": user_ids,
            "age": np.random.randint(18, 65, size=num_users),
            "gender": np.random.choice(genders, size=num_users, p=[0.55, 0.45]),
            "occupation": np.random.choice(occupations, size=num_users),
            "zip_code": [f"{np.random.randint(10000, 99999)}" for _ in range(num_users)]
        })

        # 2. Items
        all_genres = [
            "Action", "Adventure", "Animation", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        item_ids = list(range(1, num_items + 1))
        item_titles = [f"Movie_{i:04d}" for i in item_ids]
        item_genres = [
            "|".join(np.random.choice(all_genres, size=np.random.randint(1, 4), replace=False))
            for _ in range(num_items)
        ]
        items_df = pd.DataFrame({
            "item_id": item_ids,
            "title": item_titles,
            "genres": item_genres,
            "release_year": np.random.randint(1980, 2024, size=num_items)
        })

        # 3. Interactions with Power-Law / Zipf distribution
        # Active users interact much more; popular items get more ratings
        user_weights = np.random.pareto(a=1.5, size=num_users) + 1.0
        user_weights /= user_weights.sum()

        item_weights = np.random.pareto(a=1.2, size=num_items) + 1.0
        item_weights /= item_weights.sum()

        sampled_users = np.random.choice(user_ids, size=num_interactions, p=user_weights)
        sampled_items = np.random.choice(item_ids, size=num_interactions, p=item_weights)

        # Rating distribution centered around 3.5 (standard MovieLens distribution)
        base_ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=num_interactions, p=[0.06, 0.11, 0.27, 0.34, 0.22])

        # Timestamps over a 2-year span ending currently
        end_time = int(datetime(2025, 12, 31).timestamp())
        start_time = end_time - (730 * 24 * 3600)
        timestamps = np.random.randint(start_time, end_time, size=num_interactions)

        interactions_df = pd.DataFrame({
            "user_id": sampled_users,
            "item_id": sampled_items,
            "rating": base_ratings,
            "timestamp": timestamps
        })

        # Drop exact duplicate interactions initially so we control noise
        interactions_df = interactions_df.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

        # Deliberately inject anomalies if configured, so validation layer can quarantine them
        if inject_noise:
            noise_frac = self.config["ingestion"].get("noise_fraction", 0.02)
            n_noise = int(len(interactions_df) * noise_frac)
            print(f"[Ingestion] Injecting {n_noise} intentional anomalous records for validation quarantine testing...")

            noise_indices = np.random.choice(len(interactions_df), size=n_noise, replace=False)
            
            # Anomaly types:
            # 1. Invalid ratings (> 5.0 or < 1.0)
            # 2. Future timestamps
            # 3. Orphaned user_id (not in users table)
            # 4. Orphaned item_id (not in items table)
            # 5. Missing values
            chunk = n_noise // 5
            
            # Invalid ratings
            interactions_df.loc[noise_indices[:chunk], "rating"] = 9.5
            # Future timestamps (year 2035)
            interactions_df.loc[noise_indices[chunk:2*chunk], "timestamp"] = int(datetime(2035, 1, 1).timestamp())
            # Orphaned user_ids
            interactions_df.loc[noise_indices[2*chunk:3*chunk], "user_id"] = 999999
            # Orphaned item_ids
            interactions_df.loc[noise_indices[3*chunk:4*chunk], "item_id"] = 888888
            # Null ratings
            interactions_df.loc[noise_indices[4*chunk:], "rating"] = np.nan

            # Also duplicate a few rows
            dup_rows = interactions_df.iloc[:20].copy()
            interactions_df = pd.concat([interactions_df, dup_rows], ignore_index=True)

        return users_df, items_df, interactions_df

    def ingest(self):
        """
        Executes ingestion and writes users.csv, items.csv, and interactions.csv to data/raw.
        """
        source = self.config["ingestion"].get("source_type", "movielens_synthetic")
        users_file = os.path.join(self.raw_dir, "users.csv")
        items_file = os.path.join(self.raw_dir, "items.csv")
        interactions_file = os.path.join(self.raw_dir, "interactions.csv")

        users_df, items_df, interactions_df = self.generate_synthetic_movielens(
            num_users=self.config["ingestion"].get("num_users", 1000),
            num_items=self.config["ingestion"].get("num_items", 500),
            num_interactions=self.config["ingestion"].get("num_interactions", 50000),
            inject_noise=self.config["ingestion"].get("inject_noise_for_validation", True)
        )

        users_df.to_csv(users_file, index=False)
        items_df.to_csv(items_file, index=False)
        interactions_df.to_csv(interactions_file, index=False)

        print(f"[Ingestion] Successfully ingested raw data:")
        print(f"  - Users: {len(users_df):,} records -> {users_file}")
        print(f"  - Items: {len(items_df):,} records -> {items_file}")
        print(f"  - Interactions: {len(interactions_df):,} records -> {interactions_file}")

        return {
            "users_file": users_file,
            "items_file": items_file,
            "interactions_file": interactions_file,
            "user_count": len(users_df),
            "item_count": len(items_df),
            "interaction_count": len(interactions_df)
        }

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.ingest()
