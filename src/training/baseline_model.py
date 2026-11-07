import numpy as np
import pandas as pd

class PopularityRecommender:
    """
    Non-personalized baseline recommender.
    Scores items based on Bayesian smoothed average ratings and interaction frequency.
    """
    def __init__(self, bayesian_prior_weight: float = 10.0):
        self.bayesian_prior_weight = bayesian_prior_weight
        self.global_mean = 3.5
        self.item_scores = {}
        self.sorted_items = []
        self.user_training_items = {}

    def fit(self, train_df: pd.DataFrame):
        """Fits baseline popularity model from training interactions."""
        self.global_mean = float(train_df["rating"].mean())
        
        # Track training items seen per user to avoid re-recommending
        self.user_training_items = (
            train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        item_stats = train_df.groupby("item_id")["rating"].agg(["count", "mean"]).reset_index()
        m = self.bayesian_prior_weight
        C = self.global_mean

        v = item_stats["count"]
        R = item_stats["mean"]
        # Bayesian weighted score
        item_stats["score"] = (v / (v + m)) * R + (m / (v + m)) * C

        item_stats = item_stats.sort_values(by="score", ascending=False)
        self.item_scores = dict(zip(item_stats["item_id"], item_stats["score"]))
        self.sorted_items = list(item_stats["item_id"])

        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predicts item smoothed rating as baseline rating estimation."""
        return float(self.item_scores.get(item_id, self.global_mean))

    def recommend(self, user_id: int, top_k: int = 10, all_candidate_items: list = None) -> list:
        """
        Recommends top_k most popular items excluding already interacted items.
        """
        seen_items = self.user_training_items.get(user_id, set())
        recs = []
        for iid in self.sorted_items:
            if iid not in seen_items:
                recs.append(iid)
                if len(recs) == top_k:
                    break
        
        # If candidate pool is exhausted, fallback to top candidates
        if len(recs) < top_k and all_candidate_items:
            for iid in all_candidate_items:
                if iid not in seen_items and iid not in recs:
                    recs.append(iid)
                    if len(recs) == top_k:
                        break

        return recs
