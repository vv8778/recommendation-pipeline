import numpy as np

def precision_at_k(actual_relevant: set, recommended_items: list, k: int = 10) -> float:
    """
    Computes Precision@K: Fraction of top-K recommended items that are relevant to the user.
    """
    if k <= 0 or not recommended_items:
        return 0.0
    top_k = recommended_items[:k]
    hits = len(set(top_k).intersection(actual_relevant))
    return float(hits / k)

def recall_at_k(actual_relevant: set, recommended_items: list, k: int = 10) -> float:
    """
    Computes Recall@K: Fraction of user's ground-truth relevant items captured in top-K.
    """
    if not actual_relevant or not recommended_items:
        return 0.0
    top_k = recommended_items[:k]
    hits = len(set(top_k).intersection(actual_relevant))
    return float(hits / len(actual_relevant))

def ndcg_at_k(actual_relevant: set, recommended_items: list, k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    """
    if not actual_relevant or not recommended_items:
        return 0.0
    top_k = recommended_items[:k]
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in actual_relevant:
            dcg += 1.0 / np.log2(idx + 2) # idx=0 -> log2(2) = 1.0

    # Ideal DCG (IDCG)
    ideal_hits = min(k, len(actual_relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return float(dcg / idcg) if idcg > 0.0 else 0.0

def catalog_coverage(all_recommended_lists: list, total_catalog_items: set) -> float:
    """
    Computes Catalog Coverage: Fraction of all available catalog items recommended
    at least once across all evaluated users.
    """
    if not total_catalog_items:
        return 0.0
    unique_recommended = set()
    for rec_list in all_recommended_lists:
        unique_recommended.update(rec_list)
    return float(len(unique_recommended.intersection(total_catalog_items)) / len(total_catalog_items))

def intra_list_diversity(recommended_items: list, item_genres_map: dict) -> float:
    """
    Computes Intra-List Diversity (ILD) based on pairwise Jaccard genre dissimilarity.
    Diversity = (2 / (k * (k - 1))) * sum(1 - Jaccard(genre_i, genre_j))
    """
    k = len(recommended_items)
    if k <= 1:
        return 1.0

    dissimilarities = []
    for i in range(k):
        genres_i = set(item_genres_map.get(recommended_items[i], []))
        for j in range(i + 1, k):
            genres_j = set(item_genres_map.get(recommended_items[j], []))
            union = genres_i.union(genres_j)
            if not union:
                dissimilarities.append(1.0)
            else:
                jaccard_sim = len(genres_i.intersection(genres_j)) / len(union)
                dissimilarities.append(1.0 - jaccard_sim)

    return float(np.mean(dissimilarities)) if dissimilarities else 1.0

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE) for rating predictions."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE) for rating predictions."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
