# Recommendation Pipeline Feature Catalog

This catalog documents all registered features across User, Item, and Interaction entities. All features are defined with strict calculation semantics and stored in columnar Parquet format to ensure zero-leakage training and sub-millisecond retrieval during inference.

| Feature Name | Entity | Data Type | Description | Calculation Logic | Reusability Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_avg_rating` | **User** | `Float` | Historical mean rating provided by the user across training records. | `mean(ratings) per user` | Offline Training & Online Serving |
| `user_rating_count` | **User** | `Integer` | Total number of interactions submitted by user. | `count(ratings) per user` | Offline Training & Online Serving |
| `user_rating_std` | **User** | `Float` | Standard deviation of user ratings measuring rating generosity/criticality. | `std(ratings) per user` | Offline Training & Online Serving |
| `user_recency_days` | **User** | `Float` | Elapsed days since the user's latest interaction timestamp. | `(max_pipeline_ts - last_user_ts) / 86400` | Offline Training & Online Serving |
| `user_preferred_genre` | **User** | `String` | The most frequently consumed / highest rated genre by user. | `mode(genres) per user interactions` | Offline Training & Online Serving |
| `gender_encoded` | **User** | `Integer` | Binary encoded user gender (1 = Male, 0 = Female). | `gender == 'M' -> 1 else 0` | Offline Training & Online Serving |
| `user_rating_bias` | **User** | `Float` | Offset of user average rating from global mean rating. | `user_avg_rating - global_mean` | Offline Training & Online Serving |
| `item_avg_rating` | **Item** | `Float` | Historical mean rating received by the item. | `mean(ratings) per item` | Offline Training & Online Serving |
| `item_rating_count` | **Item** | `Integer` | Total interaction volume received by item. | `count(ratings) per item` | Offline Training & Online Serving |
| `item_bayesian_rating` | **Item** | `Float` | Bayesian shrinkage rating balancing raw item mean with global prior. | `(v/(v+m))*R + (m/(v+m))*C` | Offline Training & Online Serving |
| `item_popularity_score` | **Item** | `Float` | Normalized percentile popularity rank of the item [0.0, 1.0]. | `rank(item_rating_count, pct=True)` | Offline Training & Online Serving |
| `item_age_years` | **Item** | `Integer` | Age of item in years calculated from release year. | `current_year - release_year` | Offline Training & Online Serving |
| `genre_* (multi-hot)` | **Item** | `Binary Vector` | 17-dimensional binary one-hot/multi-hot genre indicator flags. | `1 if genre in item.genres else 0` | Offline Training & Online Serving |
| `item_rating_bias` | **Item** | `Float` | Offset of item average rating from global mean rating. | `item_avg_rating - global_mean` | Offline Training & Online Serving |
| `genre_match` | **Interaction** | `Binary` | Indicates whether the interacted item contains the user's preferred genre. | `1 if user_preferred_genre in item.genres else 0` | Offline Training |
| `user_item_diff_from_avg` | **Interaction** | `Float` | Residual deviation between observed rating and user/item baseline. | `rating - (user_avg + item_avg)/2` | Offline Training |
