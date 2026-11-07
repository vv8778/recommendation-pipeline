import os
import pickle
import yaml
import pandas as pd
import mlflow
from src.training.baseline_model import PopularityRecommender
from src.training.collaborative_model import MatrixFactorizationRecommender

class ModelTrainer:
    """
    Orchestrates recommendation model training, serialization,
    and experiment tracking with MLflow.
    """
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models_dir = self.config["paths"]["models_dir"]
        self.processed_dir = self.config["paths"]["processed_data_dir"]
        os.makedirs(self.models_dir, exist_ok=True)

        mlflow_uri = self.config["paths"].get("mlflow_tracking_uri", "sqlite:///models/mlflow.db")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(self.config["pipeline"].get("name", "recommendation-pipeline"))

    def train_models(self, train_df: pd.DataFrame = None):
        """
        Trains both Baseline Popularity and Personalized Matrix Factorization models.
        Logs parameters and artifacts to MLflow.
        """
        if train_df is None:
            train_df = pd.read_parquet(os.path.join(self.processed_dir, "train.parquet"))

        print(f"[Training] Training recommendation models on {len(train_df):,} interactions...")

        # -------------------------------------------------------------
        # 1. Baseline Popularity Recommender
        # -------------------------------------------------------------
        print("\n--- Training Model 1: Baseline Popularity Recommender ---")
        pop_cfg = self.config["models"]["baseline_popularity"]
        baseline_model = PopularityRecommender(
            bayesian_prior_weight=pop_cfg.get("bayesian_prior_weight", 10.0)
        )
        
        with mlflow.start_run(run_name="Baseline_Popularity"):
            baseline_model.fit(train_df)
            mlflow.log_param("model_type", "Popularity_Recommender")
            mlflow.log_param("bayesian_prior_weight", pop_cfg.get("bayesian_prior_weight", 10.0))
            mlflow.log_param("num_train_records", len(train_df))
            mlflow.log_param("num_catalog_items", len(baseline_model.item_scores))

            baseline_path = os.path.join(self.models_dir, "baseline_popularity.pkl")
            with open(baseline_path, "wb") as f:
                pickle.dump(baseline_model, f)
            mlflow.log_artifact(baseline_path)
            print(f"  -> Baseline model saved to: {baseline_path}")

        # -------------------------------------------------------------
        # 2. Personalized Matrix Factorization Recommender
        # -------------------------------------------------------------
        print("\n--- Training Model 2: Personalized Matrix Factorization (FunkSVD) ---")
        mf_cfg = self.config["models"]["collaborative_svd"]
        cf_model = MatrixFactorizationRecommender(
            n_factors=mf_cfg.get("n_factors", 20),
            learning_rate=mf_cfg.get("learning_rate", 0.005),
            regularization=mf_cfg.get("regularization", 0.02),
            n_epochs=mf_cfg.get("n_epochs", 20),
            random_seed=self.config["pipeline"].get("random_seed", 42)
        )

        with mlflow.start_run(run_name="Personalized_Matrix_Factorization"):
            cf_model.fit(train_df)
            mlflow.log_param("model_type", "Matrix_Factorization_FunkSVD")
            mlflow.log_param("n_factors", cf_model.n_factors)
            mlflow.log_param("learning_rate", cf_model.learning_rate)
            mlflow.log_param("regularization", cf_model.regularization)
            mlflow.log_param("n_epochs", cf_model.n_epochs)
            mlflow.log_param("num_train_records", len(train_df))

            cf_path = os.path.join(self.models_dir, "collaborative_svd.pkl")
            with open(cf_path, "wb") as f:
                pickle.dump(cf_model, f)
            mlflow.log_artifact(cf_path)
            print(f"  -> Personalized model saved to: {cf_path}")

        return {
            "baseline_model": baseline_model,
            "collaborative_model": cf_model,
            "baseline_path": baseline_path,
            "collaborative_path": cf_path
        }

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_models()
