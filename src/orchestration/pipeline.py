import os
import sys
import time
import argparse
import yaml
from datetime import datetime

from src.ingestion.ingest import DataIngestor
from src.validation.validate import DataValidator
from src.preparation.prepare import DataPreparator
from src.transformation.transform import DataTransformer
from src.features.feature_store import FeatureStore
from src.training.train import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator

class RecommendationPipeline:
    """
    End-to-End Orchestrator for the Recommendation System Data Management Pipeline.
    Coordinates dependencies across:
      1. Data Ingestion
      2. Data Validation & Quarantine
      3. Data Preparation & Time-Based Split
      4. Data Transformation & Analytical Summaries
      5. Feature Store Construction & Cataloging
      6. Model Training & MLflow Experiment Tracking
      7. Model Evaluation & Comparative Benchmarking
    """
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.execution_log = []

    def log_stage(self, stage_name: str, duration: float, status: str = "SUCCESS", details: str = ""):
        self.execution_log.append({
            "stage": stage_name,
            "duration_sec": round(duration, 2),
            "status": status,
            "details": details
        })

    def run(self):
        start_total = time.time()
        print("=" * 80)
        print("STARTING END-TO-END RECOMMENDATION SYSTEM DATA MANAGEMENT PIPELINE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        try:
            # ---------------------------------------------------------
            # Stage 1: Data Ingestion
            # ---------------------------------------------------------
            t0 = time.time()
            print("\n[STAGE 1/7] DATA INGESTION")
            ingestor = DataIngestor(self.config_path)
            ingest_res = ingestor.ingest()
            self.log_stage(
                "Data Ingestion", time.time() - t0, "SUCCESS",
                f"Ingested {ingest_res['interaction_count']:,} interactions"
            )

            # ---------------------------------------------------------
            # Stage 2: Data Validation & Quarantine
            # ---------------------------------------------------------
            t0 = time.time()
            print("\n[STAGE 2/7] DATA VALIDATION & QUALITY ASSURANCE")
            validator = DataValidator(self.config_path)
            val_res = validator.validate()
            summary = val_res["report"]["summary"]
            self.log_stage(
                "Data Validation", time.time() - t0, "SUCCESS",
                f"Passed {summary['total_valid_interactions']:,} / Quarantined {summary['total_quarantined_interactions']:,} ({summary['quarantine_error_rate_pct']}%)"
            )

            # ---------------------------------------------------------
            # Stage 3: Data Preparation & Time-Based Split
            # ---------------------------------------------------------
            t0 = time.time()
            print("\n[STAGE 3/7] DATA PREPARATION & CHRONOLOGICAL TRAIN/TEST SPLIT")
            preparator = DataPreparator(self.config_path)
            prep_res = preparator.prepare(
                val_res["valid_users_df"],
                val_res["valid_items_df"],
                val_res["valid_interactions_df"]
            )
            self.log_stage(
                "Data Preparation", time.time() - t0, "SUCCESS",
                f"Train: {len(prep_res['train_df']):,} | Test: {len(prep_res['test_df']):,}"
            )

            # ---------------------------------------------------------
            # Stage 4: Data Transformation
            # ---------------------------------------------------------
            t0 = time.time()
            print("\n[STAGE 4/7] DATA TRANSFORMATION & DERIVED DATASETS")
            transformer = DataTransformer(self.config_path)
            trans_res = transformer.transform(
                prep_res["train_df"],
                prep_res["users_clean"],
                prep_res["items_clean"]
            )
            self.log_stage(
                "Data Transformation", time.time() - t0, "SUCCESS",
                f"Generated User Summary, Item Popularity & User-Item Matrix"
            )

            # ---------------------------------------------------------
            # Stage 5: Feature Management & Feature Store
            # ---------------------------------------------------------
            t0 = time.time()
            print("\n[STAGE 5/7] FEATURE ENGINEERING & FEATURE CATALOGING")
            feature_store = FeatureStore(self.config_path)
            feat_res = feature_store.build_features(
                prep_res["train_df"],
                prep_res["users_clean"],
                prep_res["items_clean"]
            )
            self.log_stage(
                "Feature Engineering", time.time() - t0, "SUCCESS",
                f"Cataloged {feat_res['user_features'].shape[1]} user & {feat_res['item_features'].shape[1]} item features"
            )

            # ---------------------------------------------------------
            # Stage 6: Model Training & Experiment Tracking
            # ---------------------------------------------------------
            t0 = time.time()
            print("\n[STAGE 6/7] MODEL TRAINING & MLFLOW TRACKING")
            trainer = ModelTrainer(self.config_path)
            train_res = trainer.train_models(prep_res["train_df"])
            self.log_stage(
                "Model Training", time.time() - t0, "SUCCESS",
                "Trained Baseline Popularity & Matrix Factorization (FunkSVD)"
            )

            # ---------------------------------------------------------
            # Stage 7: Model Evaluation & Benchmarking
            # ---------------------------------------------------------
            t0 = time.time()
            print("\n[STAGE 7/7] MODEL EVALUATION & BENCHMARKING")
            evaluator = ModelEvaluator(self.config_path)
            eval_res = evaluator.evaluate(
                train_res["baseline_model"],
                train_res["collaborative_model"]
            )
            self.log_stage(
                "Model Evaluation", time.time() - t0, "SUCCESS",
                f"Evaluated Precision@K, Recall@K, NDCG@K, Coverage, Diversity, RMSE, MAE"
            )

            total_duration = time.time() - start_total
            print("\n" + "=" * 80)
            print("PIPELINE EXECUTION SUMMARY")
            print("=" * 80)
            print(f"{'Stage':<30} | {'Status':<10} | {'Duration (s)':<12} | {'Output Details'}")
            print("-" * 80)
            for item in self.execution_log:
                print(f"{item['stage']:<30} | {item['status']:<10} | {item['duration_sec']:<12.2f} | {item['details']}")
            print("-" * 80)
            print(f"Total Pipeline Runtime: {total_duration:.2f} seconds")
            print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"\n[PIPELINE FAILURE] An error occurred: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Recommendation Pipeline Runner")
    parser.add_argument("--config", default="configs/pipeline_config.yaml", help="Path to config file")
    args = parser.parse_args()

    pipeline = RecommendationPipeline(config_path=args.config)
    success = pipeline.run()
    sys.exit(0 if success else 1)
