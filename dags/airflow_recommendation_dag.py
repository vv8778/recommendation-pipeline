"""
Apache Airflow DAG for Recommendation System End-to-End Pipeline
Author: Machine Learning Data Management
Schedule: Daily @midnight
"""

from datetime import datetime, timedelta
import os
import sys

# Attempt importing Airflow components
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

default_args = {
    'owner': 'data-ml-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    dag = DAG(
        'recommendation_system_pipeline_dag',
        default_args=default_args,
        description='End-to-End Data Management Pipeline for Recommendation System',
        schedule_interval=timedelta(days=1),
        catchup=False,
        tags=['recommendation', 'data_management', 'mlflow', 'collaborative_filtering']
    )

    def task_ingest(**kwargs):
        from src.ingestion.ingest import DataIngestor
        ingestor = DataIngestor()
        return ingestor.ingest()

    def task_validate(**kwargs):
        from src.validation.validate import DataValidator
        validator = DataValidator()
        return validator.validate()

    def task_prepare(**kwargs):
        import pandas as pd
        from src.preparation.prepare import DataPreparator
        from src.validation.validate import DataValidator
        val = DataValidator().validate()
        prep = DataPreparator()
        return prep.prepare(val["valid_users_df"], val["valid_items_df"], val["valid_interactions_df"])

    def task_transform(**kwargs):
        import pandas as pd
        from src.transformation.transform import DataTransformer
        train_df = pd.read_parquet("data/processed/train.parquet")
        users_df = pd.read_parquet("data/processed/users_clean.parquet")
        items_df = pd.read_parquet("data/processed/items_clean.parquet")
        transformer = DataTransformer()
        return transformer.transform(train_df, users_df, items_df)

    def task_features(**kwargs):
        import pandas as pd
        from src.features.feature_store import FeatureStore
        train_df = pd.read_parquet("data/processed/train.parquet")
        users_df = pd.read_parquet("data/processed/users_clean.parquet")
        items_df = pd.read_parquet("data/processed/items_clean.parquet")
        fs = FeatureStore()
        return fs.build_features(train_df, users_df, items_df)

    def task_train(**kwargs):
        from src.training.train import ModelTrainer
        trainer = ModelTrainer()
        return trainer.train_models()

    def task_evaluate(**kwargs):
        from src.evaluation.evaluate import ModelEvaluator
        evaluator = ModelEvaluator()
        return evaluator.evaluate()

    # Airflow PythonOperators
    t1_ingest = PythonOperator(task_id='ingest_raw_data', python_callable=task_ingest, dag=dag)
    t2_validate = PythonOperator(task_id='validate_and_quarantine', python_callable=task_validate, dag=dag)
    t3_prepare = PythonOperator(task_id='prepare_and_time_split', python_callable=task_prepare, dag=dag)
    t4_transform = PythonOperator(task_id='transform_derived_datasets', python_callable=task_transform, dag=dag)
    t5_features = PythonOperator(task_id='build_feature_store', python_callable=task_features, dag=dag)
    t6_train = PythonOperator(task_id='train_models_mlflow', python_callable=task_train, dag=dag)
    t7_evaluate = PythonOperator(task_id='evaluate_and_benchmark', python_callable=task_evaluate, dag=dag)

    # Define Stage Dependencies
    t1_ingest >> t2_validate >> t3_prepare >> t4_transform >> t5_features >> t6_train >> t7_evaluate

else:
    print("Airflow is not installed in the local environment. Use 'python -m src.orchestration.pipeline' for execution.")
