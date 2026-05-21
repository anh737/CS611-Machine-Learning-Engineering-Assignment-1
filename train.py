import os
import glob
import argparse
import pickle
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score


def create_dynamic_config(model_train_date_str, train_test_period_months=12, oot_period_months=2, train_test_ratio=0.8):
    """
    Generate chronological training, testing, and Out-of-Time (OOT) anchors
    based on the input target snapshot train date.
    """
    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] = oot_period_months
    config["train_test_ratio"] = train_test_ratio
    
    # Timeline calculations
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
    config["oot_start_date"] = config['model_train_date'] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    
    return config


def main(snapshot_date_arg):
    print("\n=====================================================================")
    print("      INITIATING PRODUCTION MODEL TRAINING PIPELINE ARCHITECTURE     ")
    print("=====================================================================\n")

    # 1. Initialize heavy local-optimized Spark Session
    spark = pyspark.sql.SparkSession.builder \
        .appName("Credit-Model-Training") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Build configuration setup
    config = create_dynamic_config(model_train_date_str=snapshot_date_arg)
    print("[SYSTEM INFO] Training configuration timeline anchors finalized:")
    pprint.pprint(config)

    # 3. Connect and extract from Gold Label Store (Now incorporating all custom features)
    label_folder_path = "datamart/gold/label_store/"
    label_files = glob.glob(os.path.join(label_folder_path, '*.parquet'))
    
    if not label_files:
        raise FileNotFoundError(f"[CRITICAL] No gold label parquet shards discovered at {label_folder_path}")
        
    label_store_sdf = spark.read.option("header", "true").parquet(*label_files)
    
    # Apply dynamic historical timeline filter constraints
    labels_sdf = label_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) & 
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print(f"\n[INFO] Extracted Gold Datasets snapshot records count: {labels_sdf.count()}")

    # 4. Migrate full unified analytical store to Pandas DataFrame
    print("[INFO] Loading analytical storage framework into Pandas ecosystem...")
    data_pdf = labels_sdf.toPandas()
    
    # Release Spark Session background JVM resources immediately to free memory
    spark.stop()

    if data_pdf.empty:
        print("[WARNING] Compiled master dataframe is empty. Terminating process.")
        return

    # 5. Categorize timelines into Train-Test and Out-of-Time (OOT) sets
    print("[INFO] Stratifying datasets into specialized chronological cohorts...")
    
    # Cast dates safely for Pandas filtering comparison operations
    data_pdf['snapshot_date'] = pd.to_datetime(data_pdf['snapshot_date']).dt.date
    
    oot_start_d = config["oot_start_date"].date()
    oot_end_d = config["oot_end_date"].date()
    tt_start_d = config["train_test_start_date"].date()
    tt_end_d = config["train_test_end_date"].date()

    # Split into chronological data blocks
    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= oot_start_d) & (data_pdf['snapshot_date'] <= oot_end_d)].copy()
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= tt_start_d) & (data_pdf['snapshot_date'] <= tt_end_d)].copy()

    # Define features block explicitly from your updated schema 
    # CRITICAL: Exclude identifiers, text-based raw categories (Occupation, Payment_Behaviour) and labels!
    feature_cols = [
        "Debt_to_Income_Ratio", "EMI_to_Salary_Ratio", "Savings_Propensity", 
        "Credit_History_Age_Years", "is_Credit_Age_gt_15", "Num_Bank_Accounts", 
        "Num_Credit_Card", "Num_of_Loan", "Interest_Rate", "Monthly_Inhand_Salary", 
        "Annual_Income", "Outstanding_Debt", "Credit_Utilization_Ratio", 
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance", 
        "Is_Age_gt_45", "Age", 
        "Occupation_Idx", "Spend_Level_Idx", "Payment_Value_Level_Idx", # Custom target encoded numerical features
        "fe_1", "fe_2", "fe_3", "fe_4", "fe_5", "fe_6", "fe_7", "fe_8", "fe_9", "fe_10",
        "fe_11", "fe_12", "fe_13", "fe_14", "fe_15", "fe_16", "fe_17", "fe_18", "fe_19", "fe_20"
    ]

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], 
        train_test_pdf["label"], 
        test_size=1.0 - config["train_test_ratio"],
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )

    print(f"   ↳ Finalized Model Features Count: {len(feature_cols)}")
    print(f"   ↳ Train Size: {X_train_raw.shape[0]} | Bad Rate: {round(y_train.mean(), 2)}")
    print(f"   ↳ Test Size:  {X_test_raw.shape[0]} | Bad Rate: {round(y_test.mean(), 2)}")
    print(f"   ↳ OOT Size:   {X_oot.shape[0]} | Bad Rate: {round(y_oot.mean(), 2)}\n")

    # 6. Apply Standard Scaler normalization mapped to Training boundaries
    print("[INFO] Fitting Standard Scaler preprocessing pipelines...")
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train_raw)

    X_train_processed = transformer_stdscaler.transform(X_train_raw)
    X_test_processed = transformer_stdscaler.transform(X_test_raw)
    X_oot_processed = transformer_stdscaler.transform(X_oot)

    # 7. Train Model via Hyperparameter Optimization (XGBoost)
    print("[INFO] Initializing grid optimization for Hyperparameter Tuning...")
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)

    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    auc_scorer = make_scorer(roc_auc_score)
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=100,
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    print(">>> Fitting CV candidate permutations (totalling 300 fits)...")
    random_search.fit(X_train_processed, y_train)

    best_model = random_search.best_estimator_
    print(f"[SUCCESS] Optimal parameters unlocked: {random_search.best_params_}")

    # 8. Compute validation metrics & evaluate Gini coefficients
    train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train_processed)[:, 1])
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_processed)[:, 1])
    oot_auc = roc_auc_score(y_oot, best_model.predict_proba(X_oot_processed)[:, 1])

    print("\n--- Model Validation Diagnostic Summary ---")
    print(f"   ↳ Train AUC: {round(train_auc, 4)} | Gini: {round(2 * train_auc - 1, 3)}")
    print(f"   ↳ Test AUC:  {round(test_auc, 4)}  | Gini: {round(2 * test_auc - 1, 3)}")
    print(f"   ↳ OOT AUC:   {round(oot_auc, 4)}  | Gini: {round(2 * oot_auc - 1, 3)}\n")

    # 9. Construct unified serialization metadata object (Artefact)
    model_version = "credit_model_" + config["model_train_date_str"].replace('-', '_')
    
    model_artefact = {
        'model': best_model,
        'model_version': model_version,
        'feature_names': feature_cols, # Secured column track mapping array sequence
        'preprocessing_transformers': {'stdscaler': transformer_stdscaler},
        'data_dates': config,
        'data_stats': {
            'X_train': X_train_raw.shape[0], 'X_test': X_test_raw.shape[0], 'X_oot': X_oot.shape[0],
            'y_train': round(y_train.mean(), 2), 'y_test': round(y_test.mean(), 2), 'y_oot': round(y_oot.mean(), 2)
        },
        'results': {
            'auc_train': train_auc, 'auc_test': test_auc, 'auc_oot': oot_auc,
            'gini_train': round(2 * train_auc - 1, 3), 'gini_test': round(2 * test_auc - 1, 3), 'gini_oot': round(2 * oot_auc - 1, 3)
        },
        'hp_params': random_search.best_params_
    }

    # 10. Persistence layer to model bank storage
    model_bank_dir = "model_bank/"
    if not os.path.exists(model_bank_dir):
        os.makedirs(model_bank_dir)

    output_file_path = os.path.join(model_bank_dir, f"{model_version}.pkl")
    with open(output_file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"[SUCCESS] Core model artefact secured and exported to: {output_file_path}")
    print("\n=====================================================================")
    print("             PIPELINE SYSTEM SHUTDOWN COMPLETED CLEANLY              ")
    print("=====================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Credit Scoring Model Training Platform")
    parser.add_argument(
        "--snapshot", 
        type=str, 
        required=True, 
        help="Target execution snapshot point in time formatted as YYYY-MM-DD"
    )
    
    args = parser.parse_args()
    main(snapshot_date_arg=args.snapshot)