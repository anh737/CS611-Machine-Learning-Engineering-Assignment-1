import os
import glob
import argparse
from datetime import datetime
import pyspark

# Import only the required structural Gold ETL sub-module
import utils.data_processing_gold_table

def generate_first_of_month_dates(start_date_str, end_date_str):
    """
    Compute and generate a sequence of monthly snapshot strings ('YYYY-MM-DD')
    representing the first day of each calendar month within the specified range.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Increment calendar month index systematically
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

def main(snapshot_date_arg=None):
    print("\n=====================================================================")
    print("      INITIATING PRODUCTION METRICS GOLD LAYER BACKFILL ONLY         ")
    print("=====================================================================\n")
    
    # 1. Spark Session Initialization with localized heavy multi-threading parameters
    spark = pyspark.sql.SparkSession.builder \
        .appName("Medallion-Orchestration-Gold-Only") \
        .master("local[*]") \
        .getOrCreate()
    
    # Enforce severe error filtering to minimize diagnostic console clutter
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Environment Path Configurations
    gold_label_store_directory = "datamart/gold/label_store/"

    # Scaffold base storage directory infrastructure if absent
    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    # 3. Determine Timeline Execution Mode (Single-run vs Full Backfill)
    if snapshot_date_arg:
        # Single-date targeted execution mode
        dates_str_lst = [snapshot_date_arg]
        print(f"[SYSTEM INFO] Targeted single-date Gold execution triggered for: {snapshot_date_arg}\n")
    else:
        # Default full historical monthly backfill timeline mode
        start_date_str = "2023-01-01"
        end_date_str = "2025-11-01"
        dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
        print(f"[SYSTEM INFO] Default multi-month historical backfill mode activated for Gold Layer.")
        print(f"[SYSTEM INFO] Coverage timeframe: {dates_str_lst[0]} through {dates_str_lst[-1]}")
        print(f"[SYSTEM INFO] Total batches to process: {len(dates_str_lst)}\n")

    # =========================================================================
    # PIPELINE EXECUTION: TIER 3 — GOLD PROCESSING LAYER ONLY
    # =========================================================================
    print(">>> Executing Feature Engineering Loops on Gold Framework...")
    for date_str in dates_str_lst:
        print(f"\n--- Processing batch window: {date_str} ---")
        utils.data_processing_gold_table.process_labels_gold_table(
            snapshot_date_str=date_str, 
            gold_label_store_directory=gold_label_store_directory, 
            spark=spark, 
            mob=6, 
            dpd=30
        )
    print("\n[SUCCESS] Gold Layer feature store assets generated.")

    print("\n=====================================================================")
    print("         GOLD AGGREGATION & CONSOLIDATION VALIDATION RUNNING          ")
    print("=====================================================================\n")
    
    # Secure safe paths for all created parquet shards using glob
    # Only target non-empty generated directories to prevent empty schema analysis crashes
    files_list = glob.glob(os.path.join(gold_label_store_directory, "*.parquet"))
    
    if len(files_list) > 0:
        print(f"[INFO] Found {len(files_list)} generated Parquet partitions. Consolidating records...")
        
        # Read the generated gold parquets back to evaluate state records
        df = spark.read.option("header", "true").parquet(*files_list)
        print("Total Consolidated Gold Row Count:", df.count())
        
        # Display sample data structure
        #df.show(20, truncate=False)
    else:
        print("[WARNING] No valid gold Parquet files were found in the storage target.")
        print("[WARNING] Ensure source Silver snapshots have data fields that mature past the requested MOB window.")
    
    # Safe release of multi-threaded JVM background resources
    spark.stop()
    print("\n[PROCESS CONCLUDED] Spark session released successfully.")

if __name__ == "__main__":
    # Configure argument parser to accept optional snapshot date flag
    parser = argparse.ArgumentParser(description="Orchestrated Gold Layer Feature Ingestion Runner")
    parser.add_argument(
        "--snapshotdate", 
        type=str, 
        required=False, 
        help="Target date filter formatted as YYYY-MM-DD. Omit flag to execute full historical backfill cycle."
    )
    
    args = parser.parse_args()
    
    # Delegate runtime execution flow to main
    main(snapshot_date_arg=args.snapshotdate)