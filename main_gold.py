"""Rebuild only the gold label store, reusing existing silver partitions.

Useful when the label definition changes and bronze/silver do not need to be
re-ingested.

Usage:
    python main_gold.py                            # rebuild every month
    python main_gold.py --snapshotdate 2024-06-01  # rebuild a single month
"""

import os
import glob
import argparse
from datetime import datetime

import pyspark

import utils.data_processing_gold_table


def generate_first_of_month_dates(start_date_str, end_date_str):
    """Return the first day of each calendar month in [start, end] as YYYY-MM-DD."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def main(snapshot_date_arg=None):
    spark = pyspark.sql.SparkSession.builder \
        .appName("medallion-gold-only") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    gold_label_store_directory = "datamart/gold/label_store/"
    os.makedirs(gold_label_store_directory, exist_ok=True)

    if snapshot_date_arg:
        dates_str_lst = [snapshot_date_arg]
        print(f"Single-month gold rebuild: {snapshot_date_arg}")
    else:
        start_date_str = "2023-01-01"
        end_date_str = "2025-11-01"
        dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
        print(f"Gold rebuild window: {dates_str_lst[0]} to {dates_str_lst[-1]} ({len(dates_str_lst)} months)")

    print("\n== Gold: label store ==")
    for date_str in dates_str_lst:
        print(f"\n-- {date_str} --")
        utils.data_processing_gold_table.process_labels_gold_table(
            snapshot_date_str=date_str,
            gold_label_store_directory=gold_label_store_directory,
            spark=spark,
            mob=6,
            dpd=30,
        )

    # Read the partitions back as a consolidation check.
    files_list = glob.glob(os.path.join(gold_label_store_directory, "*.parquet"))

    if files_list:
        print(f"\nConsolidating {len(files_list)} gold partitions...")
        df = spark.read.option("header", "true").parquet(*files_list)
        print(f"Gold label store row count: {df.count()}")
    else:
        print("\nNo gold partitions found.")
        print("Check that the silver snapshots contain loans matured past the requested MOB window.")

    spark.stop()
    print("\nGold rebuild complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold label store rebuild runner")
    parser.add_argument(
        "--snapshotdate",
        type=str,
        required=False,
        help="Rebuild a single month (YYYY-MM-DD). Omit to rebuild the full window.",
    )
    args = parser.parse_args()
    main(snapshot_date_arg=args.snapshotdate)
