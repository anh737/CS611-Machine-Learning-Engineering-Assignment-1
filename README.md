# Credit Default Risk — Medallion Data Pipeline

A PySpark data pipeline and XGBoost model for predicting loan default. Four raw
operational feeds are processed through bronze, silver and gold layers into a
point-in-time feature store with labels, and a model is trained on the result.

## Problem

Predict whether a loan will go bad, defined as **30 or more days past due within
the first 6 months on book**.

Data: 12,500 customers, 137,500 monthly loan snapshots, January 2023 to
November 2025.

## Architecture

```
data/                          Raw source feeds
  ├── lms_loan_daily.csv           loan book, one row per installment per month
  ├── features_attributes.csv      demographics, one row per customer
  ├── features_financials.csv      income, debt, repayment behaviour
  └── feature_clickstream.csv      20 anonymised behavioural counters
        │
        ▼
  ┌───────────┐   as-received copy of each feed, partitioned by month
  │  BRONZE   │
  └───────────┘
        │
        ▼
  ┌───────────┐   schema enforced, placeholder values nulled, outliers
  │  SILVER   │   repaired, duplicates dropped. Clickstream events after
  └───────────┘   the snapshot date are discarded.
        │
        ▼
  ┌───────────┐   label (30 DPD within 6 MOB) joined to 41 model features
  │   GOLD    │   on (Customer_ID, snapshot_date)
  └───────────┘
        │
        ▼
  ┌───────────┐   chronological train / test / out-of-time split,
  │   MODEL   │   XGBoost with randomised hyperparameter search.
  └───────────┘   Artefact pickled to model_bank/ with its scaler.
```

Each layer writes one partition per month, so a single month can be rebuilt
independently.

## Repository layout

```
main.py                                   full backfill: bronze → silver → gold
main_gold.py                              rebuild gold only
train.py                                  train, evaluate and persist the model
utils/
  ├── data_processing_bronze_table.py     raw ingest
  ├── data_processing_silver_table.py     cleaning and schema enforcement
  └── data_processing_gold_table.py       labels and feature engineering
data/sample/                              1,000-customer sample dataset
data_processing_main.ipynb                pipeline walkthrough
test.ipynb                                exploratory data cleaning
Dockerfile, docker-compose.yaml           JupyterLab + Spark environment
```

Generated at runtime and git-ignored: `datamart/` (bronze, silver, gold) and
`model_bank/` (pickled artefacts).

## Running it

Spark needs Java, so the Docker environment is the simplest route.

```bash
docker-compose up --build          # JupyterLab on http://localhost:8888
```

Inside the container, or locally with Java 17 and `pip install -r requirements.txt`:

```bash
cp data/sample/*.csv data/         # use the bundled sample; skip if you have the full feed

python main.py                     # backfill all 35 months, bronze → silver → gold
python main.py --snapshotdate 2024-06-01   # or a single month

python train.py --snapshot 2024-12-01      # train/test/OOT windows derived backwards
```

`train.py` prints AUC and Gini for the train, test and out-of-time sets and
writes `model_bank/credit_model_<date>.pkl` containing the fitted model, the
scaler and the training configuration.

## Data

The full dataset is not committed. `data/sample/` contains 1,000 customers
drawn at random and filtered consistently across all four feeds, so the
pipeline runs end to end on it. Model metrics from the full feed are not
reproducible from the sample.

Cleaning rules applied in the silver layer:

| Column | Raw value | Rule |
|---|---|---|
| `Occupation` | `_______` | nulled |
| `SSN` | `#F%$D@*&8` | nulled |
| `Age` | `-500`, `8698`, `28_` | non-digits stripped; values outside 18–100 replaced with the modal valid age |
| `Payment_Behaviour` | `!@9#%8` | nulled |
| `Credit_Mix` | `_` | nulled |
| `Num_of_Loan` | negative or > 15 | replaced with the length of the `Type_of_Loan` list |
| `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_Credit_Inquiries` | out of range | replaced with the modal in-range value |
| `Delay_from_due_date` | negative | absolute value |
| `Credit_History_Age` | `"22 Years and 1 Months"` | parsed to months |

## Design decisions

- **Bronze is stored raw.** Cleaning rules can be revised and the affected
  months replayed from bronze without re-ingesting from source.
- **Outliers are repaired rather than dropped**, so the label distribution is
  not biased by removing rows.
- **The train/test split is chronological.** OOT is the two most recent
  months; train/test is the twelve months before that.
- **The scaler is fitted on train only** and stored in the model artefact.
- **Clickstream is filtered to on-or-before the snapshot date** in silver.

## Known limitations

- The gold row is keyed on the latest snapshot in the performance window rather
  than the application date. Financials and attributes only exist at month 0,
  so they join as null on later dates, and clickstream joined at month 6
  includes behaviour after the application. The follow-up project
  ([CS611-Machine-Learning-Engineering-Assignment-2](https://github.com/anh737/CS611-Machine-Learning-Engineering-Assignment-2)) anchors the
  row at the application date.
- About 28% of customers have no clickstream rows, so `fe_1`–`fe_20` are null
  for them.
- No automated tests.
- `main.py` and `main_gold.py` each contain a copy of the same date-window helper.

## Stack

PySpark 3.5 · XGBoost · scikit-learn · pandas · Docker

## License

MIT — see [LICENSE](LICENSE).
