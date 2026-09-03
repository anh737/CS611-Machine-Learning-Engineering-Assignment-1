# Credit Default Risk — Medallion Data Pipeline

A PySpark pipeline that turns four messy operational feeds into a point-in-time
feature store, then trains an XGBoost model to predict loan default.

The interesting part of this project is not the model — it is everything that has
to be true before a model is worth training. Raw credit data arrives with
placeholder junk in the categorical columns, ages in the hundreds, negative
account counts, and repayment history that spans months. Getting a trustworthy
label out of that, without accidentally showing the model the future, is the work.

> Built for CS611 (Machine Learning Engineering) at Singapore Management University.
> A later iteration — [mle_assignment_2](https://github.com/anh737/mle_assignment_2) —
> puts this pipeline behind Airflow and adds inference, drift monitoring and a
> leakage fix described under [Known limitations](#known-limitations).

---

## The problem

Given a loan applicant's demographics, financials and behavioural history, predict
whether the loan will go bad. "Bad" is defined the way a credit risk team defines
it: **30 or more days past due within the first 6 months on book.**

The data covers 12,500 customers and 137,500 monthly loan snapshots from January
2023 to November 2025.

## Architecture

```
data/                          Raw source feeds
  ├── lms_loan_daily.csv           loan book, one row per installment per month
  ├── features_attributes.csv      demographics, one row per customer
  ├── features_financials.csv      income, debt, repayment behaviour
  └── feature_clickstream.csv      20 anonymised behavioural counters
        │
        ▼
  ┌───────────┐   as-received copy, partitioned by month.
  │  BRONZE   │   Nothing is cleaned here on purpose — bronze stays
  └───────────┘   replayable if a downstream rule turns out wrong.
        │
        ▼
  ┌───────────┐   schema enforced, placeholder junk nulled out,
  │  SILVER   │   outliers repaired, duplicates dropped.
  └───────────┘   Clickstream events after the snapshot date are discarded.
        │
        ▼
  ┌───────────┐   label (30 DPD within 6 MOB) joined to 41 model features
  │   GOLD    │   on (Customer_ID, snapshot_date).
  └───────────┘
        │
        ▼
  ┌───────────┐   time-based train / test / out-of-time split,
  │   MODEL   │   XGBoost with randomised hyperparameter search.
  └───────────┘   Artefact pickled to model_bank/ with its scaler.
```

Each layer writes one partition per month, so any single month can be rebuilt
without touching the rest.

## Repository layout

```
main.py                                   full backfill: bronze → silver → gold
main_gold.py                              rebuild gold only (when the label definition changes)
train.py                                  train + evaluate + persist the model artefact
utils/
  ├── data_processing_bronze_table.py     raw ingest
  ├── data_processing_silver_table.py     cleaning and schema enforcement
  └── data_processing_gold_table.py       labels + feature engineering
data/sample/                              1,000-customer sample so the repo runs out of the box
data_processing_main.ipynb                pipeline walkthrough
test.ipynb                                exploratory data cleaning
Dockerfile, docker-compose.yaml           JupyterLab + Spark environment
```

Generated at runtime and git-ignored: `datamart/` (bronze, silver, gold) and
`model_bank/` (pickled artefacts).

## Running it

The pipeline needs Java for Spark, so Docker is the path of least resistance.

```bash
docker-compose up --build          # JupyterLab on http://localhost:8888
```

Then, inside the container (or in a local environment with Java 17 and
`pip install -r requirements.txt`):

```bash
cp data/sample/*.csv data/         # use the bundled sample; skip if you have the full feed

python main.py                     # backfill all 35 months, bronze → silver → gold
python main.py --snapshotdate 2024-06-01   # or just one month

python train.py --snapshot 2024-12-01      # train/test/OOT windows derived backwards
```

`train.py` prints AUC and Gini for the train, test and out-of-time sets, then
writes `model_bank/credit_model_<date>.pkl` containing the fitted model, the
scaler and the training configuration.

## About the data

The full dataset is synthetic course data and is not committed here. What ships
instead is `data/sample/` — 1,000 customers drawn at random, filtered
consistently across all four feeds so joins and labels still behave. It is enough
to run the pipeline end to end and see real output; it is not enough to reproduce
model metrics from the full feed.

Two quirks worth knowing before reading the silver layer:

| Column | What arrives | What the pipeline does |
|---|---|---|
| `Occupation` | `_______` | nulled |
| `SSN` | `#F%$D@*&8` | nulled |
| `Age` | `-500`, `8698`, `28_` | strip non-digits, then replace anything outside 18–100 with the modal valid age |
| `Payment_Behaviour` | `!@9#%8` | nulled |
| `Credit_Mix` | `_` | nulled |
| `Num_of_Loan` | negative, or in the hundreds | replaced with the length of the `Type_of_Loan` list |
| `Delay_from_due_date` | negative | absolute value |
| `Credit_History_Age` | `"22 Years and 1 Months"` | regex-parsed to months |

## Design decisions

**Bronze stores raw, not clean.** Every cleaning rule above is a judgement call
that could turn out wrong. Keeping an untouched copy means a rule can be revised
and the affected months replayed, rather than re-requested from the source system.

**Outliers are repaired, not dropped.** An applicant with a corrupted age still
has a real loan and a real outcome. Dropping the row would bias the label
distribution; imputing the modal valid age keeps the row and confines the damage
to one feature.

**The train/test split is chronological, not random.** OOT is the two most recent
months, train/test the twelve before that. A random split would let the model
learn from months that, in production, would not have happened yet — and would
flatter the metrics accordingly.

**The scaler is fitted on train only** and carried inside the model artefact, so
scoring applies the same transformation rather than re-deriving it.

**Clickstream is filtered to on-or-before the snapshot date** in silver, so
behavioural events logged after the application cannot reach the feature set.

## Known limitations

Worth stating plainly, since the follow-up project exists mostly to fix the first
one:

- **The gold row is keyed on the latest snapshot in the performance window, not
  the application date.** Financials and attributes only exist at month 0, so
  joining on a later date returns nulls for those columns; clickstream joined at
  month 6 carries information that would not exist at scoring time. The
  [Airflow version](https://github.com/anh737/mle_assignment_2) anchors the row at
  the application date instead. This repo is left as it was submitted.
- **Roughly 28% of customers have no clickstream rows at all**, so `fe_1`–`fe_20`
  are missing for them rather than zero-valued.
- **No automated tests.** Correctness rests on the notebook audits.
- `main.py` and `main_gold.py` share a copy of the same date-window helper.

## Stack

PySpark 3.5 · XGBoost · scikit-learn · pandas · Docker

## License

MIT — see [LICENSE](LICENSE).
