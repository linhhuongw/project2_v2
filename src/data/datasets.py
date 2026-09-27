"""Tier A — data declaration (configs/preprocessing_v1.md, section A).

Each loader reads data/interim and applies only rules A1–A9: pick the labelled table, filter rows
(undefined label, unfinished loans, out-of-scope applications), label 1 = bad, convert the split
date, drop ID / free-text / leakage columns, documented special codes -> NaN, string/date formats
-> numbers, and declare numeric vs categorical columns.
Every step passes the one-row test (no value is learned from the data), so it runs before splitting.

Every prepared frame has:
  * `row_id` — row position in the interim file (stable key for split assignments)
  * `target` — 1 = bad, 0 = good
  * the declared numeric and categorical feature columns
  * optional non-feature columns (`DatasetSpec.non_features`): join keys, the split date
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"


@dataclass
class DatasetSpec:
    name: str
    load: Callable[[], pd.DataFrame]
    categorical: list | None = field(default_factory=list)  # None = every string column
    method: str = "stratified_group_kfold"                  # or "time_holdout"
    non_features: list = field(default_factory=list)
    time_col: str | None = None
    periods: Callable[[], dict] | None = None               # time_holdout: {"train": (y0, y1), ...}
    notes: list = field(default_factory=list)


def _with_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df


# ------------------------------------------------------------------------------ loaders

def load_give_me_some_credit():
    # 96/98 in the late-payment counts are kept as-is to match Baesens
    return _with_row_id(pd.read_csv(INTERIM / "give_me_some_credit.csv"))


def load_taiwan():
    return _with_row_id(pd.read_csv(INTERIM / "taiwan_default_credit.csv")).drop(columns="ID")


def load_home_credit():
    df = _with_row_id(pd.read_parquet(INTERIM / "home_credit" / "application_train.parquet"))
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].mask(df["DAYS_EMPLOYED"] == 365243)
    return df


def load_hmeq():
    return _with_row_id(pd.read_csv(INTERIM / "hmeq.csv"))


def load_south_german_credit():
    return _with_row_id(pd.read_csv(INTERIM / "south_german_credit.csv"))


def load_fico_heloc():
    # cleaned version (9,871 rows): the 588 rows with every feature = -9 are already absent
    df = _with_row_id(pd.read_csv(INTERIM / "fico_heloc.csv"))
    feats = df.columns.drop(["row_id", "target"])
    df[feats] = df[feats].mask(df[feats].isin([-7, -8, -9]))
    return df


# ------------------------------------------------------------------------------ Lending Club

LC_FINISHED = {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
LC_POLICY_PREFIX = "Does not meet the credit policy. Status:"
LC_DROP_ID_TEXT = ["id", "member_id", "url", "desc", "title", "emp_title"]
LC_DROP_LEAK = ["loan_status", "recoveries", "collection_recovery_fee", "next_pymnt_d",
                "last_credit_pull_d", "pymnt_plan", "deferral_term", "payment_plan_start_date",
                "orig_projected_additional_accrued_interest"]
LC_DROP_LEAK_PREFIX = ("total_pymnt", "total_rec_", "out_prncp", "last_pymnt_", "last_fico_range_",
                       "debt_settlement_", "settlement_", "hardship_")
LC_DROP_OTHER = ["zip_code"]
# constant by construction after the row filters (individual only, 36-month only)
LC_DROP_CONSTANT = ["application_type", "verification_status_joint", "term"]
LC_START_YEAR = 2013  # bureau fields (tot_cur_bal, num_*, mo_sin_*, ...) are ~31% empty before 2013
LC_CATEGORICAL = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose",
                  "addr_state", "emp_length", "initial_list_status", "disbursement_method"]


def _month_to_year(s: pd.Series) -> pd.Series:
    """'Aug-2003' -> 2003 + 7/12 (format conversion only, A8)."""
    d = pd.to_datetime(s, format="%b-%Y")
    return d.dt.year + (d.dt.month - 1) / 12


@lru_cache(maxsize=1)
def _lc_individual() -> pd.DataFrame:
    """Accepted loans, individual applications only, statuses merged, split date parsed.
    Unfinished loans are still present (needed to pick the version-B years)."""
    df = _with_row_id(pd.read_parquet(INTERIM / "lending_club" / "accepted.parquet"))
    df = df[df["application_type"] == "Individual"].copy()                  # A2 (2)
    df["loan_status"] = df["loan_status"].str.replace(LC_POLICY_PREFIX, "", regex=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")          # A4
    df["term"] = df["term"].str.extract(r"(\d+)", expand=False).astype(float)  # A8
    return df


def lc_version_b_years() -> dict:
    df = _lc_individual()
    raw36 = df[df["term"] == 36]
    share = raw36["loan_status"].isin(LC_FINISHED).groupby(raw36["issue_d"].dt.year).mean()
    test_year = int(share[share >= 0.95].index.max())
    return {"train": (LC_START_YEAR, test_year - 2),
            "val": (test_year - 1, test_year - 1),
            "test": (test_year, test_year),
            "finished_share_36m": {int(k): round(float(v), 4) for k, v in share.items()}}


def load_lending_club_b():
    df = _lc_individual()
    df = df[(df["term"] == 36) & (df["issue_d"].dt.year >= LC_START_YEAR)]  # A2 (3), start year
    df = df[df["loan_status"].isin(LC_FINISHED)].copy()                    # A2 (1)
    df["target"] = df["loan_status"].map(LC_FINISHED).astype("int8")       # A3
    leak = [c for c in df.columns if c.startswith(LC_DROP_LEAK_PREFIX)]
    df = df.drop(columns=LC_DROP_ID_TEXT + LC_DROP_LEAK + leak + LC_DROP_OTHER + LC_DROP_CONSTANT)
    for c in ("earliest_cr_line", "sec_app_earliest_cr_line"):
        df[c] = _month_to_year(df[c])                                      # A8
    return df


LC_NOTES = [
    "individual applications only; 36-month loans only; issued from 2013 (earlier loans lack "
    "~22 bureau fields); finished loans only (Fully Paid / Charged Off / Default, "
    "'Does not meet the credit policy' statuses merged into their base status)",
    "dropped ID/text, leakage (incl. loan_status and the 5 extra post-origination columns), zip_code",
    "dropped application_type, verification_status_joint, term: constant by construction of the filters",
    "earliest_cr_line -> decimal year; issue_d kept as split column only",
]

SPECS = {
    "give_me_some_credit": DatasetSpec(
        "give_me_some_credit", load_give_me_some_credit,
        notes=["96/98 codes in late-payment counts kept (as in Baesens)"]),
    "taiwan_default_credit": DatasetSpec(
        "taiwan_default_credit", load_taiwan, categorical=["SEX", "EDUCATION", "MARRIAGE"],
        notes=["dropped ID", "PAY_0..PAY_6 kept numeric"]),
    "home_credit": DatasetSpec(
        "home_credit", load_home_credit, categorical=None, non_features=["SK_ID_CURR"],
        notes=["application_train only", "SK_ID_CURR kept as non-feature key for later FE joins",
               "DAYS_EMPLOYED 365243 -> NaN"]),
    "hmeq": DatasetSpec("hmeq", load_hmeq, categorical=["REASON", "JOB"]),
    "south_german_credit": DatasetSpec(
        "south_german_credit", load_south_german_credit,
        categorical=["status", "credit_history", "purpose", "savings", "employment_duration",
                     "installment_rate", "personal_status_sex", "other_debtors", "present_residence",
                     "property", "other_installment_plans", "housing", "number_credits", "job",
                     "people_liable", "telephone", "foreign_worker"],
        notes=["used instead of Statlog German", "credit_risk recoded so bad = 1",
               "numeric: duration, amount, age; all other coded group variables categorical"]),
    "fico_heloc": DatasetSpec(
        "fico_heloc", load_fico_heloc, categorical=["MaxDelq2PublicRecLast12M", "MaxDelqEver"],
        notes=["cleaned version, 9,871 rows (588 all -9 rows of heloc_dataset_v1 absent)",
               "-7/-8/-9 -> NaN in every column"]),
    "lending_club_B": DatasetSpec(
        "lending_club_B", load_lending_club_b, categorical=LC_CATEGORICAL, method="time_holdout",
        non_features=["issue_d"], time_col="issue_d", periods=lc_version_b_years,
        notes=["version B (matured window): test = latest issue year with >= 95% of 36-month loans "
               f"finished, val = year before, train = {LC_START_YEAR} up to the year before val "
               "(version A dropped)",
               *LC_NOTES]),
}


def categorical_columns(spec: DatasetSpec, df: pd.DataFrame) -> list:
    if spec.categorical is None:
        return [c for c in df.columns
                if c not in {"row_id", "target", *spec.non_features}
                and not pd.api.types.is_numeric_dtype(df[c])]
    return list(spec.categorical)
