"""Convert every raw dataset in data/raw into a tabular file in data/interim.

Only format conversion and label standardisation happen here:
  * small ARFF / ASC files  -> CSV
  * large Home Credit / Lending Club tables -> Parquet
  * each labelled table gets a `target` column (1 = bad / default, 0 = good);
    the original label column is dropped so it cannot leak into features.

No imputation, outlier handling or feature engineering is done at this stage.

Run from the project root:  python -m src.data.make_interim [name ...]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

SOUTH_GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount", "savings",
    "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age",
    "other_installment_plans", "housing", "number_credits", "job",
    "people_liable", "telephone", "foreign_worker", "credit_risk",
]

LC_BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
LC_GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


def read_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF reader: handles % comments, STRING/nominal attributes, '?' as NA."""
    names, data_line = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            low = s.lower()
            if low.startswith("@attribute"):
                m = re.match(r"@attribute\s+('([^']*)'|\"([^\"]*)\"|(\S+))", s, re.I)
                names.append(m.group(2) or m.group(3) or m.group(4))
            elif low.startswith("@data"):
                data_line = i + 1
                break
    df = pd.read_csv(
        path, skiprows=data_line, header=None, names=names, na_values=["?"],
        quotechar="'", skipinitialspace=True, comment="%", skip_blank_lines=True,
    )
    # nominal values are sometimes quoted with extra whitespace
    for c in df.select_dtypes("object"):
        df[c] = df[c].str.strip()
    return df


def finish(df: pd.DataFrame, label: str, target: pd.Series) -> pd.DataFrame:
    df = df.drop(columns=[label])
    df["target"] = target.astype("Int8")
    return df


def write_csv(df: pd.DataFrame, name: str) -> None:
    out = INTERIM / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"  {out.relative_to(ROOT)}: {df.shape[0]:,} rows x {df.shape[1]} cols")


# --------------------------------------------------------------------- small

def hmeq():
    df = read_arff(RAW / "HMEQ")
    write_csv(finish(df, "BAD", df["BAD"]), "hmeq")


def fico_heloc():
    df = read_arff(RAW / "fico_heloc")
    write_csv(finish(df, "RiskPerformance", (df["RiskPerformance"] == "Bad").astype(int)), "fico_heloc")


def taiwan():
    df = read_arff(RAW / "taiwan_default_credit")
    float_cols = [c for c in df.columns if c.startswith(("LIMIT_BAL", "BILL_AMT", "PAY_AMT"))]
    int_cols = [c for c in df.columns if c not in float_cols]
    df[int_cols] = df[int_cols].astype(int)
    label = "default.payment.next.month"
    write_csv(finish(df, label, df[label]), "taiwan_default_credit")


def give_me_some_credit():
    df = read_arff(RAW / "give_me_some_credit")
    write_csv(finish(df, "SeriousDlqin2yrs", df["SeriousDlqin2yrs"]), "give_me_some_credit")


def german_credit():
    df = read_arff(RAW / "german_credit.arff")
    write_csv(finish(df, "class", (df["class"] == "bad").astype(int)), "german_credit")


def south_german_credit():
    df = pd.read_csv(RAW / "south_german_credit" / "SouthGermanCredit.asc", sep=r"\s+")
    df.columns = SOUTH_GERMAN_COLUMNS
    # codetable: credit_risk 0 = bad, 1 = good
    write_csv(finish(df, "credit_risk", 1 - df["credit_risk"]), "south_german_credit")


# --------------------------------------------------------------------- large

def csv_to_parquet(src: Path, dst: Path, **read_kw) -> pa.Table:
    """Read a CSV with pyarrow; columns that fail type inference fall back to string."""
    col_types = {}
    while True:
        try:
            table = pacsv.read_csv(
                src,
                read_options=pacsv.ReadOptions(block_size=64 << 20),
                convert_options=pacsv.ConvertOptions(column_types=col_types, strings_can_be_null=True),
                parse_options=pacsv.ParseOptions(**read_kw),
            )
            break
        except pa.ArrowInvalid as e:
            m = re.search(r"In CSV column #(\d+)", str(e))
            if not m:
                raise
            with pacsv.open_csv(src) as r:
                col = r.schema.names[int(m.group(1))]
            col_types[col] = pa.string() if col in col_types else pa.float64()
            print(f"    retype {col} -> {col_types[col]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression="zstd")
    print(f"  {dst.relative_to(ROOT)}: {table.num_rows:,} rows x {table.num_columns} cols")
    return table


def home_credit():
    src_dir, out_dir = RAW / "home_credit", INTERIM / "home_credit"
    for src in sorted(src_dir.glob("*.csv")):
        if src.name in {"sample_submission.csv", "HomeCredit_columns_description.csv"}:
            continue
        dst = out_dir / (src.stem + ".parquet")
        table = csv_to_parquet(src, dst)
        if src.name == "application_train.csv":
            df = table.to_pandas()
            df = finish(df, "TARGET", df["TARGET"])
            df.to_parquet(dst, compression="zstd", index=False)
    # description file keeps its awkward encoding -> copy as utf-8 csv
    desc = pd.read_csv(src_dir / "HomeCredit_columns_description.csv", encoding="latin-1", index_col=0)
    desc.to_csv(out_dir / "columns_description.csv", index=False)


def _skip_short_rows(row):
    # Lending Club appends "Total amount funded ..." footer lines with too few fields
    return "skip"


def lending_club():
    src_dir, out_dir = RAW / "lending_club", INTERIM / "lending_club"
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = src_dir / "accepted_2007_to_2018Q4.csv.gz"
    # Read everything as string first (151 sparse columns defeat block-wise type inference),
    # then cast numeric-looking columns.
    with pacsv.open_csv(acc, read_options=pacsv.ReadOptions(block_size=1 << 20)) as r:
        names = r.schema.names
    table = pacsv.read_csv(
        acc,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types={n: pa.string() for n in names},
                                             strings_can_be_null=True),
        parse_options=pacsv.ParseOptions(invalid_row_handler=_skip_short_rows),
    )
    import pyarrow.compute as pc

    table = table.filter(pc.is_valid(table["loan_amnt"]))  # drop footer rows
    # cast string columns that are fully numeric; strip "%" / "months" style suffixes are left as text
    cols = []
    for name in table.column_names:
        col = table[name]
        try:
            col = pc.cast(col, pa.float64())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            pass
        cols.append(col)
    table = pa.table(cols, names=table.column_names)
    status = table["loan_status"]
    target = pc.if_else(pc.is_in(status, pa.array(sorted(LC_BAD))), 1,
                        pc.if_else(pc.is_in(status, pa.array(sorted(LC_GOOD))), 0, None))
    # null target = loan still running (Current / Late / In Grace Period)
    table = table.append_column("target", pc.cast(target, pa.int8()))
    dst = out_dir / "accepted.parquet"
    pq.write_table(table, dst, compression="zstd")
    print(f"  {dst.relative_to(ROOT)}: {table.num_rows:,} rows x {table.num_columns} cols")
    del table

    csv_to_parquet(src_dir / "rejected_2007_to_2018Q4.csv.gz", out_dir / "rejected.parquet")


DATASETS = {
    "hmeq": hmeq,
    "fico_heloc": fico_heloc,
    "taiwan": taiwan,
    "give_me_some_credit": give_me_some_credit,
    "german_credit": german_credit,
    "south_german_credit": south_german_credit,
    "home_credit": home_credit,
    "lending_club": lending_club,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help=f"datasets to convert (default: all): {', '.join(DATASETS)}")
    args = p.parse_args(argv)
    unknown = set(args.names) - set(DATASETS)
    if unknown:
        p.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    INTERIM.mkdir(parents=True, exist_ok=True)
    for name in args.names or DATASETS:
        print(f"[{name}]")
        DATASETS[name]()


if __name__ == "__main__":
    sys.exit(main())
