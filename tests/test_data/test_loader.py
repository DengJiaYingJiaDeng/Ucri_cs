from pathlib import Path

import pandas as pd

from src.data.loader import (
    ACCEPTED_ONLY_FEATURES,
    FORBIDDEN_FEATURES,
    compute_file_checksum,
    load_accepted,
    load_rejected,
    record_data_manifest,
)


FIXTURE_DIR = Path("tests/fixtures")
ACCEPTED_SAMPLE = FIXTURE_DIR / "accepted_sample.csv"
REJECTED_SAMPLE = FIXTURE_DIR / "rejected_sample.csv"

RAW_DATA_DIR = Path("data/raw/lendingclub")
ACCEPTED_RAW = RAW_DATA_DIR / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"
REJECTED_RAW = RAW_DATA_DIR / "rejected_2007_to_2018q4.csv" / "rejected_2007_to_2018Q4.csv"


def test_load_accepted_returns_dataframe():
    df = load_accepted(ACCEPTED_SAMPLE)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "loan_status" in df.columns


def test_load_rejected_returns_dataframe():
    df = load_rejected(REJECTED_SAMPLE)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Risk_Score" in df.columns


def test_raw_loader_preserves_label_and_post_approval_columns():
    df = load_accepted(ACCEPTED_SAMPLE)

    assert "loan_status" in df.columns
    assert "total_pymnt" in df.columns
    assert "recoveries" in df.columns
    assert {"loan_status", "total_pymnt", "recoveries"}.issubset(FORBIDDEN_FEATURES)


def test_accepted_only_feature_list_marks_pricing_artifacts():
    assert ACCEPTED_ONLY_FEATURES == ["int_rate", "installment", "grade", "sub_grade"]


def test_checksum_and_manifest_for_fixture_files():
    checksum = compute_file_checksum(ACCEPTED_SAMPLE)
    manifest = record_data_manifest(ACCEPTED_SAMPLE, REJECTED_SAMPLE)

    assert len(checksum) == 64
    assert manifest["accepted_file"] == str(ACCEPTED_SAMPLE)
    assert manifest["rejected_file"] == str(REJECTED_SAMPLE)
    assert manifest["accepted_sha256"] == checksum
    assert len(manifest["rejected_sha256"]) == 64
    assert "snapshot_date" in manifest


def test_real_lendingclub_raw_files_exist_and_headers_match():
    assert ACCEPTED_RAW.exists()
    assert REJECTED_RAW.exists()

    accepted_header = pd.read_csv(ACCEPTED_RAW, nrows=0, low_memory=False)
    rejected_head = pd.read_csv(REJECTED_RAW, nrows=5, low_memory=False)

    assert {"loan_status", "loan_amnt", "fico_range_low", "total_pymnt", "recoveries"}.issubset(
        accepted_header.columns
    )
    assert list(rejected_head.columns) == [
        "Amount Requested",
        "Application Date",
        "Loan Title",
        "Risk_Score",
        "Debt-To-Income Ratio",
        "Zip Code",
        "State",
        "Employment Length",
        "Policy Code",
    ]
    assert len(rejected_head) > 0
