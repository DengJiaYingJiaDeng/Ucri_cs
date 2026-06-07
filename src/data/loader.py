from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_FEATURES = [
    # Post-approval payment fields.
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    # Collection and recovery fields.
    "recoveries",
    "collection_recovery_fee",
    # Post-approval date and amount fields.
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    # Target/proxy labels and settlement fields.
    "loan_status",
    "hardship_flag",
    "debt_settlement_flag",
    "settlement_status",
    # Post-approval balances.
    "out_prncp",
    "out_prncp_inv",
    # Delinquency fields that require time-window auditing before use.
    "acc_now_delinq",
    "delinq_amnt",
]

# These are accepted-only pricing or policy artifacts. They are excluded from
# the shared-feature view, but can be used in accepted-rich baselines.
ACCEPTED_ONLY_FEATURES = ["int_rate", "installment", "grade", "sub_grade"]


def load_accepted(path: str | Path) -> pd.DataFrame:
    """Load LendingClub accepted loan records from a raw CSV file."""
    return pd.read_csv(path, low_memory=False)


def load_rejected(path: str | Path) -> pd.DataFrame:
    """Load LendingClub rejected application records from a raw CSV file.

    Rejected records are unlabeled applications that reached the public
    LendingClub rejected-file record. They are not the full universe of denied,
    pre-screened, withdrawn, or abandoned applications.
    """
    return pd.read_csv(path, low_memory=False)


def compute_file_checksum(path: str | Path) -> str:
    """Compute the SHA256 checksum for a raw data file."""
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def record_data_manifest(accepted_path: str | Path, rejected_path: str | Path) -> dict[str, Any]:
    """Record raw data paths, checksums, and snapshot metadata."""
    accepted = Path(accepted_path)
    rejected = Path(rejected_path)
    return {
        "accepted_file": str(accepted),
        "accepted_sha256": compute_file_checksum(accepted),
        "rejected_file": str(rejected),
        "rejected_sha256": compute_file_checksum(rejected),
        "snapshot_date": datetime.now().isoformat(),
    }
