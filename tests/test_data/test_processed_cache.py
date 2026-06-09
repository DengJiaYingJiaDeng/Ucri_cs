import pandas as pd
import pytest

from src.data.processed_cache import load_or_build_accepted_labeled_rich


def make_accepted_file(path, rows_per_year=2):
    rows = []
    for year in [2012, 2013, 2014, 2015, 2016, 2017]:
        for index in range(rows_per_year):
            rows.append(
                {
                    "loan_status": "Charged Off" if index % 2 else "Fully Paid",
                    "issue_d": f"{year}-0{(index % 9) + 1}",
                    "loan_amnt": 5000 + index * 500,
                    "dti": 8 + index,
                    "emp_length": index,
                    "annual_inc": 45000 + index * 3000,
                    "fico_range_low": 650 + index * 5,
                    "fico_range_high": 654 + index * 5,
                    "open_acc": 5 + index,
                    "revol_bal": 1000 + index * 200,
                    "revol_util": 10 + index * 3,
                    "total_acc": 12 + index,
                    "int_rate": 8.0 + index,
                    "installment": 150.0 + index * 10,
                    "delinq_2yrs": index % 3,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_or_build_accepted_labeled_rich_creates_and_reuses_cache(tmp_path):
    data_path = make_accepted_file(tmp_path / "accepted.csv", rows_per_year=2)
    cache_path = tmp_path / "accepted_labeled_rich.pkl"

    first_frame, first_hit = load_or_build_accepted_labeled_rich(data_path, cache_path=cache_path)
    second_frame, second_hit = load_or_build_accepted_labeled_rich(data_path, cache_path=cache_path)

    assert first_hit is False
    assert second_hit is True
    assert cache_path.exists()
    assert {"application_date", "default_label", "issue_d", "loan_amnt", "fico_avg"}.issubset(first_frame.columns)
    pd.testing.assert_frame_equal(first_frame, second_frame)


def test_load_or_build_accepted_labeled_rich_refreshes_cache(tmp_path):
    data_path = make_accepted_file(tmp_path / "accepted.csv", rows_per_year=1)
    cache_path = tmp_path / "accepted_labeled_rich.pkl"

    first_frame, _ = load_or_build_accepted_labeled_rich(data_path, cache_path=cache_path)
    make_accepted_file(data_path, rows_per_year=3)
    cached_frame, cached_hit = load_or_build_accepted_labeled_rich(data_path, cache_path=cache_path)
    refreshed_frame, refreshed_hit = load_or_build_accepted_labeled_rich(
        data_path,
        cache_path=cache_path,
        refresh_cache=True,
    )

    assert cached_hit is True
    assert refreshed_hit is False
    assert len(cached_frame) == len(first_frame)
    assert len(refreshed_frame) > len(first_frame)


def test_load_or_build_accepted_labeled_rich_validates_cached_columns(tmp_path):
    data_path = make_accepted_file(tmp_path / "accepted.csv")
    cache_path = tmp_path / "broken.pkl"
    pd.DataFrame({"loan_amnt": [1000]}).to_pickle(cache_path)

    with pytest.raises(ValueError, match="missing required columns"):
        load_or_build_accepted_labeled_rich(data_path, cache_path=cache_path)
