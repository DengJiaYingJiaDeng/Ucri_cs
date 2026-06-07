import pandas as pd
import pytest

from src.data.splitter import split_accepted_rejected, time_split


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "application_date": [
                "2012-06",
                "2013-01",
                "2014-06",
                "2015-03",
                "2015-11",
                "2016-02",
                "2016-08",
                "2017-05",
                "2018-01",
                "2019-06",
                "2020-03",
            ],
            "loan_amount": [10000] * 11,
            "default_label": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        }
    )


def test_time_split_train_range(sample_data):
    splits = time_split(sample_data)

    train_years = set(splits["train"]["application_date"].str[:4])

    assert train_years <= {"2012", "2013", "2014"}


def test_time_split_validation_range(sample_data):
    splits = time_split(sample_data)

    validation_years = set(splits["validation"]["application_date"].str[:4])

    assert validation_years <= {"2015"}


def test_time_split_test_normal_range(sample_data):
    splits = time_split(sample_data)

    test_years = set(splits["test_normal"]["application_date"].str[:4])

    assert test_years <= {"2016", "2017"}


def test_time_split_no_overlap(sample_data):
    splits = time_split(sample_data)
    train_idx = set(splits["train"].index)
    validation_idx = set(splits["validation"].index)
    test_idx = set(splits["test_normal"].index)

    assert train_idx.isdisjoint(validation_idx)
    assert train_idx.isdisjoint(test_idx)
    assert validation_idx.isdisjoint(test_idx)


def test_time_split_test_extended(sample_data):
    splits = time_split(sample_data, include_extended=True)

    assert set(splits["test_extended"]["application_date"].str[:4]) == {"2018", "2019"}


def test_time_split_can_exclude_extended_period(sample_data):
    splits = time_split(sample_data, include_extended=False)

    assert "test_extended" not in splits
    assert "test_structural_break" in splits


def test_time_split_structural_break(sample_data):
    splits = time_split(sample_data)

    assert set(splits["test_structural_break"]["application_date"].str[:4]) == {"2020"}


def test_time_split_preserves_original_date_values_and_dtype(sample_data):
    splits = time_split(sample_data)

    assert splits["train"]["application_date"].tolist() == ["2012-06", "2013-01", "2014-06"]
    assert splits["train"]["application_date"].dtype == sample_data["application_date"].dtype


def test_time_split_custom_date_column():
    df = pd.DataFrame({"issue_d": ["2015-01", "2016-01"], "loan_amount": [10000, 20000]})

    splits = time_split(df, date_col="issue_d")

    assert len(splits["validation"]) == 1
    assert len(splits["test_normal"]) == 1


def test_time_split_missing_date_column_raises(sample_data):
    with pytest.raises(KeyError, match="missing_date"):
        time_split(sample_data, date_col="missing_date")


def test_split_accepted_rejected_aligns_periods():
    accepted = pd.DataFrame({"application_date": ["2012-06", "2015-03", "2020-03"], "id": [1, 2, 3]})
    rejected = pd.DataFrame({"application_date": ["2013-01", "2018-01"], "id": [10, 20]})

    splits = split_accepted_rejected(accepted, rejected)

    assert set(splits) == {"train", "validation", "test_normal", "test_extended", "test_structural_break"}
    assert splits["train"]["accepted"]["id"].tolist() == [1]
    assert splits["train"]["rejected"]["id"].tolist() == [10]
    assert splits["test_extended"]["rejected"]["id"].tolist() == [20]
