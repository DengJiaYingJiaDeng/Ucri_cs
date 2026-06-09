import pandas as pd

from experiments.protocol1_accepted_only import main


def test_protocol1_experiment_writes_metrics_csv(tmp_path):
    data_path = tmp_path / "accepted.csv"
    output_path = tmp_path / "protocol1_metrics.csv"
    cache_path = tmp_path / "accepted_labeled_rich.pkl"
    rows = []

    for year in [2012, 2013, 2014, 2015, 2016, 2017]:
        for index in range(8):
            default = index % 2
            rows.append(
                {
                    "loan_status": "Charged Off" if default else "Fully Paid",
                    "issue_d": f"{year}-0{(index % 9) + 1}",
                    "loan_amnt": 5000 + index * 500 + (year - 2012) * 100,
                    "dti": 8 + index * 2 + (year - 2012),
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

    pd.DataFrame(rows).to_csv(data_path, index=False)

    result_frame = main(
        str(data_path),
        str(output_path),
        model_names=["LogisticRegression"],
        cache_path=str(cache_path),
    )

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert cache_path.exists()
    assert result_frame["model"].tolist() == ["LogisticRegression"]
    assert saved["model"].tolist() == ["LogisticRegression"]
    assert {"AUROC", "KS", "Brier", "ECE"}.issubset(saved.columns)
