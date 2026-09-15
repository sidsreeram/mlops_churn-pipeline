import json
import os
import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import DriftedColumnsCount
from evidently.tests import lt


def load_datasets():
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    reference_df = pd.read_csv(train_path)
    current_df = pd.read_csv(test_path).copy()

    np.random.seed(42)
    drift_noise = np.random.normal(loc=15.0, scale=5.0, size=len(current_df))
    current_df["MonthlyCharges"] = (current_df["MonthlyCharges"] + drift_noise).clip(
        lower=18.0
    )

    return reference_df, current_df


def _collect_test_results(node, found):
    """Recursively pull individual test-result dicts out of the report dict."""
    if isinstance(node, dict):
        if "status" in node and ("name" in node or "id" in node):
            found.append(node)
        for v in node.values():
            _collect_test_results(v, found)
    elif isinstance(node, list):
        for item in node:
            _collect_test_results(item, found)


def run_drift_monitoring(
    reports_dir: str = "reports",
    drift_threshold_share: float = 0.25,
    max_drifted_columns: int = 5,
):
    os.makedirs(reports_dir, exist_ok=True)
    reference_df, current_df = load_datasets()

    print("Running Evidently Data Drift evaluation...")

    # Report + tests are unified now — no separate TestSuite object.
    # DriftedColumnsCount exposes both a count and a share; `tests=` gates
    # the count, `share_tests=` gates the share.
    report = Report(
        [
            DataDriftPreset(),
            DriftedColumnsCount(
                tests=[lt(max_drifted_columns)],
                share_tests=[lt(drift_threshold_share)],
            ),
        ]
    )

    # NOTE: positional order is (current_data, reference_data)
    my_eval = report.run(current_df, reference_df)

    html_path = os.path.join(reports_dir, "drift_report.html")
    my_eval.save_html(html_path)
    print(f"Visual drift dashboard generated: {html_path}")

    result = my_eval.dict()
    test_results = []
    _collect_test_results(result, test_results)

    total_tests = len(test_results)
    failed = [
        t for t in test_results
        if str(t.get("status", "")).upper() not in ("SUCCESS", "PASS", "PASSED")
    ]
    all_passed = total_tests > 0 and len(failed) == 0

    summary = {
        "total_tests": total_tests,
        "failed_tests": len(failed),
        "all_passed": all_passed,
        "failed_details": failed,
    }

    json_path = os.path.join(reports_dir, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n--- Drift Test Results ---")
    print(f"Total Tests: {total_tests} | Failed: {len(failed)}")

    if not all_passed:
        print("\n[ALERT] Data drift threshold exceeded! Upstream retrain required.")
        print(
            "Recommended action: Trigger Prefect pipeline ('python src/pipeline.py') on new batch."
        )
    else:
        print("\n[STABLE] No significant feature drift detected.")


if __name__ == "__main__":
    run_drift_monitoring()