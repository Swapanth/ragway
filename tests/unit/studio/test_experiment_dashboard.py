from __future__ import annotations

from pathlib import Path

import pandas as pd

from studio.experiment_dashboard import ExperimentDashboard


def test_experiment_dashboard_logs_and_compares(tmp_path: Path) -> None:
    """ExperimentDashboard should store rows and return comparison DataFrame."""
    db_path = tmp_path / "experiments.db"
    dashboard = ExperimentDashboard(db_path=db_path)

    dashboard.log_experiment(
        name="naive",
        config={"top_k": 5},
        scores={"faithfulness": 0.8, "overall_score": 0.75},
    )
    dashboard.log_experiment(
        name="hybrid",
        config={"top_k": 10},
        scores={"faithfulness": 0.85, "overall_score": 0.8},
    )

    df = dashboard.compare_experiments(["naive", "hybrid"])

    assert isinstance(df, pd.DataFrame)
    assert set(df["name"].tolist()) == {"naive", "hybrid"}
    assert "overall_score" in df.columns
