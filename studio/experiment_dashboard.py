"""Experiment dashboard backed by a SQLite document-style result store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class ExperimentDashboard:
    """Logs experiment runs and compares them in tabular form."""

    db_path: Path = Path("studio/experiments.db")

    def __post_init__(self) -> None:
        """Ensure the SQLite document store table exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_experiment(self, name: str, config: dict, scores: dict[str, float]) -> None:
        """Persist one experiment record to the SQLite store."""
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiments (name, config_json, scores_json, created_at) VALUES (?, ?, ?, ?)",
                (name, json.dumps(config), json.dumps(scores), timestamp),
            )
            conn.commit()

    def compare_experiments(self, names: list[str]) -> pd.DataFrame:
        """Return a DataFrame comparing score fields for selected experiments."""
        if not names:
            return pd.DataFrame(columns=["name", "created_at"])

        placeholders = ",".join("?" for _ in names)
        query = (
            "SELECT name, config_json, scores_json, created_at "
            f"FROM experiments WHERE name IN ({placeholders}) ORDER BY created_at ASC"
        )

        rows: list[dict] = []
        with self._connect() as conn:
            for name, config_json, scores_json, created_at in conn.execute(query, names):
                config = json.loads(config_json)
                scores = json.loads(scores_json)
                row = {
                    "name": name,
                    "created_at": created_at,
                    "config": config,
                }
                for score_name, score_value in scores.items():
                    row[str(score_name)] = float(score_value)
                rows.append(row)

        return pd.DataFrame(rows)

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection to the experiment store."""
        return sqlite3.connect(self.db_path)
