# Copyright 2026 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared execution helpers for the RCPSP formulation comparison demo."""

from __future__ import annotations

from statistics import mean

from src.demo_enums import SolverType
from src.highs import solve_highs
from src.scip import solve_scip
from src.stride import solve_stride

SOLVER_RUNNERS = {
    SolverType.HIGHS: solve_highs,
    SolverType.SCIP: solve_scip,
    SolverType.STRIDE: solve_stride,
}


def compare_formulations(
    selection: list[str],
    time_limit: float,
    runs: int,
    input_path: str,
    *,
    runners: dict = SOLVER_RUNNERS,
) -> list[dict]:
    """Run selected solver formulations repeatedly and collect run-level results.

    Args:
        selection: List of solver value strings (from ``SolverType``).
        time_limit: Maximum runtime per solver run in seconds.
        runs: Number of repeated runs per solver.
        input_path: Path to the MPS-format problem file.
        runners: Optional mapping of ``SolverType`` → callable used to invoke each
            solver.  Defaults to ``SOLVER_RUNNERS``.  Pass a custom dict to inject
            alternative implementations (e.g. a mock Stride runner in tests or when
            a D-Wave API token is not available).

    Returns:
        A list of result dictionaries, one per (solver, run) pair, each containing
        ``"formulation"``, ``"run"``, ``"status"``, ``"energy"``, ``"ok"``,
        ``"starts"``, and ``"modes"`` keys.
    """
    selected_types = sorted(
        {SolverType(int(value)) for value in selection}, key=lambda entry: entry.value
    )

    rows = []
    for solver_type in selected_types:
        runner = runners[solver_type]
        for run_idx in range(1, runs + 1):
            result = runner(time_limit=time_limit, input_path=input_path)
            rows.append(
                {
                    "formulation": solver_type.label,
                    "run": run_idx,
                    "status": result["status"],
                    "energy": result["energy"],
                    "ok": result["ok"],
                    "starts": result.get("starts", {}),
                    "modes": result.get("modes", {}),
                }
            )

    return rows


def summarize_runs(rows: list[dict]) -> list[dict]:
    """Aggregate run-level result rows into one summary row per formulation.

    Args:
        rows: Run-level result rows as returned by ``compare_formulations``.

    Returns:
        A list of summary dictionaries sorted by formulation name, each containing
        ``"formulation"``, ``"runs"``, ``"ok_runs"``, ``"best_energy"``, and
        ``"avg_energy"`` keys.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["formulation"], []).append(row)

    summary = []
    for formulation, items in groups.items():
        valid_energies = [
            item["energy"] for item in items if isinstance(item["energy"], (int, float))
        ]
        ok_runs = sum(1 for item in items if item["ok"])
        summary.append(
            {
                "formulation": formulation,
                "runs": len(items),
                "ok_runs": ok_runs,
                "best_energy": min(valid_energies) if valid_energies else "n/a",
                "avg_energy": round(mean(valid_energies), 4) if valid_energies else "n/a",
            }
        )

    return sorted(summary, key=lambda row: row["formulation"])
