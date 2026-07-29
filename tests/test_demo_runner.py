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

"""Tests for src/demo_runner.py."""

from __future__ import annotations

import pytest

from src.demo_enums import SolverType
from src.demo_runner import compare_formulations, summarize_runs

# ---------------------------------------------------------------------------
# Shared fake solver outputs
# ---------------------------------------------------------------------------

FAKE_HIGHS_RESULT = {
    "solver": "HiGHS (MILP)",
    "status": "Optimal",
    "energy": 302.0,
    "ok": True,
    "starts": {1: 0, 2: 5},
    "modes": {1: 1, 2: 2},
}

FAKE_SCIP_RESULT = {
    "solver": "SCIP (MILP)",
    "status": "optimal",
    "energy": 310.0,
    "ok": True,
    "starts": {1: 0, 2: 6},
    "modes": {1: 2, 2: 1},
}

FAKE_FAIL_RESULT = {
    "solver": "HiGHS (MILP)",
    "status": "Infeasible",
    "energy": None,
    "ok": False,
    "starts": {},
    "modes": {},
}


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_single_solver_single_run(self):
        rows = compare_formulations(
            [str(SolverType.HIGHS.value)], time_limit=5.0, runs=1, input_path="x.mps",
            runners={SolverType.HIGHS: lambda *a, **kw: FAKE_HIGHS_RESULT},
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["formulation"] == SolverType.HIGHS.label
        assert row["run"] == 1
        assert row["energy"] == 302.0
        assert row["ok"] is True

    def test_multiple_runs(self):
        rows = compare_formulations(
            [str(SolverType.HIGHS.value)], time_limit=5.0, runs=3, input_path="x.mps",
            runners={SolverType.HIGHS: lambda *a, **kw: FAKE_HIGHS_RESULT},
        )
        assert len(rows) == 3
        assert [r["run"] for r in rows] == [1, 2, 3]

    def test_multiple_solvers(self):
        rows = compare_formulations(
            [str(SolverType.HIGHS.value), str(SolverType.SCIP.value)],
            time_limit=5.0,
            runs=1,
            input_path="x.mps",
            runners={
                SolverType.HIGHS: lambda *a, **kw: FAKE_HIGHS_RESULT,
                SolverType.SCIP: lambda *a, **kw: FAKE_SCIP_RESULT,
            },
        )
        assert len(rows) == 2
        formulations = {r["formulation"] for r in rows}
        assert SolverType.HIGHS.label in formulations
        assert SolverType.SCIP.label in formulations

    def test_row_contains_starts_and_modes(self):
        rows = compare_formulations(
            [str(SolverType.HIGHS.value)], time_limit=5.0, runs=1, input_path="x.mps",
            runners={SolverType.HIGHS: lambda *a, **kw: FAKE_HIGHS_RESULT},
        )
        assert rows[0]["starts"] == FAKE_HIGHS_RESULT["starts"]
        assert rows[0]["modes"] == FAKE_HIGHS_RESULT["modes"]

    def test_row_required_keys(self):
        rows = compare_formulations(
            [str(SolverType.HIGHS.value)], time_limit=5.0, runs=1, input_path="x.mps",
            runners={SolverType.HIGHS: lambda *a, **kw: FAKE_HIGHS_RESULT},
        )
        for key in ("formulation", "run", "status", "energy", "ok", "starts", "modes"):
            assert key in rows[0]

    def test_solvers_sorted_by_value(self):
        """SCIP (1) should appear before Stride (2) in the output."""
        rows = compare_formulations(
            [str(SolverType.STRIDE.value), str(SolverType.SCIP.value)],
            time_limit=5.0,
            runs=1,
            input_path="x.mps",
            runners={
                SolverType.SCIP: lambda *a, **kw: FAKE_SCIP_RESULT,
                SolverType.STRIDE: lambda *a, **kw: {**FAKE_FAIL_RESULT, "solver": "Stride"},
            },
        )
        assert rows[0]["formulation"] == SolverType.SCIP.label
        assert rows[1]["formulation"] == SolverType.STRIDE.label


# ---------------------------------------------------------------------------
# summarize_runs
# ---------------------------------------------------------------------------


class TestSummarizeRuns:
    def _make_rows(self, energies: list[float | None], ok_flags: list[bool]) -> list[dict]:
        return [
            {
                "formulation": "HiGHS (MILP)",
                "run": i + 1,
                "status": "ok",
                "energy": e,
                "ok": ok,
                "starts": {},
                "modes": {},
            }
            for i, (e, ok) in enumerate(zip(energies, ok_flags))
        ]

    def test_best_energy_is_minimum(self):
        rows = self._make_rows([310.0, 302.0, 320.0], [True, True, True])
        summary = summarize_runs(rows)
        assert summary[0]["best_energy"] == 302.0

    def test_avg_energy_rounded(self):
        rows = self._make_rows([300.0, 310.0], [True, True])
        summary = summarize_runs(rows)
        assert summary[0]["avg_energy"] == pytest.approx(305.0)

    def test_ok_runs_count(self):
        rows = self._make_rows([300.0, None, 310.0], [True, False, True])
        summary = summarize_runs(rows)
        assert summary[0]["ok_runs"] == 2

    def test_all_failed_gives_na(self):
        rows = self._make_rows([None, None], [False, False])
        summary = summarize_runs(rows)
        assert summary[0]["best_energy"] == "n/a"
        assert summary[0]["avg_energy"] == "n/a"

    def test_multiple_formulations_sorted(self):
        rows = [
            {
                "formulation": "SCIP (MILP)",
                "run": 1,
                "status": "ok",
                "energy": 302.0,
                "ok": True,
                "starts": {},
                "modes": {},
            },
            {
                "formulation": "HiGHS (MILP)",
                "run": 1,
                "status": "ok",
                "energy": 310.0,
                "ok": True,
                "starts": {},
                "modes": {},
            },
        ]
        summary = summarize_runs(rows)
        names = [s["formulation"] for s in summary]
        assert names == sorted(names)

    def test_empty_input(self):
        summary = summarize_runs([])
        assert summary == []

    def test_runs_count(self):
        rows = self._make_rows([302.0, 310.0, 320.0], [True, True, False])
        summary = summarize_runs(rows)
        assert summary[0]["runs"] == 3
