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

"""Tests for src/plot.py."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from src.plot import (
    _choose_business_mode,
    _compute_demand,
    _earliest_start_schedule,
    build_comparison_graph,
    build_input_graph,
    build_solution_graph,
    parse_mps_structure,
)

# ---------------------------------------------------------------------------
# parse_mps_structure
# ---------------------------------------------------------------------------


class TestParseMpsStructure:
    def test_missing_file_returns_empty(self):
        result = parse_mps_structure("nonexistent/path/to/file.mps")
        assert result["jobs"] == []
        assert result["edges"] == []
        assert result["durations"] == {}
        assert result["mechanic_use"] == {}
        assert result["technician_use"] == {}
        assert result["capacities"]["Mechaniker"] == 0.0
        assert result["capacities"]["Techniker"] == 0.0

    def test_real_file_has_30_jobs(self, mps_path):
        result = parse_mps_structure(mps_path)
        assert len(result["jobs"]) == 30

    def test_real_file_has_precedences(self, mps_path):
        result = parse_mps_structure(mps_path)
        assert len(result["edges"]) > 0

    def test_real_file_has_durations(self, mps_path):
        result = parse_mps_structure(mps_path)
        assert len(result["durations"]) > 0

    def test_real_file_resource_usage_non_negative(self, mps_path):
        result = parse_mps_structure(mps_path)
        for v in result["mechanic_use"].values():
            assert v >= 0
        for v in result["technician_use"].values():
            assert v >= 0


# ---------------------------------------------------------------------------
# _choose_business_mode
# ---------------------------------------------------------------------------


class TestChooseBusinessMode:
    def test_single_mode_returns_that_mode(self, simple_profile):
        assert _choose_business_mode(1, simple_profile) == 1
        assert _choose_business_mode(2, simple_profile) == 1
        assert _choose_business_mode(3, simple_profile) == 1

    def test_picks_shortest_duration(self, multi_mode_profile):
        # Job 1: mode 1 → duration 10, mode 2 → duration 4 — should pick mode 2
        result = _choose_business_mode(1, multi_mode_profile)
        assert result == 2

    def test_tie_broken_by_resource_then_mode(self):
        profile = {
            "jobs": [1],
            "edges": [],
            "durations": {(1, 1): 5, (1, 2): 5},
            "mechanic_use": {(1, 1): 3.0, (1, 2): 1.0},
            "technician_use": {(1, 1): 0.0, (1, 2): 0.0},
            "capacities": {"Mechaniker": 5.0, "Techniker": 0.0},
        }
        # Same duration, mode 2 uses fewer resources → mode 2
        assert _choose_business_mode(1, profile) == 2

    def test_unknown_job_returns_one(self, simple_profile):
        assert _choose_business_mode(99, simple_profile) == 1


# ---------------------------------------------------------------------------
# _earliest_start_schedule
# ---------------------------------------------------------------------------


class TestEarliestStartSchedule:
    def test_chain_topology(self, simple_profile):
        starts, durations, modes = _earliest_start_schedule(simple_profile)
        # Job 1 starts at 0; job 2 after job 1 finishes; job 3 after job 2 finishes
        assert starts[1] == 0
        assert starts[2] == durations[1]
        assert starts[3] == durations[1] + durations[2]

    def test_all_jobs_present(self, simple_profile):
        starts, durations, modes = _earliest_start_schedule(simple_profile)
        assert set(starts.keys()) == {1, 2, 3}
        assert set(durations.keys()) == {1, 2, 3}
        assert set(modes.keys()) == {1, 2, 3}

    def test_durations_at_least_one(self, simple_profile):
        _, durations, _ = _earliest_start_schedule(simple_profile)
        for d in durations.values():
            assert d >= 1

    def test_no_edges_all_start_at_zero(self):
        profile = {
            "jobs": [1, 2, 3],
            "edges": [],
            "durations": {(1, 1): 5, (2, 1): 3, (3, 1): 4},
            "mechanic_use": {(1, 1): 1.0, (2, 1): 1.0, (3, 1): 1.0},
            "technician_use": {(1, 1): 0.0, (2, 1): 0.0, (3, 1): 0.0},
            "capacities": {"Mechaniker": 3.0, "Techniker": 0.0},
        }
        starts, _, _ = _earliest_start_schedule(profile)
        for s in starts.values():
            assert s == 0


# ---------------------------------------------------------------------------
# _compute_demand
# ---------------------------------------------------------------------------


class TestComputeDemand:
    def test_demand_length_equals_horizon(self, simple_profile):
        starts = {1: 0, 2: 5, 3: 8}
        modes = {1: 1, 2: 1, 3: 1}
        mech, tech, horizon = _compute_demand(simple_profile, starts, modes)
        assert len(mech) == horizon
        assert len(tech) == horizon

    def test_demand_non_negative(self, simple_profile):
        starts = {1: 0, 2: 5, 3: 8}
        modes = {1: 1, 2: 1, 3: 1}
        mech, tech, _ = _compute_demand(simple_profile, starts, modes)
        assert all(v >= 0 for v in mech)
        assert all(v >= 0 for v in tech)

    def test_mechanic_demand_in_job1_slot(self, simple_profile):
        # job 1 uses 2 mechanics from t=0 to t=4
        starts = {1: 0, 2: 5, 3: 8}
        modes = {1: 1, 2: 1, 3: 1}
        mech, _, _ = _compute_demand(simple_profile, starts, modes)
        assert mech[0] == pytest.approx(2.0)
        assert mech[4] == pytest.approx(2.0)
        assert mech[5] == pytest.approx(0.0)  # job 1 done, job 2 uses 0 mechanics

    def test_fallback_mode_used_when_missing(self, simple_profile):
        # Passing empty modes dict should fall back without error
        starts = {1: 0, 2: 5, 3: 8}
        mech, tech, horizon = _compute_demand(simple_profile, starts, {})
        assert horizon > 0
        assert len(mech) == horizon


# ---------------------------------------------------------------------------
# build_input_graph
# ---------------------------------------------------------------------------


class TestBuildInputGraph:
    def test_returns_figure(self, mps_path):
        fig = build_input_graph(mps_path)
        assert isinstance(fig, go.Figure)

    def test_empty_path_returns_figure(self):
        fig = build_input_graph("")
        assert isinstance(fig, go.Figure)

    def test_has_two_subplots(self, mps_path):
        fig = build_input_graph(mps_path)
        # Two subplots means at least 2 axes in layout
        layout_keys = list(fig.layout.to_plotly_json().keys())
        yaxis_keys = [k for k in layout_keys if k.startswith("yaxis")]
        assert len(yaxis_keys) >= 2

    def test_has_traces(self, mps_path):
        fig = build_input_graph(mps_path)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# build_comparison_graph
# ---------------------------------------------------------------------------


class TestBuildComparisonGraph:
    def test_empty_schedules_returns_figure(self, mps_path):
        fig = build_comparison_graph(mps_path, {})
        assert isinstance(fig, go.Figure)

    def test_empty_schedules_has_message(self, mps_path):
        fig = build_comparison_graph(mps_path, {})
        title_text = fig.layout.title.text or ""
        assert "compare" in title_text.lower() or "no" in title_text.lower()

    def test_with_one_solver(self, mps_path):
        # Use a trivial schedule: all jobs start at 0 in mode 1
        starts = {j: 0 for j in range(1, 31)}
        modes = {j: 1 for j in range(1, 31)}
        fig = build_comparison_graph(mps_path, {"HiGHS": (starts, modes)})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_with_multiple_solvers(self, mps_path):
        starts = {j: 0 for j in range(1, 31)}
        modes = {j: 1 for j in range(1, 31)}
        schedules = {
            "HiGHS": (starts, modes),
            "SCIP": (starts, modes),
        }
        fig = build_comparison_graph(mps_path, schedules)
        solver_names = {trace.name for trace in fig.data}
        assert "HiGHS" in solver_names
        assert "SCIP" in solver_names


# ---------------------------------------------------------------------------
# build_solution_graph
# ---------------------------------------------------------------------------


class TestBuildSolutionGraph:
    def test_returns_figure_with_solution(self, mps_path):
        starts = {j: 0 for j in range(1, 31)}
        modes = {j: 1 for j in range(1, 31)}
        fig = build_solution_graph(mps_path, starts, modes, "Test Title")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_without_solution(self, mps_path):
        fig = build_solution_graph(mps_path, None, None, "No Solution")
        assert isinstance(fig, go.Figure)

    def test_fallback_title_when_no_solution(self, mps_path):
        fig = build_solution_graph(mps_path, None, None, "MySolver")
        title_text = fig.layout.title.text or ""
        assert "MySolver" in title_text

    def test_has_traces(self, mps_path):
        starts = {j: j for j in range(1, 31)}
        modes = {j: 1 for j in range(1, 31)}
        fig = build_solution_graph(mps_path, starts, modes, "Test")
        assert len(fig.data) > 0
