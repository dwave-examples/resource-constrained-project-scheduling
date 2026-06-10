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

"""Tests for callback functions in demo_callbacks.py.

Callbacks are pure Python functions decorated with @dash.callback.  They can be
called directly in tests.  Dash context objects (``ctx``) are patched where the
callback inspects ``ctx.triggered_id``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dash
import plotly.graph_objects as go
import pytest

import demo_callbacks as cb
from src.demo_enums import SolverType

# ---------------------------------------------------------------------------
# toggle_left_column
# ---------------------------------------------------------------------------


class TestToggleLeftColumn:
    def test_collapses_when_not_collapsed(self):
        class_name, aria = cb.toggle_left_column(1, "sidebar")
        assert "collapsed" in class_name
        assert aria == "false"

    def test_expands_when_collapsed(self):
        class_name, aria = cb.toggle_left_column(1, "sidebar collapsed")
        assert "collapsed" not in class_name
        assert aria == "true"

    def test_empty_class_becomes_collapsed(self):
        class_name, aria = cb.toggle_left_column(1, "")
        assert class_name == "collapsed"
        assert aria == "false"

    def test_none_class_becomes_collapsed(self):
        class_name, aria = cb.toggle_left_column(1, None)
        assert class_name == "collapsed"
        assert aria == "false"

    def test_preserves_other_classes_on_collapse(self):
        class_name, _ = cb.toggle_left_column(1, "my-class other-class")
        assert "my-class" in class_name
        assert "other-class" in class_name
        assert "collapsed" in class_name

    def test_removes_only_collapsed_class(self):
        class_name, _ = cb.toggle_left_column(1, "my-class collapsed extra")
        assert "collapsed" not in class_name
        assert "my-class" in class_name
        assert "extra" in class_name


# ---------------------------------------------------------------------------
# update_button_visibility
# ---------------------------------------------------------------------------


class TestUpdateButtonVisibility:
    def test_all_running_hides_run_button(self):
        run_style, cancel_style, in_progress = cb.update_button_visibility(True, True, True)
        assert run_style == {"display": "none"}
        assert in_progress is True

    def test_none_running_restores_run_button(self):
        run_style, cancel_style, in_progress = cb.update_button_visibility(False, False, False)
        assert cancel_style == {"display": "none"}
        assert in_progress is False

    def test_partial_running_still_hides_run_button(self):
        run_style, _, in_progress = cb.update_button_visibility(True, False, False)
        assert run_style == {"display": "none"}
        assert in_progress is True


# ---------------------------------------------------------------------------
# _solver_tab_class
# ---------------------------------------------------------------------------


class TestSolverTabClass:
    def test_returns_success_when_any_ok(self):
        rows = [{"ok": False}, {"ok": True}, {"ok": False}]
        assert cb._solver_tab_class(rows) == "tab-success"

    def test_returns_fail_when_none_ok(self):
        rows = [{"ok": False}, {"ok": False}]
        assert cb._solver_tab_class(rows) == "tab-fail"

    def test_empty_rows_returns_fail(self):
        assert cb._solver_tab_class([]) == "tab-fail"

    def test_all_ok_returns_success(self):
        rows = [{"ok": True}, {"ok": True}]
        assert cb._solver_tab_class(rows) == "tab-success"


# ---------------------------------------------------------------------------
# _solver_panel
# ---------------------------------------------------------------------------


class TestSolverPanel:
    def _make_row(self, energy, ok=True, starts=None, modes=None):
        return {
            "energy": energy,
            "ok": ok,
            "starts": starts or {1: 0, 2: 5},
            "modes": modes or {1: 1, 2: 2},
        }

    def test_returns_div(self, mps_path):
        rows = [self._make_row(302.0)]
        with patch("demo_callbacks.build_solution_graph", return_value=go.Figure()):
            panel = cb._solver_panel("HiGHS (MILP)", rows, mps_path)
        assert panel is not None

    def test_no_solution_row(self, mps_path):
        rows = [self._make_row(None, ok=False, starts={}, modes={})]
        with patch("demo_callbacks.build_solution_graph", return_value=go.Figure()):
            panel = cb._solver_panel("HiGHS (MILP)", rows, mps_path)
        assert panel is not None

    def test_picks_best_energy_row(self, mps_path):
        rows = [
            self._make_row(400.0, starts={1: 10}, modes={1: 1}),
            self._make_row(302.0, starts={1: 0}, modes={1: 2}),
        ]
        captured_starts = {}

        def fake_build(path, starts, modes, title):
            captured_starts.update(starts)
            return go.Figure()

        with patch("demo_callbacks.build_solution_graph", side_effect=fake_build):
            cb._solver_panel("HiGHS (MILP)", rows, mps_path)

        assert captured_starts.get(1) == 0  # best row has starts={1: 0}


# ---------------------------------------------------------------------------
# update_run_state
# ---------------------------------------------------------------------------


def _ctx(triggered_id):
    """Return a context manager that patches demo_callbacks.ctx.triggered_id."""
    mock = MagicMock()
    mock.triggered_id = triggered_id
    return patch("demo_callbacks.ctx", mock)


class TestUpdateRunState:
    def test_run_button_click_all_selected(self):
        selection = [
            str(SolverType.HIGHS.value),
            str(SolverType.SCIP.value),
            str(SolverType.STRIDE.value),
        ]
        with _ctx("run-button"):
            result = cb.update_run_state(1, 0, selection)
        assert result.run_button_style == {"display": "none"}

    def test_run_button_none_selected_results_tab_not_loading(self):
        with _ctx("run-button"):
            result = cb.update_run_state(1, 0, [])
        # When nothing is selected, results tab label stays "Results" (not "Loading...")
        assert result.results_tab_label == "Results"

    def test_cancel_button_restores_run_button(self):
        with _ctx("cancel-button"):
            result = cb.update_run_state(0, 1, [])
        assert result.cancel_button_style == {"display": "none"}
        assert result.run_button_style == {}

    def test_no_trigger_raises_prevent_update(self):
        with _ctx(None):
            with pytest.raises(dash.exceptions.PreventUpdate):
                cb.update_run_state(0, 0, [])


# ---------------------------------------------------------------------------
# run_highs / run_scip / run_stride — "not selected" branch
# ---------------------------------------------------------------------------

FAKE_RESULT = {
    "solver": "HiGHS (MILP)",
    "status": "Optimal",
    "energy": 302.0,
    "ok": True,
    "starts": {1: 0},
    "modes": {1: 1},
}


class TestRunHighsCallback:
    def test_not_selected_returns_not_selected_panel(self):
        # If HiGHS is not in selection, it should short-circuit
        result = cb.run_highs(
            run_click=1,
            solver_selection=[str(SolverType.SCIP.value)],
            time_limit=10,
            runs=1,
            input_file="input/30n20b8.mps",
        )
        assert result.running_highs is False
        assert result.highs_tab_label == "HiGHS"
        assert result.highs_store == {"run_click": 1, "rows": []}

    def test_selected_calls_compare_formulations(self, mps_path):
        fake_row = {
            **FAKE_RESULT,
            "formulation": "HiGHS (MILP)",
            "run": 1,
            "starts": {1: 0},
            "modes": {1: 1},
        }
        with (
            patch("demo_callbacks.compare_formulations", return_value=[fake_row]),
            patch("demo_callbacks.build_solution_graph", return_value=go.Figure()),
        ):
            result = cb.run_highs(
                run_click=1,
                solver_selection=[str(SolverType.HIGHS.value)],
                time_limit=10,
                runs=1,
                input_file=mps_path,
            )
        assert result.running_highs is False
        assert result.highs_tab_label == "HiGHS"
        assert result.highs_store["rows"] == [fake_row]


class TestRunScipCallback:
    def test_not_selected_returns_not_selected_panel(self):
        result = cb.run_scip(
            run_click=1,
            solver_selection=[str(SolverType.HIGHS.value)],
            time_limit=10,
            runs=1,
            input_file="input/30n20b8.mps",
        )
        assert result.running_scip is False
        assert result.scip_tab_label == "SCIP"

    def test_selected_calls_compare_formulations(self, mps_path):
        fake_row = {
            **FAKE_RESULT,
            "formulation": "SCIP (MILP)",
            "run": 1,
            "starts": {1: 0},
            "modes": {1: 1},
        }
        with (
            patch("demo_callbacks.compare_formulations", return_value=[fake_row]),
            patch("demo_callbacks.build_solution_graph", return_value=go.Figure()),
        ):
            result = cb.run_scip(
                run_click=1,
                solver_selection=[str(SolverType.SCIP.value)],
                time_limit=10,
                runs=1,
                input_file=mps_path,
            )
        assert result.running_scip is False
        assert result.scip_store["rows"] == [fake_row]


class TestRunStrideCallback:
    def test_not_selected_returns_not_selected_panel(self):
        result = cb.run_stride(
            run_click=1,
            solver_selection=[str(SolverType.HIGHS.value)],
            time_limit=10,
            runs=1,
            input_file="input/30n20b8.mps",
        )
        assert result.running_stride is False
        assert result.stride_tab_label == "Stride"

    def test_selected_calls_compare_formulations(self, mps_path):
        fake_row = {
            **FAKE_RESULT,
            "formulation": "Stride Hybrid Solver",
            "run": 1,
            "starts": {1: 0},
            "modes": {1: 1},
        }
        with (
            patch("demo_callbacks.compare_formulations", return_value=[fake_row]),
            patch("demo_callbacks.build_solution_graph", return_value=go.Figure()),
        ):
            result = cb.run_stride(
                run_click=1,
                solver_selection=[str(SolverType.STRIDE.value)],
                time_limit=10,
                runs=1,
                input_file=mps_path,
            )
        assert result.running_stride is False
        assert result.stride_store["rows"] == [fake_row]


# ---------------------------------------------------------------------------
# render_aggregate_results
# ---------------------------------------------------------------------------

SAMPLE_ROW = {
    "formulation": "HiGHS (MILP)",
    "run": 1,
    "status": "Optimal",
    "energy": 302.0,
    "ok": True,
    "starts": {"1": 0, "2": 5},
    "modes": {"1": 1, "2": 2},
}


class TestRenderAggregateResults:
    def test_empty_stores_returns_waiting_panel(self):
        content, disabled, label = cb.render_aggregate_results(
            highs_store={},
            scip_store={},
            stride_store={},
            run_click=1,
            input_file="input/30n20b8.mps",
        )
        assert disabled is True
        assert label == "Results"

    def test_stale_store_ignored(self):
        # Store from a different run_click should not count
        stale_store = {"run_click": 0, "rows": [SAMPLE_ROW]}
        content, disabled, label = cb.render_aggregate_results(
            highs_store=stale_store,
            scip_store={},
            stride_store={},
            run_click=1,
            input_file="input/30n20b8.mps",
        )
        assert disabled is True

    def test_matching_store_enables_results_tab(self, mps_path):
        store = {"run_click": 1, "rows": [SAMPLE_ROW]}
        with patch("demo_callbacks.build_comparison_graph", return_value=go.Figure()):
            content, disabled, label = cb.render_aggregate_results(
                highs_store=store,
                scip_store={},
                stride_store={},
                run_click=1,
                input_file=mps_path,
            )
        assert disabled is False
        assert label == "Results"

    def test_none_stores_treated_as_empty(self):
        content, disabled, label = cb.render_aggregate_results(
            highs_store=None,
            scip_store=None,
            stride_store=None,
            run_click=1,
            input_file="input/30n20b8.mps",
        )
        assert disabled is True

    def test_result_content_not_none_when_rows_present(self, mps_path):
        store = {"run_click": 2, "rows": [SAMPLE_ROW]}
        with patch("demo_callbacks.build_comparison_graph", return_value=go.Figure()):
            content, _, _ = cb.render_aggregate_results(
                highs_store=store,
                scip_store={},
                stride_store={},
                run_click=2,
                input_file=mps_path,
            )
        assert content is not None
