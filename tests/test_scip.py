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

"""Tests for src/scip.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scip import _extract_assignment_scip, solve_scip


# ---------------------------------------------------------------------------
# _extract_assignment_scip
# ---------------------------------------------------------------------------

def _make_scip_var(name: str, value: float) -> MagicMock:
    """Helper: create a mock SCIP variable with .name and a fixed value."""
    var = MagicMock()
    var.name = name
    return var


class TestExtractAssignmentScip:
    def _make_model(self, var_data: list[tuple[str, float]]) -> MagicMock:
        """Build a mock SCIP model whose getVars()/getVal() match var_data."""
        model = MagicMock()
        vars_ = []
        values = {}
        for name, val in var_data:
            v = _make_scip_var(name, val)
            vars_.append(v)
            values[id(v)] = val
        model.getVars.return_value = vars_
        model.getVal.side_effect = lambda v: values[id(v)]
        return model

    def test_basic_extraction(self):
        model = self._make_model([("x_1_2_5", 1.0), ("x_2_1_0", 1.0)])
        starts, modes = _extract_assignment_scip(model)
        assert starts == {1: 5, 2: 0}
        assert modes  == {1: 2, 2: 1}

    def test_below_threshold_excluded(self):
        model = self._make_model([("x_3_1_10", 0.4)])
        starts, modes = _extract_assignment_scip(model)
        assert starts == {}
        assert modes  == {}

    def test_above_threshold_included(self):
        model = self._make_model([("x_3_1_10", 0.51)])
        starts, modes = _extract_assignment_scip(model)
        assert starts == {3: 10}
        assert modes  == {3: 1}

    def test_best_candidate_wins(self):
        # Two candidates for job 5: lower value first, then higher
        model = self._make_model([("x_5_1_0", 0.3), ("x_5_2_7", 0.9)])
        starts, modes = _extract_assignment_scip(model)
        assert starts == {5: 7}
        assert modes  == {5: 2}

    def test_non_x_variables_ignored(self):
        model = self._make_model([("C_1", 5.0), ("S_2", 3.0)])
        starts, modes = _extract_assignment_scip(model)
        assert starts == {}
        assert modes  == {}

    def test_empty_model(self):
        model = self._make_model([])
        starts, modes = _extract_assignment_scip(model)
        assert starts == {}
        assert modes  == {}

    def test_exception_returns_empty(self):
        model = MagicMock()
        model.getVars.side_effect = RuntimeError("no vars")
        starts, modes = _extract_assignment_scip(model)
        assert starts == {}
        assert modes  == {}


# ---------------------------------------------------------------------------
# solve_scip
# ---------------------------------------------------------------------------

class TestSolveScip:
    def test_unavailable_when_model_is_none(self):
        with patch("src.scip.Model", None):
            result = solve_scip(10.0, "any.mps")
        assert result["ok"] is False
        assert "Unavailable" in result["status"]
        assert result["energy"] is None

    def test_successful_run(self, mps_path):
        mock_model_instance = MagicMock()
        mock_model_instance.getNSols.return_value = 1
        mock_model_instance.getStatus.return_value = "optimal"
        mock_model_instance.getPrimalbound.return_value = 302.0
        mock_model_instance.getVars.return_value = []

        with patch("src.scip.Model", return_value=mock_model_instance):
            result = solve_scip(10.0, mps_path)

        assert result["solver"] == "SCIP (MILP)"
        assert result["ok"] is True
        assert result["energy"] == pytest.approx(302.0)
        assert isinstance(result["starts"], dict)
        assert isinstance(result["modes"], dict)

    def test_infeasible_returns_none_energy(self, mps_path):
        mock_model_instance = MagicMock()
        mock_model_instance.getNSols.return_value = 0
        mock_model_instance.getStatus.return_value = "infeasible"
        mock_model_instance.getVars.return_value = []

        with patch("src.scip.Model", return_value=mock_model_instance):
            result = solve_scip(10.0, mps_path)

        assert result["ok"] is False
        assert result["energy"] is None
        assert result["starts"] == {}
        assert result["modes"] == {}

    def test_exception_returns_error_dict(self, mps_path):
        with patch("src.scip.Model", side_effect=RuntimeError("crash")):
            result = solve_scip(10.0, mps_path)

        assert result["ok"] is False
        assert "Error" in result["status"]
        assert result["energy"] is None

    def test_result_keys_present(self, mps_path):
        mock_model_instance = MagicMock()
        mock_model_instance.getNSols.return_value = 1
        mock_model_instance.getStatus.return_value = "optimal"
        mock_model_instance.getPrimalbound.return_value = 200.0
        mock_model_instance.getVars.return_value = []

        with patch("src.scip.Model", return_value=mock_model_instance):
            result = solve_scip(5.0, mps_path)

        for key in ("solver", "status", "energy", "ok", "starts", "modes"):
            assert key in result
