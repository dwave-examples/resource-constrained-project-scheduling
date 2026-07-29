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

"""Tests for src/highs.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.highs import _extract_assignment_highs, solve_highs

# ---------------------------------------------------------------------------
# _extract_assignment_highs
# ---------------------------------------------------------------------------


class TestExtractAssignmentHighs:
    def _make_model(self, names: list[str], values: list[float]) -> MagicMock:
        """Build a minimal mock HiGHS model."""
        lp = SimpleNamespace(col_names_=names)
        model = MagicMock()
        model.getLp.return_value = lp
        model.allVariableValues.return_value = values
        return model

    def test_basic_extraction(self):
        # x_<job>_<mode>_<start> — job 1 mode 2 starts at 5, job 2 mode 1 starts at 0
        names = ["x_1_2_5", "x_2_1_0", "other_var"]
        values = [1.0, 1.0, 0.0]
        model = self._make_model(names, values)
        starts, modes = _extract_assignment_highs(model)
        assert starts == {1: 5, 2: 0}
        assert modes == {1: 2, 2: 1}

    def test_fractional_value_below_threshold_excluded(self):
        names = ["x_3_1_10"]
        values = [0.4]  # below 0.5 threshold
        model = self._make_model(names, values)
        starts, modes = _extract_assignment_highs(model)
        assert starts == {}
        assert modes == {}

    def test_fractional_value_above_threshold_included(self):
        names = ["x_3_1_10"]
        values = [0.51]
        model = self._make_model(names, values)
        starts, modes = _extract_assignment_highs(model)
        assert starts == {3: 10}
        assert modes == {3: 1}

    def test_best_value_wins_for_duplicate_job(self):
        # Two candidates for job 5: value 0.3 and 0.9 — only the higher one counts
        names = ["x_5_1_0", "x_5_2_7"]
        values = [0.3, 0.9]
        model = self._make_model(names, values)
        starts, modes = _extract_assignment_highs(model)
        assert starts == {5: 7}
        assert modes == {5: 2}

    def test_no_matching_variables(self):
        names = ["C_1", "S_2", "R_Mechaniker"]
        values = [1.0, 1.0, 40.0]
        model = self._make_model(names, values)
        starts, modes = _extract_assignment_highs(model)
        assert starts == {}
        assert modes == {}

    def test_empty_names_returns_empty(self):
        model = self._make_model([], [])
        starts, modes = _extract_assignment_highs(model)
        assert starts == {}
        assert modes == {}

    def test_get_lp_raises_falls_back(self):
        model = MagicMock()
        model.getLp.side_effect = RuntimeError("no LP")
        starts, modes = _extract_assignment_highs(model)
        assert starts == {}
        assert modes == {}

    def test_all_variable_values_raises_falls_back(self):
        lp = SimpleNamespace(col_names_=["x_1_1_0"])
        model = MagicMock()
        model.getLp.return_value = lp
        model.allVariableValues.side_effect = RuntimeError("no values")
        model.getSolution.side_effect = RuntimeError("no solution either")
        starts, modes = _extract_assignment_highs(model)
        assert starts == {}
        assert modes == {}


# ---------------------------------------------------------------------------
# solve_highs
# ---------------------------------------------------------------------------


class TestSolveHighs:
    def test_unavailable_when_hs_is_none(self):
        with patch("src.highs.hs", None):
            result = solve_highs(10.0, "any.mps")
        assert result["ok"] is False
        assert "Unavailable" in result["status"]
        assert result["energy"] is None

    def test_successful_run(self, mps_path):
        fake_info = SimpleNamespace(objective_function_value=302.0)
        fake_model = MagicMock()
        fake_model.getInfo.return_value = fake_info
        fake_model.getModelStatus.return_value = "Optimal"

        lp = SimpleNamespace(col_names_=["x_1_1_0"])
        fake_model.getLp.return_value = lp
        fake_model.allVariableValues.return_value = [1.0]

        mock_hs = MagicMock()
        mock_hs.Highs.return_value = fake_model

        with patch("src.highs.hs", mock_hs):
            result = solve_highs(10.0, mps_path)

        assert result["solver"] == "HiGHS (MILP)"
        assert result["ok"] is True
        assert result["energy"] == pytest.approx(302.0)
        assert isinstance(result["starts"], dict)
        assert isinstance(result["modes"], dict)

    def test_infeasible_returns_none_energy(self, mps_path):
        fake_info = SimpleNamespace(objective_function_value=float("inf"))
        fake_model = MagicMock()
        fake_model.getInfo.return_value = fake_info
        fake_model.getModelStatus.return_value = "Infeasible"
        fake_model.getLp.return_value = SimpleNamespace(col_names_=[])
        fake_model.allVariableValues.return_value = []

        mock_hs = MagicMock()
        mock_hs.Highs.return_value = fake_model

        with patch("src.highs.hs", mock_hs):
            result = solve_highs(10.0, mps_path)

        assert result["ok"] is False
        assert result["energy"] is None
        assert result["starts"] == {}
        assert result["modes"] == {}

    def test_exception_returns_error_dict(self, mps_path):
        mock_hs = MagicMock()
        mock_hs.Highs.side_effect = RuntimeError("boom")

        with patch("src.highs.hs", mock_hs):
            result = solve_highs(10.0, mps_path)

        assert result["ok"] is False
        assert "Error" in result["status"]
        assert result["energy"] is None

    def test_result_keys_present(self, mps_path):
        fake_info = SimpleNamespace(objective_function_value=100.0)
        fake_model = MagicMock()
        fake_model.getInfo.return_value = fake_info
        fake_model.getModelStatus.return_value = "Optimal"
        fake_model.getLp.return_value = SimpleNamespace(col_names_=[])
        fake_model.allVariableValues.return_value = []

        mock_hs = MagicMock()
        mock_hs.Highs.return_value = fake_model

        with patch("src.highs.hs", mock_hs):
            result = solve_highs(5.0, mps_path)

        for key in ("solver", "status", "energy", "ok", "starts", "modes"):
            assert key in result
