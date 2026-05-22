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

"""Tests for src/stride.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.stride import (
    _preprocessed_data,
    create_precedence_pairs,
    create_runtime_use_matrices,
    solve_stride,
)


# ---------------------------------------------------------------------------
# create_runtime_use_matrices  (integration — uses real MPS file)
# ---------------------------------------------------------------------------

class TestCreateRuntimeUseMatrices:
    def test_shapes(self, mps_path):
        rt, rm, rtech = create_runtime_use_matrices(mps_path)
        assert rt.shape   == (30, 3)
        assert rm.shape   == (30, 3)
        assert rtech.shape == (30, 3)

    def test_non_negative_values(self, mps_path):
        rt, rm, rtech = create_runtime_use_matrices(mps_path)
        assert (rt   >= 0).all()
        assert (rm   >= 0).all()
        assert (rtech >= 0).all()

    def test_runtimes_are_positive_for_at_least_some_jobs(self, mps_path):
        rt, _, _ = create_runtime_use_matrices(mps_path)
        assert rt.sum() > 0


# ---------------------------------------------------------------------------
# create_precedence_pairs  (integration — uses real MPS file)
# ---------------------------------------------------------------------------

class TestCreatePrecedencePairs:
    def test_returns_list_of_tuples(self, mps_path):
        pairs = create_precedence_pairs(mps_path)
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, tuple)
            assert len(p) == 2

    def test_job_ids_in_range(self, mps_path):
        pairs = create_precedence_pairs(mps_path)
        for j1, j2 in pairs:
            assert 1 <= j1 <= 30
            assert 1 <= j2 <= 30

    def test_no_self_loops(self, mps_path):
        pairs = create_precedence_pairs(mps_path)
        for j1, j2 in pairs:
            assert j1 != j2


# ---------------------------------------------------------------------------
# _preprocessed_data  (caching wrapper)
# ---------------------------------------------------------------------------

class TestPreprocessedData:
    def test_returns_four_tuple(self, mps_path):
        result = _preprocessed_data(mps_path)
        assert len(result) == 4

    def test_cached_second_call_same_object(self, mps_path):
        r1 = _preprocessed_data(mps_path)
        r2 = _preprocessed_data(mps_path)
        # lru_cache should return identical objects
        assert r1 is r2


# ---------------------------------------------------------------------------
# solve_stride
# ---------------------------------------------------------------------------

class TestSolveStride:
    def test_unavailable_when_sampler_is_none(self):
        with patch("src.stride.LeapHybridNLSampler", None):
            result = solve_stride(5.0, "any.mps")
        assert result["ok"] is False
        assert "Unavailable" in result["status"]
        assert result["energy"] is None
        assert result["solver"] == "Stride"

    def test_successful_run(self, mps_path):
        """Mock the sampler and model state to exercise the happy path."""
        runtimes = np.zeros((30, 3))
        rm_use   = np.zeros((30, 3))
        rt_use   = np.zeros((30, 3))
        pairs    = [(1, 2)]

        fake_starts = MagicMock()
        fake_starts.state.return_value = np.zeros(30)
        fake_modes  = MagicMock()
        fake_modes.state.return_value  = np.zeros(30)

        fake_objective = MagicMock()
        fake_objective.state.return_value = np.array([302.0])

        fake_constraint = MagicMock()
        fake_constraint.state.return_value = True

        fake_model = MagicMock()
        fake_model.objective = fake_objective
        fake_model.iter_constraints.return_value = [fake_constraint]

        fake_sampler_instance = MagicMock()

        with (
            patch("src.stride._preprocessed_data", return_value=(runtimes, rm_use, rt_use, pairs)),
            patch("src.stride.create_model", return_value=(fake_model, fake_starts, fake_modes)),
            patch("src.stride.LeapHybridNLSampler", return_value=fake_sampler_instance),
        ):
            result = solve_stride(5.0, mps_path)

        assert result["solver"] == "Stride"
        assert result["status"] == "Completed"
        assert result["energy"] == pytest.approx(302.0)
        assert isinstance(result["ok"], bool)
        assert isinstance(result["starts"], dict)
        assert isinstance(result["modes"], dict)
        assert len(result["starts"]) == 30
        assert len(result["modes"])  == 30

    def test_modes_are_one_indexed_in_output(self, mps_path):
        """Modes returned by solve_stride should be 1-indexed (raw value + 1)."""
        runtimes = np.zeros((30, 3))
        rm_use   = np.zeros((30, 3))
        rt_use   = np.zeros((30, 3))
        pairs    = []

        fake_starts = MagicMock()
        fake_starts.state.return_value = np.zeros(30)
        fake_modes  = MagicMock()
        # All modes at raw index 0 should become mode 1
        fake_modes.state.return_value = np.zeros(30)

        fake_objective = MagicMock()
        fake_objective.state.return_value = np.array([0.0])

        fake_model = MagicMock()
        fake_model.objective = fake_objective
        fake_model.iter_constraints.return_value = []

        with (
            patch("src.stride._preprocessed_data", return_value=(runtimes, rm_use, rt_use, pairs)),
            patch("src.stride.create_model", return_value=(fake_model, fake_starts, fake_modes)),
            patch("src.stride.LeapHybridNLSampler", return_value=MagicMock()),
        ):
            result = solve_stride(5.0, mps_path)

        for mode_val in result["modes"].values():
            assert mode_val == 1  # 0 + 1

    def test_exception_returns_error_dict(self, mps_path):
        with (
            patch("src.stride._preprocessed_data", side_effect=RuntimeError("oops")),
            patch("src.stride.LeapHybridNLSampler", MagicMock()),
        ):
            result = solve_stride(5.0, mps_path)

        assert result["ok"] is False
        assert "Error" in result["status"]
        assert result["energy"] is None

    def test_result_keys_always_present_on_error(self, mps_path):
        with (
            patch("src.stride._preprocessed_data", side_effect=ValueError("bad")),
            patch("src.stride.LeapHybridNLSampler", MagicMock()),
        ):
            result = solve_stride(5.0, mps_path)

        for key in ("solver", "status", "energy", "ok", "starts", "modes"):
            assert key in result
