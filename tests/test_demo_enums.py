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

"""Tests for src/demo_enums.py."""

from src.demo_enums import SolverType


class TestSolverType:
    def test_highs_value(self):
        assert SolverType.HIGHS.value == 0

    def test_scip_value(self):
        assert SolverType.SCIP.value == 1

    def test_stride_value(self):
        assert SolverType.STRIDE.value == 2

    def test_highs_label(self):
        assert SolverType.HIGHS.label == "HiGHS (MILP)"

    def test_scip_label(self):
        assert SolverType.SCIP.label == "SCIP (MILP)"

    def test_stride_label(self):
        assert SolverType.STRIDE.label == "Stride Hybrid Solver"

    def test_from_int(self):
        assert SolverType(0) is SolverType.HIGHS
        assert SolverType(1) is SolverType.SCIP
        assert SolverType(2) is SolverType.STRIDE

    def test_all_members(self):
        members = list(SolverType)
        assert len(members) == 3
