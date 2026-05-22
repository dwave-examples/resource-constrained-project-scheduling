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

"""Shared pytest fixtures for the RCPSP test suite."""

from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPS_PATH = os.path.join(REPO_ROOT, "input", "30n20b8.mps")


@pytest.fixture
def mps_path() -> str:
    """Return the absolute path to the bundled MPS instance."""
    return MPS_PATH


# ---------------------------------------------------------------------------
# Minimal synthetic profile used by pure-function tests that don't need MPS
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_profile() -> dict:
    """Return a hand-crafted 3-job profile for lightweight unit tests.

    Topology: job 1 → job 2 → job 3  (chain)
    Modes: one mode per job (mode 1)
    Resources:
      - job 1 uses 2 mechanics, 0 technicians, duration 5
      - job 2 uses 0 mechanics, 2 technicians, duration 3
      - job 3 uses 1 mechanic,  1 technician,  duration 4
    """
    return {
        "jobs": [1, 2, 3],
        "edges": [(1, 2), (2, 3)],
        "durations": {(1, 1): 5, (2, 1): 3, (3, 1): 4},
        "mechanic_use": {(1, 1): 2.0, (2, 1): 0.0, (3, 1): 1.0},
        "technician_use": {(1, 1): 0.0, (2, 1): 2.0, (3, 1): 1.0},
        "capacities": {"Mechaniker": 3.0, "Techniker": 2.0},
    }


@pytest.fixture
def multi_mode_profile() -> dict:
    """Return a 2-job profile where each job has 2 modes, for mode-selection tests."""
    return {
        "jobs": [1, 2],
        "edges": [(1, 2)],
        "durations":      {(1, 1): 10, (1, 2): 4, (2, 1): 6, (2, 2): 3},
        "mechanic_use":   {(1, 1): 1.0, (1, 2): 3.0, (2, 1): 2.0, (2, 2): 2.0},
        "technician_use": {(1, 1): 0.0, (1, 2): 0.0, (2, 1): 0.0, (2, 2): 0.0},
        "capacities": {"Mechaniker": 5.0, "Techniker": 0.0},
    }
