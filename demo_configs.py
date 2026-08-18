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

"""This file stores input parameters for the app."""

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "Resource-Constrained Project Scheduling"
MAIN_HEADER = "Resource-Constrained Project Scheduling"
DESCRIPTION = """\
Compare objective values and solver statuses for three formulations (HiGHS MILP, SCIP MILP, and
D-Wave Stride\u2122 Hybrid Solver) on the same MIPLIB RCPSP instance.
"""

# Human-readable labels for input files, keyed by input file path.
# The first entry is the default selection.
INPUT_LABELS = {
    "input/30n20b8.mps": "30 Jobs",
}

# Known optimal objective values keyed by input file path.
# Add an entry here whenever a new instance with a known optimal is added to INPUT_LABELS.
KNOWN_OPTIMA = {
    "input/30n20b8.mps": 302,
}

#######################################
# Sliders, buttons and option entries #
#######################################

# number of repeated runs per formulation
RUNS = {
    "min": 1,
    "max": 5,
    "step": 1,
    "value": 1,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 5,
    "max": 300,
    "step": 5,
    "value": 45,
}
