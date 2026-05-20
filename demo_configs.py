# Copyright 2024 D-Wave
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
Compare objective values and solver statuses for three formulations on the same RCPSP instance:
HiGHS MILP, SCIP MILP, and D-Wave Stride NL.
"""

INPUTS = ["input/30n20b8.mps"]

#######################################
# Sliders, buttons and option entries #
#######################################

# number of repeated runs per formulation
RUNS = {
    "min": 1,
    "max": 5,
    "step": 1,
    "value": 2,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 5,
    "max": 300,
    "step": 5,
    "value": 10,
}
