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


from __future__ import annotations

import re
from typing import Any

import highspy as hs



def _extract_assignment_highs(highs_model: Any) -> tuple[dict[int, int], dict[int, int]]:
    """Extract the start-time and mode assignment from a solved HiGHS model.

    Reads all variables named ``x_<job>_<mode>_<start>`` and returns the
    assignment with value > 0.5 for each job.

    Args:
        highs_model: A solved ``highspy.Highs`` instance.

    Returns:
        A tuple containing:

        - dict[int, int]: Start time keyed by job ID.
        - dict[int, int]: Execution mode keyed by job ID.
    """
    names = []
    values = []

    try:
        lp = highs_model.getLp()
        names = list(lp.col_names_)
    except Exception:
        names = []

    try:
        values = list(highs_model.allVariableValues())
    except Exception:
        try:
            sol = highs_model.getSolution()
            values = list(getattr(sol, "col_value", []))
        except Exception:
            values = []

    if not names or not values:
        return {}, {}

    pattern = re.compile(r"x_(\d+)_(\d+)_(\d+)")
    best_choice: dict[int, tuple[float, int, int]] = {}

    for name, value in zip(names, values):
        match = pattern.fullmatch(name)
        if not match:
            continue
        job = int(match.group(1))
        mode = int(match.group(2))
        start = int(match.group(3))
        if job not in best_choice or float(value) > best_choice[job][0]:
            best_choice[job] = (float(value), mode, start)

    starts = {job: choice[2] for job, choice in best_choice.items() if choice[0] > 0.5}
    modes = {job: choice[1] for job, choice in best_choice.items() if choice[0] > 0.5}
    return starts, modes


def solve_highs(time_limit: float, input_path: str) -> dict[str, Any]:
    """Run the HiGHS MILP formulation once and return comparable result metadata.

    Args:
        time_limit: Maximum solver runtime in seconds.
        input_path: Path to the MPS-format problem file.

    Returns:
        A dictionary with keys:

        - ``"solver"``: solver label string.
        - ``"status"``: HiGHS model status string.
        - ``"energy"``: best objective value found, or ``None`` if infeasible.
        - ``"ok"``: ``True`` if a finite objective was obtained.
        - ``"starts"``: start time keyed by job ID (empty if infeasible).
        - ``"modes"``: execution mode keyed by job ID (empty if infeasible).
    """
    if hs is None:
        return {
            "solver": "HiGHS (MILP)",
            "status": "Unavailable: highspy not installed",
            "energy": None,
            "ok": False,
        }

    try:
        h = hs.Highs()
        h.readModel(input_path)
        h.setOptionValue("time_limit", time_limit)
        h.setOptionValue("output_flag", False)
        h.run()

        info = h.getInfo()
        status_str = h.getModelStatus()
        starts, modes = _extract_assignment_highs(h)

        import math
        obj = info.objective_function_value
        feasible = math.isfinite(obj)
        return {
            "solver": "HiGHS (MILP)",
            "status": str(status_str),
            "energy": obj if feasible else None,
            "ok": feasible,
            "starts": starts if feasible else {},
            "modes": modes if feasible else {},
        }
    except Exception as exc:  # pragma: no cover - runtime/system dependent
        return {
            "solver": "HiGHS (MILP)",
            "status": f"Error: {exc}",
            "energy": None,
            "ok": False,
            "starts": {},
            "modes": {},
        }
