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

from pyscipopt import Model



def _extract_assignment_scip(model: Any) -> tuple[dict[int, int], dict[int, int]]:
    """Extract the start-time and mode assignment from a solved SCIP model.

    Reads all variables named ``x_<job>_<mode>_<start>`` and returns the
    assignment with value > 0.5 for each job.

    Args:
        model: A solved ``pyscipopt.Model`` instance.

    Returns:
        A tuple containing:

        - dict[int, int]: Start time keyed by job ID.
        - dict[int, int]: Execution mode keyed by job ID.
    """
    pattern = re.compile(r"x_(\d+)_(\d+)_(\d+)")
    best_choice: dict[int, tuple[float, int, int]] = {}

    try:
        for var in model.getVars():
            name = str(getattr(var, "name", ""))
            match = pattern.fullmatch(name)
            if not match:
                continue

            value = float(model.getVal(var))
            job = int(match.group(1))
            mode = int(match.group(2))
            start = int(match.group(3))
            if job not in best_choice or value > best_choice[job][0]:
                best_choice[job] = (value, mode, start)
    except Exception:
        return {}, {}

    starts = {job: choice[2] for job, choice in best_choice.items() if choice[0] > 0.5}
    modes = {job: choice[1] for job, choice in best_choice.items() if choice[0] > 0.5}
    return starts, modes


def solve_scip(time_limit: float, input_path: str) -> dict[str, Any]:
    """Run the SCIP MILP formulation once and return comparable result metadata.

    Args:
        time_limit: Maximum solver runtime in seconds.
        input_path: Path to the MPS-format problem file.

    Returns:
        A dictionary with keys:

        - ``"solver"``: solver label string.
        - ``"status"``: SCIP status string.
        - ``"energy"``: best objective value found, or ``None`` if infeasible.
        - ``"ok"``: ``True`` if at least one feasible solution was found.
        - ``"starts"``: start time keyed by job ID (empty if infeasible).
        - ``"modes"``: execution mode keyed by job ID (empty if infeasible).
    """
    if Model is None:
        return {
            "solver": "SCIP (MILP)",
            "status": "Unavailable: pyscipopt not installed",
            "energy": None,
            "ok": False,
        }

    try:
        model = Model()
        model.readProblem(input_path)
        model.setParam("limits/time", time_limit)
        model.hideOutput()
        model.optimize()
        starts, modes = _extract_assignment_scip(model)
        feasible = model.getNSols() > 0

        return {
            "solver": "SCIP (MILP)",
            "status": model.getStatus(),
            "energy": model.getPrimalbound() if feasible else None,
            "ok": feasible,
            "starts": starts if feasible else {},
            "modes": modes if feasible else {},
        }
    except Exception as exc:  # pragma: no cover - runtime/system dependent
        return {
            "solver": "SCIP (MILP)",
            "status": f"Error: {exc}",
            "energy": None,
            "ok": False,
            "starts": {},
            "modes": {},
        }
