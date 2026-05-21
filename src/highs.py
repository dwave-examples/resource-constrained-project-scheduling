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

try:
    import highspy as hs
except Exception:  # pragma: no cover - optional dependency
    hs = None


def _extract_assignment_highs(highs_model: Any) -> tuple[dict[int, int], dict[int, int]]:
    """Extract selected x_(job,mode,start) assignments from a HiGHS solution."""
    names = []
    values = []

    try:
        lp = highs_model.getLp()
        names = list(getattr(lp, "col_names", []))
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
    """Run the HiGHS MILP formulation once and return comparable result metadata."""
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

        return {
            "solver": "HiGHS (MILP)",
            "status": str(status_str),
            "energy": info.objective_function_value,
            "ok": True,
            "starts": starts,
            "modes": modes,
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


if __name__ == "__main__":
    import pandas as pd

    time_limits = [5, 10]
    rows = []

    for time in time_limits:
        for _ in range(5):
            result = solve_highs(time, input_path="input/30n20b8.mps")
            rows.append({"time": time, **result})

    pd.DataFrame(rows).to_csv("highs_rcpsp.csv", index=False)
