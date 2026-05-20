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

from typing import Any

try:
    from pyscipopt import Model
except Exception:  # pragma: no cover - optional dependency
    Model = None


def solve_scip(time_limit: float, input_path: str) -> dict[str, Any]:
    """Run the SCIP MILP formulation once and return comparable result metadata."""
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

        return {
            "solver": "SCIP (MILP)",
            "status": model.getStatus(),
            "energy": model.getPrimalbound(),
            "ok": True,
        }
    except Exception as exc:  # pragma: no cover - runtime/system dependent
        return {
            "solver": "SCIP (MILP)",
            "status": f"Error: {exc}",
            "energy": None,
            "ok": False,
        }


if __name__ == "__main__":
    import pandas as pd

    time_limits = [5, 10]
    rows = []

    for time in time_limits:
        for _ in range(5):
            result = solve_scip(time, input_path="input/30n20b8.mps")
            rows.append({"time": time, **result})

    pd.DataFrame(rows).to_csv("scip_rcpsp.csv", index=False)
