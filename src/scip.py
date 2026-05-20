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
