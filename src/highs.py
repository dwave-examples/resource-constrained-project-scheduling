from __future__ import annotations

from typing import Any

try:
    import highspy as hs
except Exception:  # pragma: no cover - optional dependency
    hs = None


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
        print(status_str)

        return {
            "solver": "HiGHS (MILP)",
            "status": str(status_str),
            "energy": info.objective_function_value,
            "ok": True,
        }
    except Exception as exc:  # pragma: no cover - runtime/system dependent
        return {
            "solver": "HiGHS (MILP)",
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
            result = solve_highs(time, input_path="input/30n20b8.mps")
            rows.append({"time": time, **result})

    pd.DataFrame(rows).to_csv("highs_rcpsp.csv", index=False)
