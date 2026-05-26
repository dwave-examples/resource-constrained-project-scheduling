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
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


def extract_assignment(
    pairs: Iterable[tuple[str, float]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Parse ``x_<job>_<mode>_<start>`` variable names and return the best assignment per job.

    Selects the ``(mode, start)`` pair with the highest value for each job.
    Only assignments whose best value exceeds 0.5 are included in the output.

    Args:
        pairs: Iterable of ``(variable_name, value)`` tuples from a solved model.

    Returns:
        A tuple containing:

        - dict[int, int]: Start time keyed by 1-indexed job ID.
        - dict[int, int]: Execution mode keyed by 1-indexed job ID.
    """
    pattern = re.compile(r"x_(\d+)_(\d+)_(\d+)")
    best_choice: dict[int, tuple[float, int, int]] = {}

    for name, value in pairs:
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


def parse_mps_structure(input_path: str) -> dict:
    """Parse an MPS file to extract jobs, precedences, resource usage, bounds, and capacities.

    Reads the ROWS, COLUMNS, and BOUNDS sections of the file directly, without
    an LP solver library.  This is the single source of truth for MPS data used
    across the demo; both the plot builders and the Stride model builder call it.

    Args:
        input_path: Path to the MPS-format problem file.

    Returns:
        A dictionary with keys:

        - ``"jobs"``: sorted list of job IDs.
        - ``"edges"``: sorted list of ``(src, dst)`` precedence pairs.
        - ``"durations"``: mapping of ``(job, mode)`` → duration in time units.
        - ``"mechanic_use"``: mapping of ``(job, mode)`` → mechanic units consumed.
        - ``"technician_use"``: mapping of ``(job, mode)`` → technician units consumed.
        - ``"capacities"``: dict with ``"Mechaniker"`` and ``"Techniker"`` upper bounds.
        - ``"lower_bounds"``: mapping of job ID → lower bound on start time.
        - ``"upper_bounds"``: mapping of job ID → upper bound on start time.
    """
    empty: dict = {
        "jobs": [],
        "edges": [],
        "durations": {},
        "mechanic_use": {},
        "technician_use": {},
        "capacities": {"Mechaniker": 0.0, "Techniker": 0.0},
        "lower_bounds": {},
        "upper_bounds": {},
        "hire_rates": {"Mechaniker": 0.0, "Techniker": 0.0},
    }
    path = Path(input_path)
    if not path.exists() or not path.is_file():
        return empty

    rows_lines: list[str] = []
    columns_lines: list[str] = []
    bounds_lines: list[str] = []

    section = ""
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        upper = line.upper()
        if upper == "ROWS":
            section = "ROWS"
            continue
        if upper == "COLUMNS":
            section = "COLUMNS"
            continue
        if upper == "BOUNDS":
            section = "BOUNDS"
            continue
        if upper in {"RHS", "RANGES", "ENDATA"}:
            section = ""
            continue

        if section == "ROWS":
            rows_lines.append(line)
        elif section == "COLUMNS":
            columns_lines.append(line)
        elif section == "BOUNDS":
            bounds_lines.append(line)

    # ROWS: precedence edges and job IDs
    edge_pattern = re.compile(r"precedence_(\d+)_(\d+)")
    edges: list[tuple[int, int]] = []
    jobs: set[int] = set()

    for row in rows_lines:
        tokens = row.split()
        if len(tokens) < 2:
            continue
        name = tokens[1]
        match = edge_pattern.search(name)
        if match:
            src, dst = int(match.group(1)), int(match.group(2))
            edges.append((src, dst))
            jobs.update([src, dst])
        elif name.startswith("start_"):
            jobs.add(int(name.split("_")[-1]))

    # COLUMNS: durations and resource usage from x_<job>_<mode>_<start> variables
    rm_sets: dict = defaultdict(set)
    rt_sets: dict = defaultdict(set)
    rm_use: dict = defaultdict(float)
    rt_use: dict = defaultdict(float)
    x_pattern = re.compile(r"x_(\d+)_(\d+)_(\d+)")
    hire_rates: dict[str, float] = {}

    for line in columns_lines:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        if tokens[0] in ("R_Mechaniker", "R_Techniker"):
            for i in range(1, len(tokens) - 1, 2):
                if tokens[i] == "Obj":
                    try:
                        hire_rates[tokens[0]] = float(tokens[i + 1])
                    except ValueError:
                        pass
            continue
        m_var = x_pattern.match(tokens[0])
        if not m_var:
            continue
        job, mode, start_t = int(m_var.group(1)), int(m_var.group(2)), int(m_var.group(3))
        jobs.add(job)
        for i in range(1, len(tokens) - 1, 2):
            row_name = tokens[i]
            try:
                value = float(tokens[i + 1])
            except ValueError:
                continue
            if row_name.startswith("rescap_Mechaniker_"):
                rm_sets[(job, mode, start_t)].add(int(row_name.split("_")[-1]))
                rm_use[(job, mode)] = max(rm_use[(job, mode)], value)
            elif row_name.startswith("rescap_Techniker_"):
                rt_sets[(job, mode, start_t)].add(int(row_name.split("_")[-1]))
                rt_use[(job, mode)] = max(rt_use[(job, mode)], value)

    durations: dict = defaultdict(int)
    for (job, mode, _), times in rm_sets.items():
        durations[(job, mode)] = max(durations[(job, mode)], len(times))
    for (job, mode, _), times in rt_sets.items():
        durations[(job, mode)] = max(durations[(job, mode)], len(times))

    # BOUNDS: S_<job> start bounds and resource capacities
    capacities = {"Mechaniker": 0.0, "Techniker": 0.0}
    lower_bounds: dict[int, int] = {}
    upper_bounds: dict[int, int] = {}

    for line in bounds_lines:
        tokens = line.split()
        if len(tokens) < 4:
            continue
        bound_type, var_name = tokens[0], tokens[2]
        try:
            value = float(tokens[3])
        except ValueError:
            continue
        if var_name == "R_Mechaniker" and bound_type == "UP":
            capacities["Mechaniker"] = value
        elif var_name == "R_Techniker" and bound_type == "UP":
            capacities["Techniker"] = value
        elif var_name.startswith("S_"):
            job = int(var_name.split("_")[1])
            if bound_type == "LO":
                lower_bounds[job] = int(value)
            elif bound_type == "UP":
                upper_bounds[job] = int(value)

    return {
        "jobs": sorted(jobs),
        "edges": sorted(set(edges)),
        "durations": dict(durations),
        "mechanic_use": dict(rm_use),
        "technician_use": dict(rt_use),
        "capacities": capacities,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "hire_rates": {
            "Mechaniker": hire_rates.get("R_Mechaniker", 0.0),
            "Techniker": hire_rates.get("R_Techniker", 0.0),
        },
    }
