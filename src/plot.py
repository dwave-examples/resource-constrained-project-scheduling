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
from collections import defaultdict, deque
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colors keyed by (resource_type, mode): shade encodes mode speed within each resource family
_JOB_COLORS: dict[tuple[str, int], str] = {
    ("mechanic",    1): "#9BC2F1",
    ("mechanic",    2): "#3886E3",
    ("mechanic",    3): "#1757A5",
    ("technician",  1): "#F8AF7B",
    ("technician",  2): "#EF6B0D",
    ("technician",  3): "#A64A09",
}
# Mid-shade of each family used for the resource demand lines
_MECHANIC_LINE_COLOR = "#3886E3"
_TECHNICIAN_LINE_COLOR = "#EF6B0D"

def _parse_mps_structure(input_path: str) -> dict:
    """Parse an MPS file to extract jobs, precedences, resource usage, and capacities.

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
    """
    path = Path(input_path)
    if not path.exists() or not path.is_file():
        return {
            "jobs": [],
            "edges": [],
            "durations": {},
            "mechanic_use": {},
            "technician_use": {},
            "capacities": {"Mechaniker": 0.0, "Techniker": 0.0},
        }

    rows: list[str] = []
    columns: list[str] = []
    rhs: list[str] = []

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
        if upper == "RHS":
            section = "RHS"
            continue
        if upper in {"RANGES", "BOUNDS", "ENDATA"}:
            section = ""
            continue

        if section == "ROWS":
            rows.append(line)
        elif section == "COLUMNS":
            columns.append(line)
        elif section == "RHS":
            rhs.append(line)

    edge_pattern = re.compile(r"precedence_(\d+)_(\d+)")
    edges = []
    jobs = set()

    for row in rows:
        tokens = row.split()
        if len(tokens) >= 2:
            name = tokens[1]
            match = edge_pattern.search(name)
            if match:
                src = int(match.group(1))
                dst = int(match.group(2))
                edges.append((src, dst))
                jobs.update([src, dst])
            elif name.startswith("start_"):
                jobs.add(int(name.split("_")[-1]))

    rm_sets = defaultdict(set)
    rt_sets = defaultdict(set)
    rm_use = defaultdict(float)
    rt_use = defaultdict(float)
    x_pattern = re.compile(r"x_(\d+)_(\d+)_(\d+)")

    for line in columns:
        tokens = line.split()
        if len(tokens) < 3:
            continue

        var_name = tokens[0]
        m_var = x_pattern.match(var_name)
        if not m_var:
            continue

        job = int(m_var.group(1))
        mode = int(m_var.group(2))
        start_t = int(m_var.group(3))
        jobs.add(job)

        pairs = tokens[1:]
        for i in range(0, len(pairs) - 1, 2):
            row_name = pairs[i]
            try:
                value = float(pairs[i + 1])
            except ValueError:
                continue

            if row_name.startswith("rescap_Mechaniker_"):
                const_t = int(row_name.split("_")[-1])
                rm_sets[(job, mode, start_t)].add(const_t)
                rm_use[(job, mode)] = max(rm_use[(job, mode)], value)
            elif row_name.startswith("rescap_Techniker_"):
                const_t = int(row_name.split("_")[-1])
                rt_sets[(job, mode, start_t)].add(const_t)
                rt_use[(job, mode)] = max(rt_use[(job, mode)], value)

    durations = defaultdict(int)
    for key, times in rm_sets.items():
        job, mode, _ = key
        durations[(job, mode)] = max(durations[(job, mode)], len(times))
    for key, times in rt_sets.items():
        job, mode, _ = key
        durations[(job, mode)] = max(durations[(job, mode)], len(times))

    capacities = {"Mechaniker": 0.0, "Techniker": 0.0}
    for line in rhs:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        pairs = tokens[1:]
        for i in range(0, len(pairs) - 1, 2):
            row_name = pairs[i]
            try:
                value = float(pairs[i + 1])
            except ValueError:
                continue
            if row_name.startswith("rescap_Mechaniker"):
                capacities["Mechaniker"] = max(capacities["Mechaniker"], value)
            elif row_name.startswith("rescap_Techniker"):
                capacities["Techniker"] = max(capacities["Techniker"], value)

    return {
        "jobs": sorted(jobs),
        "edges": sorted(set(edges)),
        "durations": dict(durations),
        "mechanic_use": dict(rm_use),
        "technician_use": dict(rt_use),
        "capacities": capacities,
    }


def _choose_business_mode(job: int, profile: dict) -> int:
    """Pick a representative execution mode for a job for display purposes.

    Selects the mode with the shortest duration, breaking ties by lowest total
    resource consumption, then by lowest mode number.

    Args:
        job: Job ID.
        profile: Parsed MPS structure as returned by ``_parse_mps_structure``.

    Returns:
        The selected mode number (1-indexed).
    """
    durations = profile["durations"]
    mech = profile["mechanic_use"]
    tech = profile["technician_use"]

    candidate_modes = sorted(
        {
            mode
            for (job_id, mode) in durations.keys()
            if job_id == job
        }
        | {
            mode
            for (job_id, mode) in mech.keys()
            if job_id == job
        }
        | {
            mode
            for (job_id, mode) in tech.keys()
            if job_id == job
        }
    )

    if not candidate_modes:
        return 1

    return min(
        candidate_modes,
        key=lambda mode: (
            durations.get((job, mode), 9999),
            mech.get((job, mode), 9999) + tech.get((job, mode), 9999),
            mode,
        ),
    )


def _earliest_start_schedule(profile: dict) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Compute ASAP start times using a topological forward pass on the precedence graph.

    Resource constraints are ignored; this gives the theoretical lower bound on each
    job's start time. One representative mode per job is chosen by
    ``_choose_business_mode``.

    Args:
        profile: Parsed MPS structure as returned by ``_parse_mps_structure``.

    Returns:
        A tuple containing:

        - dict[int, int]: Start time keyed by job ID.
        - dict[int, int]: Duration keyed by job ID.
        - dict[int, int]: Selected mode keyed by job ID.
    """
    jobs = profile["jobs"]
    edges = profile["edges"]

    selected_mode = {job: _choose_business_mode(job, profile) for job in jobs}
    duration_by_job = {
        job: max(1, int(profile["durations"].get((job, selected_mode[job]), 1)))
        for job in jobs
    }

    outgoing = defaultdict(list)
    indegree = {job: 0 for job in jobs}
    for src, dst in edges:
        if src in indegree and dst in indegree:
            outgoing[src].append(dst)
            indegree[dst] += 1

    queue = deque([job for job in jobs if indegree[job] == 0])
    topo = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for nxt in outgoing[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(topo) != len(jobs):
        topo = sorted(jobs)

    predecessors = defaultdict(list)
    for src, dst in edges:
        predecessors[dst].append(src)

    start_by_job = {job: 0 for job in jobs}
    for job in topo:
        if predecessors[job]:
            start_by_job[job] = max(
                start_by_job[pred] + duration_by_job[pred]
                for pred in predecessors[job]
            )

    return start_by_job, duration_by_job, selected_mode


def build_input_graph(input_path: str) -> go.Figure:
    """Build a two-subplot input view: ASAP timeline and resource demand profile.

    The timeline ignores resource limits and shows the earliest-possible schedule.
    The demand subplot shows per-timestep mechanic and technician consumption
    under that schedule.

    Args:
        input_path: Path to the MPS-format problem file.

    Returns:
        A Plotly figure with a Gantt chart (row 1) and resource demand lines (row 2).
    """
    profile = _parse_mps_structure(input_path)
    jobs = profile["jobs"]

    if not jobs:
        fig = go.Figure()
        fig.update_layout(
            title=f"No graphable data found in {Path(input_path).name if input_path else 'input file'}",
            template="plotly_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    start_by_job, duration_by_job, mode_by_job = _earliest_start_schedule(profile)
    jobs_sorted = sorted(jobs, key=lambda job: (start_by_job[job], job))
    finish_by_job = {job: start_by_job[job] + duration_by_job[job] for job in jobs}
    horizon = max(finish_by_job.values()) if finish_by_job else 1

    mech_demand = [0.0] * max(1, horizon)
    tech_demand = [0.0] * max(1, horizon)
    for job in jobs:
        mode = mode_by_job[job]
        start = start_by_job[job]
        finish = finish_by_job[job]
        mech_use = float(profile["mechanic_use"].get((job, mode), 0.0))
        tech_use = float(profile["technician_use"].get((job, mode), 0.0))
        for t in range(start, min(finish, len(mech_demand))):
            mech_demand[t] += mech_use
            tech_demand[t] += tech_use

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "Timeline (Ignoring Resource Limits)",
            "Resource Demand",
        ),
    )

    for (res_type, mode_val), color in _JOB_COLORS.items():
        mode_jobs = [
            job for job in jobs_sorted
            if mode_by_job[job] == mode_val
            and (
                (res_type == "mechanic"    and profile["mechanic_use"].get((job, mode_val), 0) > 0)
                or (res_type == "technician" and profile["technician_use"].get((job, mode_val), 0) > 0)
            )
        ]
        if not mode_jobs:
            continue
        fig.add_trace(
            go.Bar(
                x=[duration_by_job[job] for job in mode_jobs],
                y=[f"Job {job}" for job in mode_jobs],
                base=[start_by_job[job] for job in mode_jobs],
                orientation="h",
                marker={"color": color},
                customdata=[
                    [
                        mode_val,
                        profile["mechanic_use"].get((job, mode_val), 0),
                        profile["technician_use"].get((job, mode_val), 0),
                        duration_by_job[job],
                    ]
                    for job in mode_jobs
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>Start: %{base}<br>Duration: %{customdata[3]}<br>"
                    "Mode: %{customdata[0]}<br>"
                    "Mechanics: %{customdata[1]}<br>"
                    "Technicians: %{customdata[2]}<extra></extra>"
                ),
                name=f"{res_type.capitalize()} – Mode {mode_val}",
                legendgroup=res_type,
                legendgrouptitle_text=res_type.capitalize() if mode_val == 1 else None,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    x_axis = list(range(len(mech_demand)))

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=mech_demand,
            mode="lines",
            line={"color": _MECHANIC_LINE_COLOR, "width": 2},
            name="Mechanics",
            legendgroup="mechanic",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=tech_demand,
            mode="lines",
            line={"color": _TECHNICIAN_LINE_COLOR, "width": 2},
            name="Technicians",
            legendgroup="technician",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend={"orientation": "v", "x": 1.01, "y": 1},
        height=760,
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_yaxes(
        title_text="Jobs", row=1, col=1,
        categoryorder="array",
        categoryarray=[f"Job {job}" for job in reversed(jobs_sorted)],
    )
    fig.update_yaxes(title_text="Resource Units", row=2, col=1)

    return fig


def parse_mps_structure(input_path: str) -> dict:
    """Return the parsed MPS problem structure for callers outside this module.

    Args:
        input_path: Path to the MPS-format problem file.

    Returns:
        A dictionary with keys ``"jobs"``, ``"edges"``, ``"durations"``,
        ``"mechanic_use"``, ``"technician_use"``, and ``"capacities"``.
        See ``_parse_mps_structure`` for full details.
    """
    return _parse_mps_structure(input_path)


def _compute_demand(
    profile: dict,
    starts_by_job: dict[int, int],
    modes_by_job: dict[int, int],
) -> tuple[list[float], list[float], int]:
    """Compute per-timestep resource demand arrays from a solver schedule.

    Args:
        profile: Parsed MPS structure as returned by ``_parse_mps_structure``.
        starts_by_job: Start time keyed by job ID.
        modes_by_job: Execution mode keyed by job ID.

    Returns:
        A tuple containing:

        - list[float]: Mechanic demand at each timestep.
        - list[float]: Technician demand at each timestep.
        - int: Total horizon (length of both demand lists).
    """
    jobs = profile["jobs"]
    fallback = {job: _choose_business_mode(job, profile) for job in jobs}
    sel_mode = {job: int(modes_by_job.get(job, fallback[job])) for job in jobs}
    start = {job: int(starts_by_job.get(job, 0)) for job in jobs}
    dur = {job: max(1, int(profile["durations"].get((job, sel_mode[job]), 1))) for job in jobs}
    finish = {job: start[job] + dur[job] for job in jobs}
    horizon = max(finish.values()) if finish else 1

    mech = [0.0] * max(1, horizon)
    tech = [0.0] * max(1, horizon)
    for job in jobs:
        m = sel_mode[job]
        for t in range(start[job], min(finish[job], len(mech))):
            mech[t] += float(profile["mechanic_use"].get((job, m), 0.0))
            tech[t] += float(profile["technician_use"].get((job, m), 0.0))
    return mech, tech, horizon


# Colors used for each solver in the comparison chart
_SOLVER_COLORS = {
    "HiGHS":  "#E83E8C",
    "SCIP":   "#2d4376",
    "Stride": "#17BEBB",
}


def build_comparison_graph(
    input_path: str,
    solver_schedules: dict[str, tuple[dict[int, int], dict[int, int]]],
) -> go.Figure:
    """Build a resource demand comparison chart overlaying all solver solutions.

    Args:
        input_path: Path to the MPS input file.
        solver_schedules: Mapping of solver name to (starts_by_job, modes_by_job).

    Returns:
        A two-row Plotly figure: mechanic demand (top) and technician demand (bottom).
    """
    if not solver_schedules:
        fig = go.Figure()
        fig.update_layout(title="No solutions to compare.", template="plotly_white")
        return fig

    profile = _parse_mps_structure(input_path)
    if not profile["jobs"]:
        fig = go.Figure()
        fig.update_layout(title="No data.", template="plotly_white")
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Mechanic Demand", "Technician Demand"),
    )

    for solver_name, (starts, modes) in solver_schedules.items():
        color = _SOLVER_COLORS.get(solver_name, "#888888")
        mech, tech, _ = _compute_demand(profile, starts, modes)
        x = list(range(len(mech)))
        fig.add_trace(
            go.Scatter(
                x=x, y=mech, mode="lines", name=solver_name,
                line={"color": color, "width": 2},
                legendgroup=solver_name, showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=tech, mode="lines", name=solver_name,
                line={"color": color, "width": 2},
                legendgroup=solver_name, showlegend=False,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.15, "x": 0},
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Resource Units", row=1, col=1)
    fig.update_yaxes(title_text="Resource Units", row=2, col=1)
    return fig


def build_solution_graph(
    input_path: str,
    starts_by_job: dict[int, int] | None,
    modes_by_job: dict[int, int] | None,
    title: str,
) -> go.Figure:
    """Build a two-subplot solution view: solver timeline and resource demand profile.

    Falls back to ``build_input_graph`` if no schedule is available.

    Args:
        input_path: Path to the MPS-format problem file.
        starts_by_job: Start time keyed by job ID, or ``None`` if no solution.
        modes_by_job: Execution mode keyed by job ID, or ``None`` if no solution.
        title: Figure title displayed above the chart.

    Returns:
        A Plotly figure with a Gantt chart (row 1) and resource demand lines (row 2),
        including a dashed capacity line and idle-capacity fill on the demand subplot.
    """
    if not starts_by_job or not modes_by_job:
        fig = build_input_graph(input_path)
        fig.update_layout(title=f"{title}: Schedule Not Available (Showing Baseline)")
        return fig

    profile = _parse_mps_structure(input_path)
    jobs = profile["jobs"]
    if not jobs:
        return build_input_graph(input_path)

    fallback_mode = {job: _choose_business_mode(job, profile) for job in jobs}
    selected_mode = {
        job: int(modes_by_job.get(job, fallback_mode[job]))
        for job in jobs
    }
    start = {
        job: int(starts_by_job.get(job, 0))
        for job in jobs
    }
    duration = {
        job: max(1, int(profile["durations"].get((job, selected_mode[job]), 1)))
        for job in jobs
    }

    jobs_sorted = sorted(jobs, key=lambda job_id: (start[job_id], job_id))
    finish_by_job = {job: start[job] + duration[job] for job in jobs}
    horizon = max(finish_by_job.values()) if finish_by_job else 1

    mech_demand = [0.0] * max(1, horizon)
    tech_demand = [0.0] * max(1, horizon)
    for job in jobs:
        mode = selected_mode[job]
        s_t = start[job]
        f_t = finish_by_job[job]
        mech_use = float(profile["mechanic_use"].get((job, mode), 0.0))
        tech_use = float(profile["technician_use"].get((job, mode), 0.0))
        for t in range(s_t, min(f_t, len(mech_demand))):
            mech_demand[t] += mech_use
            tech_demand[t] += tech_use

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "Solver Timeline",
            "Resource Demand",
        ),
    )

    for (res_type, mode_val), color in _JOB_COLORS.items():
        mode_jobs = [
            job for job in jobs_sorted
            if selected_mode[job] == mode_val
            and (
                (res_type == "mechanic"    and profile["mechanic_use"].get((job, mode_val), 0) > 0)
                or (res_type == "technician" and profile["technician_use"].get((job, mode_val), 0) > 0)
            )
        ]
        if not mode_jobs:
            continue
        fig.add_trace(
            go.Bar(
                x=[duration[job] for job in mode_jobs],
                y=[f"Job {job}" for job in mode_jobs],
                base=[start[job] for job in mode_jobs],
                orientation="h",
                marker={"color": color},
                customdata=[
                    [
                        mode_val,
                        profile["mechanic_use"].get((job, mode_val), 0),
                        profile["technician_use"].get((job, mode_val), 0),
                        duration[job],
                    ]
                    for job in mode_jobs
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>Start: %{base}<br>Duration: %{customdata[3]}<br>"
                    "Mode: %{customdata[0]}<br>"
                    "Mechanics: %{customdata[1]}<br>"
                    "Technicians: %{customdata[2]}<extra></extra>"
                ),
                name=f"{res_type.capitalize()} – Mode {mode_val}",
                legendgroup=res_type,
                legendgrouptitle_text=res_type.capitalize() if mode_val == 1 else None,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    x_axis = list(range(len(mech_demand)))

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=mech_demand,
            mode="lines",
            line={"color": _MECHANIC_LINE_COLOR, "width": 2},
            name="Mechanics",
            legendgroup="mechanic",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=tech_demand,
            mode="lines",
            line={"color": _TECHNICIAN_LINE_COLOR, "width": 2},
            name="Technicians",
            legendgroup="technician",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend={"orientation": "v", "x": 1.01, "y": 1},
        height=760,
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_yaxes(
        title_text="Jobs", row=1, col=1,
        categoryorder="array",
        categoryarray=[f"Job {job}" for job in reversed(jobs_sorted)],
    )
    fig.update_yaxes(title_text="Resource Units", row=2, col=1)

    return fig