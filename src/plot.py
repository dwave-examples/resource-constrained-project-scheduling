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

from collections import defaultdict, deque
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import parse_mps_structure

# Colors keyed by (resource_type, mode): shade encodes mode speed within each resource family
_JOB_COLORS: dict[tuple[str, int], str] = {
    ("mechanic", 1): "#9BC2F1",
    ("mechanic", 2): "#3886E3",
    ("mechanic", 3): "#1757A5",
    ("technician", 1): "#F8AF7B",
    ("technician", 2): "#EF6B0D",
    ("technician", 3): "#A64A09",
}
# Mid-shade of each family used for the resource demand lines
_MECHANIC_LINE_COLOR = _JOB_COLORS[("mechanic", 2)]
_TECHNICIAN_LINE_COLOR = _JOB_COLORS[("technician", 2)]

# Colors used for each solver in the comparison chart
_SOLVER_COLORS = {
    "HiGHS": "#E83E8C",
    "SCIP": "#2d4376",
    "Stride": "#17BEBB",
}


def _hiring_cost_text(r_m: int, r_t: int, mech_rate: float, tech_rate: float) -> str:
    """Return an HTML hiring cost breakdown for use in a Plotly annotation.

    Args:
        r_m: Number of mechanics hired.
        r_t: Number of technicians hired.
        mech_rate: Cost per mechanic.
        tech_rate: Cost per technician.

    Returns:
        A three-line HTML string with per-resource costs and a bold total.
    """
    mech_cost = int(r_m * mech_rate)
    tech_cost = int(r_t * tech_rate)
    total = mech_cost + tech_cost
    return (
        f"{r_m} mechanics @ {int(mech_rate)} = {mech_cost}<br>"
        f"{r_t} technicians @ {int(tech_rate)} = {tech_cost}<br>"
        f"<b>Total: {total}</b>"
    )


def _choose_business_mode(job: int, profile: dict) -> int:
    """Pick a representative execution mode for a job for display purposes.

    Selects the mode with the shortest duration, breaking ties by lowest total
    resource consumption, then by lowest mode number.

    Args:
        job: Job ID.
        profile: Parsed MPS structure as returned by ``parse_mps_structure``.

    Returns:
        The selected mode number (1-indexed).
    """
    durations = profile["durations"]
    mech = profile["mechanic_use"]
    tech = profile["technician_use"]

    candidate_modes = sorted(
        {mode for (job_id, mode) in durations.keys() if job_id == job}
        | {mode for (job_id, mode) in mech.keys() if job_id == job}
        | {mode for (job_id, mode) in tech.keys() if job_id == job}
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


def _earliest_start_schedule(
    profile: dict,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Compute ASAP start times using a topological forward pass on the precedence graph.

    Resource constraints are ignored; this gives the theoretical lower bound on each
    job's start time. One representative mode per job is chosen by
    ``_choose_business_mode``.

    Args:
        profile: Parsed MPS structure as returned by ``parse_mps_structure``.

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
        job: max(1, int(profile["durations"].get((job, selected_mode[job]), 1))) for job in jobs
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
                start_by_job[pred] + duration_by_job[pred] for pred in predecessors[job]
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
    profile = parse_mps_structure(input_path)
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

    hire_rates = profile.get("hire_rates", {"Mechaniker": 100.0, "Techniker": 51.0})
    r_m = math.ceil(max(mech_demand, default=0))
    r_t = math.ceil(max(tech_demand, default=0))
    upper_bounds = profile.get("upper_bounds", {})
    display_horizon = max(horizon, max(upper_bounds.values(), default=horizon))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "Greedy Schedule (Earliest Start)",
            "Resource Demand",
        ),
    )

    _grey = "#DDDDDD"
    fig.add_trace(
        go.Bar(
            x=[start_by_job[job] for job in jobs_sorted],
            y=[f"Job {job}" for job in jobs_sorted],
            base=0,
            orientation="h",
            marker={"color": _grey, "line": {"width": 0}},
            hoverinfo="skip",
            showlegend=False,
            name="",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[max(0, display_horizon - upper_bounds.get(job, display_horizon)) for job in jobs_sorted],
            y=[f"Job {job}" for job in jobs_sorted],
            base=[upper_bounds.get(job, display_horizon) for job in jobs_sorted],
            orientation="h",
            marker={"color": _grey, "line": {"width": 0}},
            hoverinfo="skip",
            showlegend=False,
            name="",
        ),
        row=1,
        col=1,
    )

    for (res_type, mode_val), color in _JOB_COLORS.items():
        mode_jobs = [
            job
            for job in jobs_sorted
            if mode_by_job[job] == mode_val
            and (
                (res_type == "mechanic" and profile["mechanic_use"].get((job, mode_val), 0) > 0)
                or (
                    res_type == "technician"
                    and profile["technician_use"].get((job, mode_val), 0) > 0
                )
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
                        f"Mechanics: {int(profile['mechanic_use'].get((job, mode_val), 0))}<br>"
                        if profile["mechanic_use"].get((job, mode_val), 0) > 0
                        else "",
                        f"Technicians: {int(profile['technician_use'].get((job, mode_val), 0))}<br>"
                        if profile["technician_use"].get((job, mode_val), 0) > 0
                        else "",
                        duration_by_job[job],
                    ]
                    for job in mode_jobs
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>Start: %{base}<br>Duration: %{customdata[3]}<br>"
                    "%{customdata[1]}%{customdata[2]}<extra></extra>"
                ),
                name=f"{mode_val} {res_type.capitalize()}{'s'[:mode_val^1]}",
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
        barmode="overlay",
        margin={"l": 20, "r": 180, "t": 30, "b": 90},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend={"orientation": "v", "x": 1, "y": 0},
        height=760,
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_yaxes(
        title_text="Jobs",
        row=1,
        col=1,
        categoryorder="array",
        categoryarray=[f"Job {job}" for job in reversed(jobs_sorted)],
    )
    fig.update_yaxes(title_text="Resource Units", row=2, col=1)
    fig.add_annotation(
        text="<i>Hover over a job to highlight<br>its predecessors</i>",
        xref="paper",
        yref="paper",
        x=1,
        y=0.7,
        showarrow=False,
        align="left",
        font={"size": 12, "color": "#666666"},
        xanchor="left",
        yanchor="bottom",
    )
    fig.add_annotation(
        text=(
            "<b>Cost (unoptimized):</b><br>"
            + _hiring_cost_text(r_m, r_t, hire_rates["Mechaniker"], hire_rates["Techniker"])
        ),
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        showarrow=False,
        align="left",
        font={"size": 13},
        xanchor="left",
        yanchor="top",
    )

    return fig


def _compute_demand(
    profile: dict,
    starts_by_job: dict[int, int],
    modes_by_job: dict[int, int],
) -> tuple[list[float], list[float], int]:
    """Compute per-timestep resource demand arrays from a solver schedule.

    Args:
        profile: Parsed MPS structure as returned by ``parse_mps_structure``.
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


def _input_demand_peaks(input_path: str) -> tuple[float, float]:
    """Return the peak mechanic and technician demand under the ASAP (input) schedule.

    Args:
        input_path: Path to the MPS-format problem file.

    Returns:
        A tuple of (peak_mech, peak_tech) demand values.
    """
    profile = parse_mps_structure(input_path)
    if not profile["jobs"]:
        return 0.0, 0.0
    start_by_job, duration_by_job, mode_by_job = _earliest_start_schedule(profile)
    finish_by_job = {job: start_by_job[job] + duration_by_job[job] for job in profile["jobs"]}
    horizon = max(finish_by_job.values()) if finish_by_job else 1
    mech_demand = [0.0] * max(1, horizon)
    tech_demand = [0.0] * max(1, horizon)
    for job in profile["jobs"]:
        mode = mode_by_job[job]
        s = start_by_job[job]
        f = finish_by_job[job]
        for t in range(s, min(f, len(mech_demand))):
            mech_demand[t] += float(profile["mechanic_use"].get((job, mode), 0.0))
            tech_demand[t] += float(profile["technician_use"].get((job, mode), 0.0))
    return max(mech_demand, default=0.0), max(tech_demand, default=0.0)


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

    profile = parse_mps_structure(input_path)
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
                x=x,
                y=mech,
                mode="lines",
                name=solver_name,
                line={"color": color, "width": 2},
                legendgroup=solver_name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=tech,
                mode="lines",
                name=solver_name,
                line={"color": color, "width": 2},
                legendgroup=solver_name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
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

    profile = parse_mps_structure(input_path)
    jobs = profile["jobs"]
    if not jobs:
        return build_input_graph(input_path)

    fallback_mode = {job: _choose_business_mode(job, profile) for job in jobs}
    selected_mode = {job: int(modes_by_job.get(job, fallback_mode[job])) for job in jobs}
    start = {job: int(starts_by_job.get(job, 0)) for job in jobs}
    duration = {
        job: max(1, int(profile["durations"].get((job, selected_mode[job]), 1))) for job in jobs
    }

    asap_start, _, _ = _earliest_start_schedule(profile)

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

    hire_rates = profile.get("hire_rates", {"Mechaniker": 100.0, "Techniker": 51.0})
    r_m = math.ceil(max(mech_demand, default=0))
    r_t = math.ceil(max(tech_demand, default=0))
    upper_bounds = profile.get("upper_bounds", {})
    display_horizon = max(horizon, max(upper_bounds.values(), default=horizon))

    fig = go.Figure()

    _grey = "#DDDDDD"
    fig.add_trace(
        go.Bar(
            x=[asap_start.get(job, 0) for job in jobs_sorted],
            y=[f"Job {job}" for job in jobs_sorted],
            base=0,
            orientation="h",
            marker={"color": _grey, "line": {"width": 0}},
            hoverinfo="skip",
            showlegend=False,
            name="",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=[max(0, display_horizon - upper_bounds.get(job, display_horizon)) for job in jobs_sorted],
            y=[f"Job {job}" for job in jobs_sorted],
            base=[upper_bounds.get(job, display_horizon) for job in jobs_sorted],
            orientation="h",
            marker={"color": _grey, "line": {"width": 0}},
            hoverinfo="skip",
            showlegend=False,
            name="",
        ),
    )

    for (res_type, mode_val), color in _JOB_COLORS.items():
        mode_jobs = [
            job
            for job in jobs_sorted
            if selected_mode[job] == mode_val
            and (
                (res_type == "mechanic" and profile["mechanic_use"].get((job, mode_val), 0) > 0)
                or (
                    res_type == "technician"
                    and profile["technician_use"].get((job, mode_val), 0) > 0
                )
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
                        f"Mechanics: {int(profile['mechanic_use'].get((job, mode_val), 0))}<br>"
                        if profile["mechanic_use"].get((job, mode_val), 0) > 0
                        else "",
                        f"Technicians: {int(profile['technician_use'].get((job, mode_val), 0))}<br>"
                        if profile["technician_use"].get((job, mode_val), 0) > 0
                        else "",
                        duration[job],
                    ]
                    for job in mode_jobs
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>Start: %{base}<br>Duration: %{customdata[3]}<br>"
                    "%{customdata[1]}%{customdata[2]}<extra></extra>"
                ),
                name=f"{mode_val} {res_type.capitalize()}{'s'[:mode_val^1]}",
                legendgroup=res_type,
                legendgrouptitle_text=res_type.capitalize() if mode_val == 1 else None,
                showlegend=True,
            ),
        )

    fig.update_layout(
        title=title + "<br><sup><i>Hover over a job to highlight its predecessors</i></sup>",
        template="plotly_white",
        barmode="overlay",
        margin={"l": 20, "r": 180, "t": 40, "b": 90},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend={"orientation": "v", "x": 1, "y": 0},
        height=500,
    )
    fig.update_xaxes(title_text="Time", showticklabels=True)
    fig.update_yaxes(
        title_text="Jobs",
        categoryorder="array",
        categoryarray=[f"Job {job}" for job in reversed(jobs_sorted)],
    )
    fig.add_annotation(
        text=(
            "<b>Hiring cost (optimized):</b><br>"
            + _hiring_cost_text(r_m, r_t, hire_rates["Mechaniker"], hire_rates["Techniker"])
        ),
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        showarrow=False,
        align="left",
        font={"size": 13},
        xanchor="left",
        yanchor="top",
    )

    return fig
