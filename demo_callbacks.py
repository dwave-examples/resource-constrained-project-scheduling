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

from __future__ import annotations

from typing import NamedTuple

import dash
from dash import ctx, html
from dash import MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from demo_configs import KNOWN_OPTIMA
from demo_interface import (
    comparison_panel,
    comparison_summary_table,
    results_layout,
    solver_not_selected_panel,
    solver_solution_panel,
    waiting_panel,
)
from src.demo_enums import SolverType
from src.plot import build_input_graph, build_solution_graph, build_comparison_graph
from src.demo_runner import compare_formulations, summarize_runs


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    Output({"type": "collapse-trigger", "index": MATCH}, "aria-expanded"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> tuple[str, str]:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger: The (total) number of times a collapse button has been clicked.
        to_collapse_class: Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        A tuple containing:

        - str: The new class name of the thing to collapse.
        - str: The aria-expanded value.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes), "true"
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed", "false"


@dash.callback(
    Output("input-graph", "figure"),
    inputs=[
        Input("input-file-select", "value"),
    ],
)
def render_initial_state(input_file: str) -> html.Div:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        input_file: Selected input file path.

    Returns:
        The content of the input tab.
    """
    selected_input = input_file or ""
    return build_input_graph(selected_input)


# ---------------------------------------------------------------------------
# 1. Run/cancel state callback — mirrors update_tab_loading_state in the example.
#    Sets tab labels/disabled, shows/hides buttons, and sets the running-* flags
#    that trigger the three independent background solver callbacks below.
# ---------------------------------------------------------------------------

@dash.callback(
    Output("results-tab", "children"),
    Output("results-tab", "disabled"),
    Output("running-highs", "data"),
    Output("highs-tab", "children"),
    Output("highs-tab", "disabled"),
    Output("highs-tab", "className"),
    Output("running-scip", "data"),
    Output("scip-tab", "children"),
    Output("scip-tab", "disabled"),
    Output("scip-tab", "className"),
    Output("running-stride", "data"),
    Output("stride-tab", "children"),
    Output("stride-tab", "disabled"),
    Output("stride-tab", "className"),
    Output("run-button", "style"),
    Output("cancel-button", "style"),
    Output("tabs", "value"),
    inputs=[
        Input("run-button", "n_clicks"),
        Input("cancel-button", "n_clicks"),
        State("solver-selection", "value"),
    ],
    prevent_initial_call=True,
)
def update_run_state(
    run_click: int,
    cancel_click: int,
    solver_selection: list[str],
) -> tuple:
    """Update tab labels, disabled state, and running flags on run/cancel.

    Args:
        run_click: Number of times run button has been clicked.
        cancel_click: Number of times cancel button has been clicked.
        solver_selection: Currently selected solver values.

    Returns:
        Tuple of outputs controlling tab state, button visibility, and
        running-* flags for each solver.
    """
    selection = solver_selection or []
    highs_selected = str(SolverType.HIGHS.value) in selection
    scip_selected  = str(SolverType.SCIP.value)  in selection
    stride_selected = str(SolverType.STRIDE.value) in selection

    if ctx.triggered_id == "run-button" and run_click:
        loading = ("Loading...", True)
        return (
            *(loading if highs_selected or scip_selected or stride_selected else ("Results", True)),
            highs_selected,
            *(loading if highs_selected else ("HiGHS", True)),
            "",  # reset highs className
            scip_selected,
            *(loading if scip_selected else ("SCIP", True)),
            "",  # reset scip className
            stride_selected,
            *(loading if stride_selected else ("Stride", True)),
            "",  # reset stride className
            {"display": "none"},  # hide run button
            {},                   # show cancel button
            "input-tab",
        )

    if ctx.triggered_id == "cancel-button" and cancel_click:
        return (
            "Results", False,
            False,
            "HiGHS", False, "",
            False,
            "SCIP", False, "",
            False,
            "Stride", False, "",
            {},                   # show run button
            {"display": "none"},  # hide cancel button
            dash.no_update,
        )

    raise PreventUpdate


# ---------------------------------------------------------------------------
# 2. Button-visibility watchdog — restores run/cancel when all solvers finish.
#    Mirrors update_button_visibility in the example.
# ---------------------------------------------------------------------------

@dash.callback(
    Output("run-button", "style", allow_duplicate=True),
    Output("cancel-button", "style", allow_duplicate=True),
    Output("run-in-progress", "data"),
    inputs=[
        Input("running-highs", "data"),
        Input("running-scip", "data"),
        Input("running-stride", "data"),
    ],
    prevent_initial_call=True,
)
def update_button_visibility(
    running_highs: bool,
    running_scip: bool,
    running_stride: bool,
) -> tuple[dict, dict, bool]:
    """Restore the run button only once every running solver has finished.

    Args:
        running_highs: Whether the HiGHS callback is still running.
        running_scip: Whether the SCIP callback is still running.
        running_stride: Whether the Stride callback is still running.

    Returns:
        A tuple containing:

        - dict: Run button style.
        - dict: Cancel button style.
        - bool: Whether any run is in progress.
    """
    if running_highs or running_scip or running_stride:
        return {"display": "none"}, {}, True
    return {}, {"display": "none"}, False


# ---------------------------------------------------------------------------
# Helper shared by the three solver background callbacks.
# ---------------------------------------------------------------------------

def _solver_panel(label: str, rows: list[dict], input_path: str) -> html.Div:
    """Build the content for a single solver's results tab.

    Args:
        label: Human-readable solver name.
        rows: Run-level result rows for this solver.
        input_path: Path to the input MPS file.

    Returns:
        A Div containing the solution graph and run details table.
    """
    best = min(
        rows,
        key=lambda row: row["energy"] if isinstance(row.get("energy"), (int, float)) else float("inf"),
    )
    has_solution = bool(best.get("starts")) and bool(best.get("modes"))
    figure = (
        build_solution_graph(
            input_path,
            best.get("starts", {}),
            best.get("modes", {}),
            title=f"{label} Best Solution View",
        )
        if has_solution else None
    )
    return solver_solution_panel(has_solution, figure)


def _solver_tab_class(rows: list[dict]) -> str:
    """Return the CSS class for a solver tab based on whether any run succeeded."""
    return "tab-success" if any(row.get("ok") for row in rows) else "tab-fail"


# ---------------------------------------------------------------------------
# 3–5. One background callback per solver — all triggered by the same run
#      button click and all execute concurrently.
# ---------------------------------------------------------------------------

class RunHiGHSReturn(NamedTuple):
    """Return type for run_highs."""
    highs_results: html.Div     = dash.no_update
    highs_store: dict            = dash.no_update
    highs_tab_label: str         = dash.no_update
    highs_tab_disabled: bool     = dash.no_update
    running_highs: bool          = dash.no_update
    highs_tab_class: str         = dash.no_update


@dash.callback(
    Output("highs-results", "children"),
    Output("highs-store", "data"),
    Output("highs-tab", "children", allow_duplicate=True),
    Output("highs-tab", "disabled", allow_duplicate=True),
    Output("running-highs", "data", allow_duplicate=True),
    Output("highs-tab", "className", allow_duplicate=True),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("solver-selection", "value"),
        State("solver-time-limit", "value"),
        State("runs", "value"),
        State("input-file-select", "value"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_highs(
    run_click: int,
    solver_selection: list[str],
    time_limit: float,
    runs: int,
    input_file: str,
) -> RunHiGHSReturn:
    """Run HiGHS independently in the background.

    Args:
        run_click: Number of times run button has been clicked.
        solver_selection: Currently selected solver values.
        time_limit: Solver time limit in seconds.
        runs: Number of repeated runs.
        input_file: Path to the selected input file.

    Returns:
        RunHiGHSReturn with updated tab content, store data, and running flag.
    """
    if str(SolverType.HIGHS.value) not in (solver_selection or []):
        return RunHiGHSReturn(
            highs_results=solver_not_selected_panel("HiGHS"),
            highs_store={"run_click": run_click, "rows": []},
            highs_tab_label="HiGHS",
            highs_tab_disabled=False,
            running_highs=False,
        )

    selected_input = input_file or ""
    rows = compare_formulations(
        [str(SolverType.HIGHS.value)], float(time_limit), int(runs), input_path=selected_input
    )
    return RunHiGHSReturn(
        highs_results=_solver_panel("HiGHS (MILP)", rows, selected_input),
        highs_store={"run_click": run_click, "rows": rows},
        highs_tab_label="HiGHS",
        highs_tab_disabled=False,
        running_highs=False,
        highs_tab_class=_solver_tab_class(rows),
    )


class RunSCIPReturn(NamedTuple):
    """Return type for run_scip."""
    scip_results: html.Div      = dash.no_update
    scip_store: dict             = dash.no_update
    scip_tab_label: str          = dash.no_update
    scip_tab_disabled: bool      = dash.no_update
    running_scip: bool           = dash.no_update
    scip_tab_class: str          = dash.no_update


@dash.callback(
    Output("scip-results", "children"),
    Output("scip-store", "data"),
    Output("scip-tab", "children", allow_duplicate=True),
    Output("scip-tab", "disabled", allow_duplicate=True),
    Output("running-scip", "data", allow_duplicate=True),
    Output("scip-tab", "className", allow_duplicate=True),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("solver-selection", "value"),
        State("solver-time-limit", "value"),
        State("runs", "value"),
        State("input-file-select", "value"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_scip(
    run_click: int,
    solver_selection: list[str],
    time_limit: float,
    runs: int,
    input_file: str,
) -> RunSCIPReturn:
    """Run SCIP independently in the background.

    Args:
        run_click: Number of times run button has been clicked.
        solver_selection: Currently selected solver values.
        time_limit: Solver time limit in seconds.
        runs: Number of repeated runs.
        input_file: Path to the selected input file.

    Returns:
        RunSCIPReturn with updated tab content, store data, and running flag.
    """
    if str(SolverType.SCIP.value) not in (solver_selection or []):
        return RunSCIPReturn(
            scip_results=solver_not_selected_panel("SCIP"),
            scip_store={"run_click": run_click, "rows": []},
            scip_tab_label="SCIP",
            scip_tab_disabled=False,
            running_scip=False,
        )

    selected_input = input_file or ""
    rows = compare_formulations(
        [str(SolverType.SCIP.value)], float(time_limit), int(runs), input_path=selected_input
    )
    return RunSCIPReturn(
        scip_results=_solver_panel("SCIP (MILP)", rows, selected_input),
        scip_store={"run_click": run_click, "rows": rows},
        scip_tab_label="SCIP",
        scip_tab_disabled=False,
        running_scip=False,
        scip_tab_class=_solver_tab_class(rows),
    )


class RunStrideReturn(NamedTuple):
    """Return type for run_stride."""
    stride_results: html.Div    = dash.no_update
    stride_store: dict           = dash.no_update
    stride_tab_label: str        = dash.no_update
    stride_tab_disabled: bool    = dash.no_update
    running_stride: bool         = dash.no_update
    stride_tab_class: str        = dash.no_update


@dash.callback(
    Output("stride-results", "children"),
    Output("stride-store", "data"),
    Output("stride-tab", "children", allow_duplicate=True),
    Output("stride-tab", "disabled", allow_duplicate=True),
    Output("running-stride", "data", allow_duplicate=True),
    Output("stride-tab", "className", allow_duplicate=True),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("solver-selection", "value"),
        State("solver-time-limit", "value"),
        State("runs", "value"),
        State("input-file-select", "value"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_stride(
    run_click: int,
    solver_selection: list[str],
    time_limit: float,
    runs: int,
    input_file: str,
) -> RunStrideReturn:
    """Run Stride independently in the background.

    Args:
        run_click: Number of times run button has been clicked.
        solver_selection: Currently selected solver values.
        time_limit: Solver time limit in seconds.
        runs: Number of repeated runs.
        input_file: Path to the selected input file.

    Returns:
        RunStrideReturn with updated tab content, store data, and running flag.
    """
    if str(SolverType.STRIDE.value) not in (solver_selection or []):
        return RunStrideReturn(
            stride_results=solver_not_selected_panel("Stride"),
            stride_store={"run_click": run_click, "rows": []},
            stride_tab_label="Stride",
            stride_tab_disabled=False,
            running_stride=False,
        )

    selected_input = input_file or ""
    rows = compare_formulations(
        [str(SolverType.STRIDE.value)], float(time_limit), int(runs), input_path=selected_input
    )
    return RunStrideReturn(
        stride_results=_solver_panel("Stride Quantum Hybrid", rows, selected_input),
        stride_store={"run_click": run_click, "rows": rows},
        stride_tab_label="Stride",
        stride_tab_disabled=False,
        running_stride=False,
        stride_tab_class=_solver_tab_class(rows),
    )


# ---------------------------------------------------------------------------
# 6. Aggregate summary — fires each time a solver store updates, building the
#    Results tab from whichever solvers have completed so far.
# ---------------------------------------------------------------------------

@dash.callback(
    Output("results", "children"),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("results-tab", "children", allow_duplicate=True),
    inputs=[
        Input("highs-store", "data"),
        Input("scip-store", "data"),
        Input("stride-store", "data"),
        State("run-button", "n_clicks"),
        State("input-file-select", "value"),
    ],
    prevent_initial_call=True,
)
def render_aggregate_results(
    highs_store: dict,
    scip_store: dict,
    stride_store: dict,
    run_click: int,
    input_file: str,
) -> tuple[html.Div, bool, str]:
    """Build the Results summary tab from completed solver stores.

    Fires each time any solver store updates, so the summary grows as
    solvers finish in parallel.

    Args:
        highs_store: Latest data stored by the HiGHS callback.
        scip_store: Latest data stored by the SCIP callback.
        stride_store: Latest data stored by the Stride callback.
        run_click: Number of times the run button has been clicked.
        input_file: Path to the selected input file.

    Returns:
        A tuple containing:

        - html.Div: Summary and run-detail tables for all finished solvers.
        - bool: Whether the Results tab should remain disabled.
        - str: Results tab label.
    """
    selected_input = input_file or ""

    rows: list[dict] = []
    store_rows: dict[str, list[dict]] = {}
    for name, store in [("HiGHS", highs_store or {}), ("SCIP", scip_store or {}), ("Stride", stride_store or {})]:
        if store.get("run_click") == run_click:
            store_rows[name] = store.get("rows", [])
            rows.extend(store_rows[name])
        else:
            store_rows[name] = []

    if not rows:
        return waiting_panel(), True, "Results"

    summary_rows = summarize_runs(rows)
    known_optimal = KNOWN_OPTIMA.get(selected_input)

    # Build per-solver best schedules for the comparison chart.
    # JSON round-trip turns int keys into strings — convert them back.
    solver_schedules: dict[str, tuple[dict, dict]] = {}
    for name, s_rows in store_rows.items():
        ok_rows = [r for r in s_rows if r.get("ok") and r.get("starts")]
        if ok_rows:
            best = min(ok_rows, key=lambda r: r.get("energy") or float("inf"))
            starts = {int(k): v for k, v in best["starts"].items()}
            modes  = {int(k): v for k, v in best["modes"].items()}
            solver_schedules[name] = (starts, modes)

    ok_energies = [
        row["best_energy"] for row in summary_rows
        if row["ok_runs"] > 0 and row["best_energy"] is not None
    ]
    min_energy = min(ok_energies) if ok_energies else None

    summary_table = comparison_summary_table(summary_rows, min_energy, known_optimal)
    fig = build_comparison_graph(selected_input, solver_schedules) if solver_schedules else None
    results = results_layout(comparison_panel(fig), summary_table)

    return results, False, "Results"
