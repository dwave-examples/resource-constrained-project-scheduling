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

import dash
from dash import html
from dash import MATCH
from dash.dependencies import Input, Output, State

from demo_configs import INPUTS
from demo_interface import generate_table
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
    Output("input", "children"),
    inputs=[
        Input("runs", "value"),
        Input("solver-selection", "value"),
        Input("solver-time-limit", "value"),
    ],
)
def render_initial_state(runs: int, selection: list[str], time_limit: float) -> html.Div:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        runs: Number of repeated runs per formulation.
        selection: Selected formulation IDs.
        time_limit: The per-solver time limit.

    Returns:
        The content of the input tab.
    """
    selected_count = len(selection or [])
    return html.Div(
        [
            html.H3("Ready To Compare Formulations"),
            html.P(
                f"Configured {selected_count} formulation(s), "
                f"{runs} run(s) each, {time_limit} second time limit per run."
            ),
            html.P("Click Run Optimization to generate a summary table and detailed run table."),
        ]
    )


@dash.callback(
    # The Outputs below must align with the return values of the function.
    Output("results", "children"),
    Output("problem-details", "children"),
    background=True,
    inputs=[
        # The first string in the Input/State elements below must match an id in demo_interface.py
        # Remove or alter the following id's to match any changes made to demo_interface.py
        Input("run-button", "n_clicks"),
        State("solver-selection", "value"),
        State("solver-time-limit", "value"),
        State("runs", "value"),
    ],
    running=[
        (Output("cancel-button", "style"), {}, {"display": "none"}),  # Show/hide cancel button.
        (Output("run-button", "style"), {"display": "none"}, {}),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "children"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
        (Output("run-in-progress", "data"), True, False),  # Can block certain callbacks.
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    # The parameters below must match the `Input` and `State` variables found
    # in the `inputs` list above.
    run_click: int,
    solver_selection: list[str],
    time_limit: float,
    runs: int,
) -> tuple[html.Div, html.Table]:
    """Runs the optimization and updates UI accordingly.

    This is the main function which is called when the ``Run Optimization`` button is clicked.
    This function takes in all form values and runs the optimization, updates the run/cancel
    buttons, deactivates (and reactivates) the results tab, and updates all relevant HTML
    components.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        solver_selection: Selected formulations to run.
        time_limit: The solver time limit.
        runs: Number of repeated runs.

    Returns:
        A tuple containing:

        - html.Div: The results component to display in the results tab.
        - list: List of the table rows for the problem details table.
    """
    selection = solver_selection or []
    if not selection:
        results = html.Div([html.P("Select at least one formulation before running.")])
        problem_details_table = generate_table(
            {
                "Selected Formulations": [0],
                "Runs Per Formulation": [runs],
                "Time Limit (s)": [time_limit],
            }
        )
        return results, problem_details_table

    run_rows = compare_formulations(selection, float(time_limit), int(runs), input_path=INPUTS[0])
    summary_rows = summarize_runs(run_rows)

    summary_table = generate_table(
        {
            "Formulation": [row["formulation"] for row in summary_rows],
            "Runs": [row["runs"] for row in summary_rows],
            "OK Runs": [row["ok_runs"] for row in summary_rows],
            "Best Energy": [row["best_energy"] for row in summary_rows],
            "Avg Energy": [row["avg_energy"] for row in summary_rows],
        }
    )
    detailed_table = generate_table(
        {
            "Formulation": [row["formulation"] for row in run_rows],
            "Run": [row["run"] for row in run_rows],
            "Status": [row["status"] for row in run_rows],
            "Energy": [row["energy"] if row["energy"] is not None else "n/a" for row in run_rows],
            "OK": ["yes" if row["ok"] else "no" for row in run_rows],
        }
    )

    results = html.Div(
        [
            html.H3("Comparison Summary"),
            summary_table,
            html.H3("Run-Level Details"),
            detailed_table,
        ]
    )

    problem_details_table = generate_table(
        {
            "Selected Formulations": [len(selection)],
            "Runs Per Formulation": [runs],
            "Time Limit (s)": [time_limit],
        }
    )

    return results, problem_details_table
