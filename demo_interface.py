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

"""This file stores the Dash HTML layout for the app."""
from __future__ import annotations
from enum import EnumMeta
from pathlib import Path

from dash import dcc, html
import dash_mantine_components as dmc

from demo_configs import (
    DESCRIPTION,
    INPUTS,
    MAIN_HEADER,
    RUNS,
    SOLVER_TIME,
    THUMBNAIL,
)
from src.demo_enums import SolverType

THEME_COLOR = "#2d4376"


def slider(label: str, id: str, config: dict) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configurations, see dmc.Slider Dash Mantine docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Slider(
                id=id,
                className="slider",
                **config,
                marks=[
                    {"value": config["min"], "label": f'{config["min"]}'},
                    {"value": config["max"], "label": f'{config["max"]}'},
                ],
                labelAlwaysOn=True,
                thumbLabel=f"{label} slider",
                color=THEME_COLOR,
            ),
        ],
    )


def range_slider(label: str, id: str, config: dict) -> html.Div:
    """Range slider element for value selection.

    Args:
        label: The title that goes above the range slider.
        id: A unique selector for this element.
        config: A dictionary of range slider configurations, see dmc.RangeSlider Dash Mantine docs.
    """
    return html.Div(
        className="rangeslider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.RangeSlider(
                id=id,
                className="slider",
                **config,
                marks=[
                    {"value": config["min"], "label": f'{config["min"]}'},
                    {"value": config["max"], "label": f'{config["max"]}'},
                ],
                labelAlwaysOn=True,
                thumbFromLabel=f"{label} slider start",
                thumbToLabel=f"{label} slider end",
                color=THEME_COLOR,
            )
        ]
    )


def dropdown(label: str, id: str, options: list, value: str | None = None) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: Optional selected value.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=value if value is not None else options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )


def checklist(label: str, id: str, options: list, values: list, inline: bool = True) -> html.Div:
    """Checklist element for option selection.

    Args:
        label: The title that goes above the checklist.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        values: A list of values that should be preselected in the checklist.
        inline: Whether the options of the checklist are displayed beside or below each other.
    """
    return html.Div(
        className="checklist-wrapper",
        children=[
            dmc.CheckboxGroup(
                id=id,
                className=f"checklist{' checklist--inline' if inline else ''}",
                label=label,
                value=values,
                children=dmc.Group(
                    [
                        dmc.Checkbox(label=option["label"], value=option["value"], color=THEME_COLOR)
                        for option in options
                    ],
                ),
            ),
        ],
    )


def checkbox(label: str, id: str, checked: bool) -> html.Div:
    """Checkbox element.

    Args:
        label: The title that goes above the checkbox.
        id: A unique selector for this element.
        checked: Whether the checkbox is checked or not.
    """
    return html.Div(
        className="checkbox-wrapper",
        children=[
            dmc.Checkbox(
                id=id,
                label=label,
                checked=checked,
                color=THEME_COLOR,
            )
        ],
    )


def radio(label: str, id: str, options: list, value: str, inline: bool = True) -> html.Div:
    """Radio element for option selection.

    Args:
        label: The title that goes above the radio.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: The value of the radio that should be preselected.
        inline: Whether the options are displayed beside or below each other.
    """
    return html.Div(
        className="radio-wrapper",
        children=[
            dmc.RadioGroup(
                id=id,
                className=f"radio{' radio--inline' if inline else ''}",
                label=label,
                value=value,
                children=dmc.Group(
                    [
                        dmc.Radio(option["label"], value=option["value"], color=THEME_COLOR)
                        for option in options
                    ]
                ),
            ),
        ],
    )


def input(label: str, id: str, configs: dict, type: str="number") -> html.Div:
    """Input element for either text or number input.

    Args:
        label: The title that goes above the input.
        id: A unique selector for this element.
        configs: A dictionary of configurations for the input element.
        type: The type of input, either "number" or "text".
    """
    return html.Div(
        className="input-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.TextInput(
                id=id,
                **configs,
            ) if type == "text" else dmc.NumberInput(
                id=id,
                **configs,
            ),
        ],
    )


def generate_options(options: list | EnumMeta | dict) -> list[dict]:
    """Format options for dropdowns, checklists, radios, etc.

    Args:
        options: A list, EnumMeta, or dictionary of options to format.

    Returns:
        A list of dictionaries with "label" and "value" keys for each option.
    """
    if isinstance(options, EnumMeta):
        return [{"label": option.label, "value": f"{option.value}"} for option in options]

    if isinstance(options, dict):
        return [{"label": f"{key}", "value": f"{value}"} for key, value in options.items()]

    return [{"label": f"{option}", "value": f"{option}"} for option in options]


def generate_settings_form() -> html.Div:
    """Generate settings for selecting the scenario, model, and solver.

    Returns:
        A Div containing the settings for selecting the scenario, model, and solver.
    """
    solver_options = generate_options(SolverType)
    input_dir = Path("input")
    file_options = sorted(
        [
            {"label": str(path.name), "value": str(path)}
            for path in input_dir.glob("*")
            if path.is_file()
        ],
        key=lambda option: option["label"],
    )

    default_input = INPUTS[0] if INPUTS else (file_options[0]["value"] if file_options else "")

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "Scenario",
                "input-file-select",
                file_options or [{"label": default_input, "value": default_input}],
                value=default_input,
            ),
            slider(
                "Runs Per Solver",
                "runs",
                RUNS,
            ),
            checklist(
                "Solvers",
                "solver-selection",
                sorted(solver_options, key=lambda op: op["value"]),
                [option["value"] for option in solver_options],
                inline=False,
            ),
            input(
                "Solver Time Limit (seconds)",
                "solver-time-limit",
                SOLVER_TIME,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Generate run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Optimization", id="run-button", className="button"),
            html.Button(
                "Cancel Optimization",
                id="cancel-button",
                className="button",
                style={"display": "none"},
            ),
        ],
    )


def solver_not_selected_panel(solver_name: str) -> html.Div:
    """Placeholder content for a solver tab when the solver was not selected."""
    return html.Div([html.P(f"{solver_name} was not selected.")])


def solver_solution_panel(has_solution: bool, figure) -> dcc.Graph | html.P:
    """Return a graph of the solver's best solution, or a 'no solution' message."""
    if has_solution:
        return dcc.Graph(figure=figure, config={"displayModeBar": False}, responsive=True)
    return html.H2(
        "No solution found within the given time limit.",
        className="placeholder-text",
    )


def waiting_panel() -> html.Div:
    """Placeholder shown on the Results tab while solvers are still running."""
    return html.Div([html.P("Waiting for solvers to finish...")])


def comparison_panel(figure) -> dcc.Graph | html.H4:
    """Wrap the comparison Plotly figure in a dcc.Graph, or show a fallback message."""
    if figure is None:
        return html.H4(
            "No solutions found to compare.",
            className="placeholder-text",
        )
    return dcc.Graph(figure=figure, config={"displayModeBar": False}, responsive=True)


def comparison_summary_table(
    summary_rows: list[dict],
    min_energy: int | float | None,
    known_optimal: int | float | None,
) -> html.Table:
    """Build the highlighted Comparison Summary table.

    Rows with 0 OK runs are highlighted red; the row with the lowest best
    energy is highlighted teal.
    """
    def fmt_energy(val: object) -> str:
        if val is None:
            return "n/a"
        s = str(val)
        if known_optimal is not None and val == known_optimal:
            return f"{s} (optimal)"
        return s

    def row_class(row: dict) -> str:
        if row["ok_runs"] == 0:
            return "row-highlight-fail"
        if min_energy is not None and row["best_energy"] == min_energy:
            return "row-highlight-best"
        return ""

    headers = ["Formulation", "Runs", "OK Runs", "Best Energy", "Avg Energy"]
    return html.Table(
        className="problem-details-table",
        children=[
            html.Thead(html.Tr([html.Th(h) for h in headers])),
            html.Tbody([
                html.Tr(
                    className=row_class(row),
                    children=[
                        html.Td(str(row["formulation"])),
                        html.Td(str(row["runs"])),
                        html.Td(str(row["ok_runs"])),
                        html.Td(fmt_energy(row["best_energy"])),
                        html.Td(str(row["avg_energy"])),
                    ],
                )
                for row in summary_rows
            ]),
        ],
    )


def results_layout(comparison_element, summary_table: html.Table) -> html.Div:
    """Results tab content: comparison graph on the left, summary table on the right."""
    return html.Div(
        className="results-layout",
        children=[
            html.Div(
                className="results-layout__graph",
                children=comparison_element,
            ),
            html.Div(
                className="results-layout__table",
                children=[
                    html.H3("Comparison Summary"),
                    summary_table,
                ],
            ),
        ],
    )


def create_interface() -> html.Div:
    """Create the main application interface."""
    return html.Div(
        id="app-container",
        children=[
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
                tabIndex=1,
            ),
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="run-in-progress", data=False),  # Indicates whether run is in progress
            dcc.Store(id="running-highs", data=False),
            dcc.Store(id="running-scip", data=False),
            dcc.Store(id="running-stride", data=False),
            dcc.Store(id="highs-store", data={}),
            dcc.Store(id="scip-store", data={}),
            dcc.Store(id="stride-store", data={}),
            # Settings and results columns
            html.Main(
                className="columns-main",
                id="main-content",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dmc.Tabs(
                                id="tabs",
                                value="input-tab",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Nav(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab("Input", value="input-tab"),
                                                            dmc.TabsTab(
                                                                "Results",
                                                                value="results-tab",
                                                                id="results-tab",
                                                                disabled=True,
                                                            ),
                                                            dmc.TabsTab(
                                                                "HiGHS",
                                                                value="highs-tab",
                                                                id="highs-tab",
                                                                disabled=True,
                                                            ),
                                                            dmc.TabsTab(
                                                                "SCIP",
                                                                value="scip-tab",
                                                                id="scip-tab",
                                                                disabled=True,
                                                            ),
                                                            dmc.TabsTab(
                                                                "Stride",
                                                                value="stride-tab",
                                                                id="stride-tab",
                                                                disabled=True,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="input-tab",
                                        tabIndex="12",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    dcc.Loading(
                                                        parent_className="input",
                                                        type="circle",
                                                        color=THEME_COLOR,
                                                        # A Dash callback (in app.py) will generate content in the Div below
                                                        children=html.Div(
                                                            id="input",
                                                            children=dcc.Graph(
                                                                id="input-graph", config={"displayModeBar": False}, responsive=True
                                                            )
                                                        ),
                                                    ),
                                                ]
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="results-tab",
                                        tabIndex="13",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    dcc.Loading(
                                                        parent_className="results",
                                                        type="circle",
                                                        color=THEME_COLOR,
                                                        # A Dash callback will generate content in the Div below
                                                        children=html.Div(id="results"),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="highs-tab",
                                        tabIndex="14",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    dcc.Loading(
                                                        parent_className="results",
                                                        type="circle",
                                                        color=THEME_COLOR,
                                                        children=html.Div(id="highs-results"),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="scip-tab",
                                        tabIndex="15",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    dcc.Loading(
                                                        parent_className="results",
                                                        type="circle",
                                                        color=THEME_COLOR,
                                                        children=html.Div(id="scip-results"),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="stride-tab",
                                        tabIndex="16",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    dcc.Loading(
                                                        parent_className="results",
                                                        type="circle",
                                                        color=THEME_COLOR,
                                                        children=html.Div(id="stride-results"),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
