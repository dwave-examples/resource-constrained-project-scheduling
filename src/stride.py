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

from functools import lru_cache
from typing import Any

import dwave.optimization
import numpy as np
from dwave.optimization import Model, put, symbols
from dwave.optimization.mathematical import argsort, concatenate
from dwave.system import LeapHybridNLSampler

from src.utils import parse_mps_structure


def create_runtime_use_matrices(
    input_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """Parse the MPS file and return per-job-mode runtime, resource-use matrices, and start bounds.

    Delegates to :func:`src.utils.parse_mps_structure` and reshapes the result
    into the numpy arrays expected by :func:`create_model`.

    Args:
        input_path: Path to the ``.mps`` instance file.

    Returns:
        A tuple containing:

        - np.ndarray: ``(30, 3)`` array of job-mode runtimes.
        - np.ndarray: ``(30, 3)`` array of Mechaniker consumption per job-mode.
        - np.ndarray: ``(30, 3)`` array of Techniker consumption per job-mode.
        - list[int]: Per-job lower bounds on start times (length 30, 0-indexed by job-1).
        - list[int]: Per-job upper bounds on start times (length 30, 0-indexed by job-1).
    """
    profile = parse_mps_structure(input_path)
    num_jobs = len(profile["jobs"])
    num_modes = 3

    runtimes_matrix = np.zeros((num_jobs, num_modes))
    rm_use_matrix = np.zeros((num_jobs, num_modes))
    rt_use_matrix = np.zeros((num_jobs, num_modes))

    for (job, mode), duration in profile["durations"].items():
        runtimes_matrix[job - 1, mode - 1] = duration
    for (job, mode), use in profile["mechanic_use"].items():
        rm_use_matrix[job - 1, mode - 1] = use
    for (job, mode), use in profile["technician_use"].items():
        rt_use_matrix[job - 1, mode - 1] = use

    lower_bounds = [profile["lower_bounds"].get(j, 0) for j in range(1, num_jobs + 1)]
    upper_bounds = [profile["upper_bounds"].get(j, 0) for j in range(1, num_jobs + 1)]

    return runtimes_matrix, rm_use_matrix, rt_use_matrix, lower_bounds, upper_bounds


def create_model(
    lower_bounds: list[int],
    upper_bounds: list[int],
    runtimes_matrix: np.ndarray,
    rm_use_matrix: np.ndarray,
    rt_use_matrix: np.ndarray,
    precedence_pairs: list[tuple[int, int]],
) -> tuple[Model, np.ndarray, np.ndarray]:
    """Build the D-Wave nonlinear optimization model for the RCPSP instance.

    Constructs a ``dwave.optimization.Model`` using integer decision variables
    for job start times and modes. Resource feasibility is enforced via an
    accumulate-zip sweep over sorted start/end events. The objective minimizes
    ``100 * R_Mechaniker + 51 * R_Techniker``.

    Args:
        lower_bounds: Per-job lower bounds on start times (length 30).
        upper_bounds: Per-job upper bounds on start times (length 30).
        runtimes_matrix: ``(30, 3)`` array of job-mode durations.
        rm_use_matrix: ``(30, 3)`` array of Mechaniker consumption per job-mode.
        rt_use_matrix: ``(30, 3)`` array of Techniker consumption per job-mode.
        precedence_pairs: List of ``(j1, j2)`` tuples encoding precedence constraints.

    Returns:
        A tuple containing:

        - Model: The constructed D-Wave optimization model.
        - np.ndarray: The model's start time variables as a (30,) array.
        - np.ndarray: The model's mode variables as a (30,) array.
    """
    # accumulate zip formulation
    model = Model()

    num_jobs = runtimes_matrix.shape[0]
    upper_bounds_modes = ((runtimes_matrix > 0).sum(axis=1) - 1).astype(int).tolist()

    starts = model.integer(num_jobs, lower_bound=lower_bounds, upper_bound=upper_bounds)

    modes = model.integer(num_jobs, lower_bound=0, upper_bound=upper_bounds_modes)

    runtimes_mat_mod = runtimes_matrix - 1

    runtimes_matrix_c = model.constant(runtimes_matrix)
    runtimes_mat_mod_c = model.constant(runtimes_mat_mod)
    rm_use_matrix_c = model.constant(rm_use_matrix)
    rt_use_matrix_c = model.constant(rt_use_matrix)

    for j1, j2 in precedence_pairs:
        model.add_constraint(
            starts[model.constant(j1 - 1)]
            + runtimes_matrix_c[model.constant(j1 - 1), modes[model.constant(j1)]]
            - starts[model.constant(j2 - 1)]
            <= 0
        )

    # upper bounds from lp file
    r_mechaniker = model.integer(1, lower_bound=0, upper_bound=40)
    r_techniker = model.integer(1, lower_bound=0, upper_bound=30)

    # get runtime for each job (depends on which mode it is in)
    runtimes = model.constant(np.zeros(num_jobs))
    for i in range(num_jobs):
        runtimes = put(
            runtimes, model.constant(i).reshape((1)), runtimes_mat_mod_c[i, modes[i]].reshape((1))
        )

    end_times = starts + runtimes

    # concatenate the consumption with the negative consumption for start and end times

    # get total consumption (mechaniker and techniker)
    #  note that each job uses only one resource and not the other, so we can just sum over all jobs

    m_consumption_pos = model.constant(np.zeros(num_jobs))
    t_consumption_pos = model.constant(np.zeros(num_jobs))

    for i in range(num_jobs):
        m_consumption_pos = put(
            m_consumption_pos,
            indices=model.constant(i).reshape((1)),
            values=rm_use_matrix_c[i, modes[i]].reshape((1)),
        )
        t_consumption_pos = put(
            t_consumption_pos,
            indices=model.constant(i).reshape((1)),
            values=rt_use_matrix_c[i, modes[i]].reshape((1)),
        )

    m_consumption = concatenate((m_consumption_pos, -1 * m_consumption_pos))
    t_consumption = concatenate((t_consumption_pos, -1 * t_consumption_pos))

    events = concatenate((starts, end_times))
    order = argsort(events)
    order_events_consumption_m = m_consumption[order]
    order_events_consumption_t = t_consumption[order]
    # track the cumulative consumption, adding consumption when a job starts
    #  and subtracting when it ends

    @dwave.optimization.expression
    def add(x, y):
        return x + y

    cumulative_consumption_m = symbols.AccumulateZip(
        add, (order_events_consumption_m,), initial=model.constant(0)
    )
    model.add_constraint((cumulative_consumption_m <= r_mechaniker).all())

    cumulative_consumption_t = symbols.AccumulateZip(
        add, (order_events_consumption_t,), initial=model.constant(0)
    )
    model.add_constraint((cumulative_consumption_t <= r_techniker).all())

    model.objective = 100 * r_mechaniker + 51 * r_techniker

    return model, starts, modes


@lru_cache(maxsize=4)
def _preprocessed_data(
    input_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[tuple[int, int]]]:
    """Return cached runtime matrices, start bounds, and precedence pairs for the given instance.

    Results are memoised so repeated calls with the same path avoid re-parsing
    the MPS file.

    Args:
        input_path: Path to the ``.mps`` instance file.

    Returns:
        A tuple containing:

        - np.ndarray: ``(30, 3)`` array of job-mode runtimes.
        - np.ndarray: ``(30, 3)`` array of Mechaniker consumption per job-mode.
        - np.ndarray: ``(30, 3)`` array of Techniker consumption per job-mode.
        - list[int]: Per-job lower bounds on start times.
        - list[int]: Per-job upper bounds on start times.
        - list[tuple[int, int]]: List of ``(j1, j2)`` precedence pairs.
    """
    runtimes_matrix, rm_use_matrix, rt_use_matrix, lower_bounds, upper_bounds = (
        create_runtime_use_matrices(input_path)
    )
    precedence_pairs = parse_mps_structure(input_path)["edges"]
    return runtimes_matrix, rm_use_matrix, rt_use_matrix, lower_bounds, upper_bounds, precedence_pairs


def solve_stride(time_limit: float, input_path: str) -> dict[str, Any]:
    """Run the Stride nonlinear formulation once and return comparable result metadata.

    Builds the D-Wave nonlinear model, submits it to ``LeapHybridNLSampler``
    with the given time limit, and extracts the resulting schedule.

    Args:
        time_limit: Maximum solver wall-clock time in seconds.
        input_path: Path to the ``.mps`` instance file.

    Returns:
        A dict with keys:

        - ``solver`` (str): ``"Stride"``.
        - ``status`` (str): ``"Completed"`` on success, or an error/unavailability message.
        - ``energy`` (float | None): Objective value, or ``None`` on failure.
        - ``ok`` (bool): ``True`` if all constraints are satisfied.
        - ``starts`` (dict[int, int]): Mapping of 1-indexed job number to start time
          (absent on failure).
        - ``modes`` (dict[int, int]): Mapping of 1-indexed job number to selected mode
          (1-indexed, absent on failure).
    """
    if LeapHybridNLSampler is None:
        return {
            "solver": "Stride",
            "status": "Unavailable: dwave optimization packages not installed",
            "energy": None,
            "ok": False,
        }

    try:
        runtimes_matrix, rm_use_matrix, rt_use_matrix, lower_bounds, upper_bounds, precedence_pairs = (
            _preprocessed_data(input_path)
        )

        model, starts, modes = create_model(
            lower_bounds,
            upper_bounds,
            runtimes_matrix,
            rm_use_matrix,
            rt_use_matrix,
            precedence_pairs,
        )
        model.lock()

        solver = LeapHybridNLSampler()
        solver.sample(model, time_limit=time_limit, label="Example - Resource-Constrained Project Scheduling")

        starts_values = [int(value) for value in starts.state().tolist()]
        modes_values = [int(value) for value in modes.state().tolist()]

        return {
            "solver": "Stride",
            "status": "Completed",
            "energy": float(np.asarray(model.objective.state()).flat[0]),
            "ok": all(sym.state() for sym in model.iter_constraints()),
            "starts": {index + 1: value for index, value in enumerate(starts_values)},
            "modes": {index + 1: value + 1 for index, value in enumerate(modes_values)},
        }
    except Exception as exc:  # pragma: no cover - runtime/system dependent
        return {
            "solver": "Stride",
            "status": f"Error: {exc}",
            "energy": None,
            "ok": False,
            "starts": {},
            "modes": {},
        }
