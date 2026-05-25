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
import pulp
from dwave.optimization import Model, put, symbols
from dwave.optimization.mathematical import argsort, concatenate
from dwave.system import LeapHybridNLSampler


def create_runtime_use_matrices(
    input_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """Parse the MPS file and return per-job-mode runtime, resource-use matrices, and start bounds.

    Reads the resource-capacity constraints from the MPS problem to derive how
    long each (job, mode) pair runs and how many Mechaniker/Techniker units it
    consumes per time step. Also reads the lower and upper bounds on each job's
    start time from the ``S_<job>`` decision variables.

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
    _, problem = pulp.LpProblem.fromMPS(input_path)

    data = problem.toDict()

    rm = {(j, m, t): [] for j in range(1, 31) for m in range(1, 4) for t in range(213)}
    rt = {(j, m, t): [] for j in range(1, 31) for m in range(1, 4) for t in range(213)}
    rm_use = {(j, m, t): 0 for j in range(1, 31) for m in range(1, 4) for t in range(213)}
    rt_use = {(j, m, t): 0 for j in range(1, 31) for m in range(1, 4) for t in range(213)}
    for const in data["constraints"]:
        name = const["name"]
        if name.startswith("rescap_Mechaniker"):
            const_time = int(name.split("_")[2])
            for coeff in const["coefficients"]:
                coeff_name = coeff["name"]
                if coeff_name.startswith("x"):
                    job = int(coeff_name.split("_")[1])
                    mode = int(coeff_name.split("_")[2])
                    time = int(coeff_name.split("_")[3])
                    rm[(job, mode, const_time)].append(time)
                    value = int(coeff["value"])
                    rm_use[(job, mode, const_time)] = value

        elif name.startswith("rescap_Techniker"):
            const_time = int(name.split("_")[2])
            for coeff in const["coefficients"]:
                coeff_name = coeff["name"]
                if coeff_name.startswith("x"):
                    job = int(coeff_name.split("_")[1])
                    mode = int(coeff_name.split("_")[2])
                    time = int(coeff_name.split("_")[3])
                    rt[(job, mode, const_time)].append(time)
                    value = int(coeff["value"])
                    rt_use[(job, mode, const_time)] = value

    rm = {k: set(v) for k, v in rm.items()}
    rt = {k: set(v) for k, v in rt.items()}

    # runtimes per job and mode and resource type
    rm_times = {(job, mode): 0 for job in range(1, 31) for mode in range(1, 4)}
    rt_times = {(job, mode): 0 for job in range(1, 31) for mode in range(1, 4)}

    for (j, m, t), v in rm.items():
        rm_times[(j, m)] = max(rm_times[(j, m)], len(v))

    for (j, m, t), v in rt.items():
        rt_times[(j, m)] = max(rt_times[(j, m)], len(v))

    # resource use per job, mode, and resource type

    rm_use_total = {(job, mode): 0 for job in range(1, 31) for mode in range(1, 4)}
    rt_use_total = {(job, mode): 0 for job in range(1, 31) for mode in range(1, 4)}

    for (j, m, t), v in rm_use.items():
        rm_use_total[(j, m)] = max(v, rm_use_total[(j, m)])

    for (j, m, t), v in rt_use.items():
        rt_use_total[(j, m)] = max(v, rt_use_total[(j, m)])

    runtimes = {}
    for k, v in rm_times.items():
        runtimes[k] = max(v, rt_times[k])

    runtimes_matrix = np.zeros((30, 3))

    for (j, m), v in runtimes.items():
        runtimes_matrix[j - 1, m - 1] = v

    rm_use_matrix = np.zeros((30, 3))
    rt_use_matrix = np.zeros((30, 3))

    for (j, m), v in rm_use_total.items():
        rm_use_matrix[j - 1, m - 1] = v

    for (j, m), v in rt_use_total.items():
        rt_use_matrix[j - 1, m - 1] = v

    num_jobs = 30
    lower_bounds = [0] * num_jobs
    upper_bounds = [0] * num_jobs
    for var in problem.variables():
        if var.name.startswith("S_"):
            job = int(var.name.split("_")[1])
            lower_bounds[job - 1] = int(var.lowBound) if var.lowBound is not None else 0
            upper_bounds[job - 1] = int(var.upBound)  if var.upBound  is not None else 0

    return runtimes_matrix, rm_use_matrix, rt_use_matrix, lower_bounds, upper_bounds


def create_precedence_pairs(input_path: str) -> list[tuple[int, int]]:
    """Extract the job precedence pairs from the MPS instance.

    Parses the ``prec`` constraints from the MPS file and returns each
    predecessor-successor relationship as a ``(j1, j2)`` tuple meaning job
    ``j1`` must finish before job ``j2`` starts.

    Args:
        input_path: Path to the ``.mps`` instance file.

    Returns:
        A list of ``(j1, j2)`` integer tuples (1-indexed job numbers) for
        every precedence constraint in the model.
    """
    _, problem = pulp.LpProblem.fromMPS(input_path)
    data = problem.toDict()

    precedence_pairs = []

    for const in data["constraints"]:
        if const["name"].startswith("prec"):
            for var in const["coefficients"]:
                if var["name"].startswith("C"):
                    v1 = int(var["name"].split("_")[1])
                elif var["name"].startswith("S"):
                    v2 = int(var["name"].split("_")[1])
            precedence_pairs.append((v1, v2))

    return precedence_pairs


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

    num_jobs = 30
    upper_bounds_modes = [
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    ]

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
    precedence_pairs = create_precedence_pairs(input_path)
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
        solver.sample(model, time_limit=time_limit)

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
