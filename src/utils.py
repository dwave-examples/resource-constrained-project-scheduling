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
from collections.abc import Iterable


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
