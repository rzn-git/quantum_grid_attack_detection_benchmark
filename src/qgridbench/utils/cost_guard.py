"""Hard cost gate for any paid quantum-hardware submission.

HARD RULE (CLAUDE.md section 6): no paid-hardware job is submitted without explicit
human approval of the dollar estimate. Approval is expressed by the operator
setting the environment variable QGB_APPROVED_BUDGET_USD to a value >= the
estimate. Anything else raises before any job is created.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Published AWS Braket on-demand prices (USD), as of 2026-08 (verify before use —
# prices change; per-task fee is uniform across QPUs).
PER_TASK_USD = 0.30
PER_SHOT_USD = {
    "ionq_aria": 0.03,
    "ionq_forte": 0.08,
    "iqm_garnet": 0.00145,
    "rigetti_ankaa": 0.0009,
}


@dataclass
class CostEstimate:
    device: str
    n_tasks: int
    shots_per_task: int

    @property
    def usd(self) -> float:
        if self.device not in PER_SHOT_USD:
            raise ValueError(f"unknown device '{self.device}'; known: {list(PER_SHOT_USD)}")
        return self.n_tasks * (PER_TASK_USD + self.shots_per_task * PER_SHOT_USD[self.device])

    def summary(self) -> str:
        return (
            f"device={self.device} tasks={self.n_tasks} shots/task={self.shots_per_task} "
            f"-> estimated ${self.usd:,.2f}"
        )


def require_approval(estimate: CostEstimate) -> None:
    """Raise unless the operator has explicitly approved a budget >= estimate."""
    approved = os.environ.get("QGB_APPROVED_BUDGET_USD")
    if approved is None:
        raise PermissionError(
            "Paid-hardware submission blocked: no human-approved budget.\n"
            f"Estimate: {estimate.summary()}\n"
            "To approve, set QGB_APPROVED_BUDGET_USD to at least the estimate."
        )
    if float(approved) < estimate.usd:
        raise PermissionError(
            f"Paid-hardware submission blocked: approved budget ${float(approved):,.2f} "
            f"< estimate ${estimate.usd:,.2f}. ({estimate.summary()})"
        )
