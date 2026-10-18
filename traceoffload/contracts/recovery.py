"""Coordinator-owned workflow recovery contracts.

These objects carry the facts created when an SLM post-commit failure cannot
be rolled back.  They deliberately live outside a normal direct RoutePlan:
ordinary direct spans have no verifier plan, while a rollback recovery still
needs a separate, synchronous recovery check owned by the Coordinator.
"""

from __future__ import annotations

from pydantic import AwareDatetime, Field, model_validator

from .base import ContractModel, Identifier, ResumeStatus
from .execution import StateOutcome, ToolExecutionResult
from .routing import ModelTarget


class RecoveryDirective(ContractModel):
    """Facts injected into the next Agent-visible recovery model request."""

    recovery_id: Identifier = Field(min_length=1, max_length=128)
    source_episode_id: Identifier = Field(min_length=1, max_length=128)
    source_logical_invocation_id: Identifier = Field(min_length=1, max_length=128)
    source_route_plan_id: Identifier = Field(min_length=1, max_length=128)
    target: ModelTarget
    failed_tool_results: tuple[ToolExecutionResult, ...] = Field(min_length=1)
    state_outcome: StateOutcome
    recovery_attempt_count: int = Field(default=0, ge=0)
    recovery_attempt_limit: int = Field(default=3, ge=1)
    deadline_at: AwareDatetime
    required_recovery_check: bool = True


class RecoveryCheck(ContractModel):
    """One post-recovery state check, outside the normal DIRECT verifier plan."""

    recovery_id: Identifier = Field(min_length=1, max_length=128)
    attempt_index: int = Field(ge=1)
    checked_transaction_ids: tuple[Identifier, ...] = ()
    status: ResumeStatus
    passed: bool
    reason: Identifier | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def status_matches_passed(self) -> RecoveryCheck:
        if self.passed != (self.status == ResumeStatus.SUCCEEDED):
            raise ValueError("recovery check passes only when resume_status is succeeded")
        return self


__all__ = ["RecoveryDirective", "RecoveryCheck"]
