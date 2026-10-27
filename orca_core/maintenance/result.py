# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class MaintenanceOutcome(Generic[T]):
    """Terminal result of one blocking maintenance operation.

    Exactly one of ``completed`` and ``cancelled`` is true. ``result`` carries
    the operation's value when it completed.

    ``cleanup_torque_disable_failed_ids`` contains only motor IDs for which
    the transport returned an explicit negative acknowledgement. An exception
    that prevented cleanup from establishing per-motor results is reported
    separately in ``cleanup_error``.
    """

    completed: bool
    cancelled: bool
    cleanup_torque_disable_failed_ids: tuple[int, ...]
    result: T | None = None
    cleanup_error: str | None = None

    def __post_init__(self) -> None:
        if self.completed == self.cancelled:
            raise ValueError("maintenance outcome must be completed or cancelled")
