"""Deterministic non-adaptive acquisition orders."""

from __future__ import annotations

import numpy as np

from .contracts import Method


def fixed_action_order(method: Method, *, seed: int) -> tuple[int, ...]:
    if method is Method.SERPENTINE:
        return tuple(
            row * 8 + column
            for row in range(8)
            for column in (range(8) if row % 2 == 0 else range(7, -1, -1))
        )
    if method is Method.CENTER_FIRST:
        return tuple(
            sorted(
                range(64),
                key=lambda index: (
                    (index // 8 - 3.5) ** 2 + (index % 8 - 3.5) ** 2,
                    index,
                ),
            )
        )
    if method is Method.GEOMETRY_SPREAD:
        selected = [0]
        remaining = set(range(1, 64))
        while remaining:
            choice = max(
                remaining,
                key=lambda index: (
                    min(
                        (index // 8 - prior // 8) ** 2
                        + (index % 8 - prior % 8) ** 2
                        for prior in selected
                    ),
                    -index,
                ),
            )
            selected.append(choice)
            remaining.remove(choice)
        return tuple(selected)
    if method is Method.RANDOM:
        return tuple(int(value) for value in np.random.default_rng(seed).permutation(64))
    raise ValueError("fixed action order requires a registered fixed method")


__all__ = ["fixed_action_order"]
