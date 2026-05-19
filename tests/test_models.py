"""Tests for `omytea.models` — basic geometric / observation primitives."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from omytea.models import Observation, Position, RectRegion


# --------------------------------------------------------------
# Position
# --------------------------------------------------------------


def test_position_construction_2d() -> None:
    p = Position(x=0.3, y=0.5)
    assert p.x == 0.3
    assert p.y == 0.5
    assert p.z is None
    assert p.space == "image"


def test_position_with_z_and_space() -> None:
    p = Position(x=1.0, y=2.0, z=3.0, space="world_meters")
    assert p.z == 3.0
    assert p.space == "world_meters"


def test_position_is_hashable_and_comparable() -> None:
    a = Position(x=0.5, y=0.5)
    b = Position(x=0.5, y=0.5)
    c = Position(x=0.5, y=0.6)
    assert a == b
    assert a != c
    # If dataclass is frozen, it's hashable.
    {a, b, c}  # should not raise


# --------------------------------------------------------------
# RectRegion
# --------------------------------------------------------------


def test_rect_region_construction() -> None:
    r = RectRegion(min_x=0.0, min_y=0.0, max_x=1.0, max_y=1.0)
    assert r.min_x == 0.0
    assert r.max_x == 1.0
    assert r.min_y == 0.0
    assert r.max_y == 1.0
    assert r.space == "image"


def test_rect_region_equality() -> None:
    a = RectRegion(min_x=0.0, min_y=0.0, max_x=1.0, max_y=1.0)
    b = RectRegion(min_x=0.0, min_y=0.0, max_x=1.0, max_y=1.0)
    c = RectRegion(min_x=0.0, min_y=0.0, max_x=2.0, max_y=1.0)
    assert a == b
    assert a != c


# --------------------------------------------------------------
# Observation
# --------------------------------------------------------------


def _obs(label: str = "person", confidence: float = 0.85) -> Observation:
    return Observation(
        stream_id="test",
        timestamp=datetime(2026, 5, 19, tzinfo=timezone.utc),
        object_id="obs_1",
        label=label,
        position=Position(x=0.5, y=0.5),
        confidence=confidence,
    )


def test_observation_required_fields() -> None:
    obs = _obs()
    assert obs.label == "person"
    assert obs.confidence == 0.85
    assert obs.object_id == "obs_1"
    assert obs.position.x == 0.5


def test_observation_optional_bbox_default_none() -> None:
    assert _obs().bbox is None


def test_observation_attributes_default_empty_dict() -> None:
    obs = _obs()
    assert obs.attributes == {}


def test_observation_confidence_can_be_zero() -> None:
    obs = _obs(confidence=0.0)
    assert obs.confidence == 0.0
