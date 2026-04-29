"""Expected failures during stochastic scene sampling.

These exceptions are intentionally narrow: they only represent the normal case
where a requested random scene/layout could not be sampled within the configured
attempt budget. Other failures should remain ordinary bugs.
"""

from __future__ import annotations


class SceneSamplingError(RuntimeError):
    """Base class for expected stochastic scene sampling failures."""


class ObstacleSamplingError(SceneSamplingError):
    """Raised when collision-free obstacle placement cannot be sampled."""


class ObjectSamplingError(SceneSamplingError):
    """Raised when collision-free object placement cannot be sampled."""
