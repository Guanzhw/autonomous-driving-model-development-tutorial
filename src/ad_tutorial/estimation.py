"""Small, causal state-estimation examples used by the second driving unit.

The estimators intentionally operate on :class:`DrivingObservation`, the
typed boundary already used by the controller.  They do not access the
simulator or future samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import cos, pi, sin
from typing import Mapping, Sequence

import numpy as np

from .driving import DrivingObservation


def _point(value: Sequence[float]) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("a point must contain two finite values")
    return point


def _angle(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("heading must be finite")
    return float((value + pi) % (2 * pi) - pi)


def world_to_ego(
    point_world: Sequence[float], ego_position: Sequence[float], heading_rad: float
) -> tuple[float, float]:
    """Transform a world point into the ego frame (x forward, y left)."""
    delta = _point(point_world) - _point(ego_position)
    heading = _angle(float(heading_rad))
    c, s = cos(heading), sin(heading)
    local = np.array([[c, s], [-s, c]]) @ delta
    return float(local[0]), float(local[1])


def ego_to_world(
    point_ego: Sequence[float], ego_position: Sequence[float], heading_rad: float
) -> tuple[float, float]:
    """Transform an ego-frame point back into world coordinates."""
    local = _point(point_ego)
    origin = _point(ego_position)
    heading = _angle(float(heading_rad))
    c, s = cos(heading), sin(heading)
    world = np.array([[c, -s], [s, c]]) @ local + origin
    return float(world[0]), float(world[1])


@dataclass(frozen=True)
class MeasurementConfig:
    """Noise and bias used to make the sensor boundary explicit."""

    seed: int = 19
    position_noise_std_m: tuple[float, float] = (0.18, 0.12)
    heading_noise_std_rad: float = 0.018
    speed_noise_std_mps: float = 0.20
    position_bias_m: tuple[float, float] = (0.10, -0.06)
    heading_bias_rad: float = 0.008
    speed_bias_mps: float = 0.05
    position_process_variance_m2_per_s: float = 0.01
    heading_process_variance_rad2_per_s: float = 0.0002
    speed_process_variance_mps2_per_s: float = 0.04

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if len(self.position_noise_std_m) != 2 or len(self.position_bias_m) != 2:
            raise ValueError("position values must contain two entries")
        values = (
            *self.position_noise_std_m,
            self.heading_noise_std_rad,
            self.speed_noise_std_mps,
            *self.position_bias_m,
            self.heading_bias_rad,
            self.speed_bias_mps,
            self.position_process_variance_m2_per_s,
            self.heading_process_variance_rad2_per_s,
            self.speed_process_variance_mps2_per_s,
        )
        if not all(np.isfinite(v) for v in values):
            raise ValueError("measurement parameters must be finite")
        if any(
            v < 0
            for v in (
                *self.position_noise_std_m,
                self.heading_noise_std_rad,
                self.speed_noise_std_mps,
            )
        ):
            raise ValueError("noise standard deviations must be nonnegative")
        if any(
            v < 0
            for v in (
                self.position_process_variance_m2_per_s,
                self.heading_process_variance_rad2_per_s,
                self.speed_process_variance_mps2_per_s,
            )
        ):
            raise ValueError("process variances must be nonnegative")

    def manifest(self) -> dict:
        return asdict(self)


class ScalarKalman:
    """One-dimensional random-walk Kalman filter.

    The state is assumed locally constant over one decision interval.  The
    process variance is supplied per second and permits slow vehicle motion;
    the measurement variance represents the declared sensor noise.
    """

    def __init__(
        self,
        measurement_variance: float,
        process_variance: float = 0.01,
        initial_variance: float = 1.0,
    ):
        if measurement_variance < 0 or process_variance < 0 or initial_variance <= 0:
            raise ValueError("invalid Kalman variances")
        self.measurement_variance = float(measurement_variance)
        self.process_variance = float(process_variance)
        self.initial_variance = float(initial_variance)
        self.reset()

    def reset(self) -> None:
        self.value: float | None = None
        self.variance = self.initial_variance

    def update(self, measurement: float, dt: float = 1.0) -> float:
        if not np.isfinite(measurement) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("measurement must be finite and dt positive")
        if self.value is None:
            self.value = float(measurement)
            return self.value
        self.variance += self.process_variance * float(dt)
        if self.measurement_variance == 0:
            self.value = float(measurement)
            self.variance = 0.0
            return self.value
        gain = self.variance / (self.variance + self.measurement_variance)
        self.value += gain * (float(measurement) - self.value)
        self.variance = (1.0 - gain) * self.variance
        return self.value


class NoisyObserver:
    """Synthetic measurement observer, optionally followed by causal filters."""

    _NOISY_FIELDS = ("heading_rad", "speed_mps")

    def __init__(
        self,
        config: MeasurementConfig | Mapping | None = None,
        *,
        filtered: bool = False,
    ):
        if config is None:
            config = MeasurementConfig()
        elif isinstance(config, Mapping):
            config = MeasurementConfig(**config)
        if not isinstance(config, MeasurementConfig):
            raise TypeError("config must be MeasurementConfig or a mapping")
        self.config = config
        self.filtered = bool(filtered)
        self.reset()

    def reset(self, lane=None) -> None:
        self._rng = np.random.default_rng(self.config.seed)
        self._lane = lane
        self.last_measurement: DrivingObservation | None = None
        self._filters: dict[str, ScalarKalman] = {}
        if self.filtered:
            std = {
                "position_x": self.config.position_noise_std_m[0],
                "position_y": self.config.position_noise_std_m[1],
                "heading_rad": self.config.heading_noise_std_rad,
                "speed_mps": self.config.speed_noise_std_mps,
            }
            process = {
                "position_x": self.config.position_process_variance_m2_per_s,
                "position_y": self.config.position_process_variance_m2_per_s,
                "heading_rad": self.config.heading_process_variance_rad2_per_s,
                "speed_mps": self.config.speed_process_variance_mps2_per_s,
            }
            for name, value in std.items():
                self._filters[name] = ScalarKalman(
                    value * value,
                    process[name],
                    value * value if value else 1.0,
                )

    def _measurement(self, truth: DrivingObservation) -> DrivingObservation:
        c = self.config
        position = np.asarray(truth.position, dtype=float)
        position += np.asarray(c.position_bias_m)
        position += self._rng.normal(0.0, c.position_noise_std_m, size=2)
        values = {
            "heading_rad": truth.heading_rad
            + c.heading_bias_rad
            + self._rng.normal(0.0, c.heading_noise_std_rad),
            "speed_mps": truth.speed_mps
            + c.speed_bias_mps
            + self._rng.normal(0.0, c.speed_noise_std_mps),
        }
        measured = replace(truth, position=tuple(map(float, position)), **values)
        if self._lane is not None:
            s, lateral = self._lane.local_coordinates(measured.position)
            measured = replace(
                measured,
                lane_longitudinal_m=float(s),
                lane_lateral_m=float(lateral),
                lane_heading_rad=float(self._lane.heading_theta_at(s)),
                lane_width_m=float(self._lane.width_at(s)),
            )
        return measured

    def observe(
        self, truth: DrivingObservation, step: int, dt: float
    ) -> DrivingObservation:
        if not isinstance(truth, DrivingObservation):
            raise TypeError("truth must be DrivingObservation")
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        measured = self._measurement(truth)
        self.last_measurement = measured
        if not self.filtered:
            return replace(
                measured,
                heading_rad=_angle(measured.heading_rad),
                lane_heading_rad=_angle(measured.lane_heading_rad),
            )
        heading_measurement = measured.heading_rad
        previous_heading = self._filters["heading_rad"].value
        if previous_heading is not None:
            heading_measurement = previous_heading + _angle(
                heading_measurement - previous_heading
            )
        estimates = {
            "heading_rad": self._filters["heading_rad"].update(heading_measurement, dt),
            "speed_mps": self._filters["speed_mps"].update(measured.speed_mps, dt),
        }
        estimates["position"] = tuple(
            self._filters[name].update(measured.position[index], dt)
            for index, name in enumerate(("position_x", "position_y"))
        )
        estimates["heading_rad"] = _angle(estimates["heading_rad"])
        if self._lane is not None:
            s, lateral = self._lane.local_coordinates(estimates["position"])
            estimates["lane_longitudinal_m"] = float(s)
            estimates["lane_lateral_m"] = float(lateral)
            estimates["lane_heading_rad"] = float(self._lane.heading_theta_at(s))
            estimates["lane_width_m"] = float(self._lane.width_at(s))
        return replace(measured, **estimates)

    def manifest(self) -> dict:
        return {
            "method": "causal_scalar_kalman"
            if self.filtered
            else "synthetic_noisy_measurement",
            "config": self.config.manifest(),
            "filter": "one_dimensional_random_walk" if self.filtered else None,
            "process_variance_time_unit": "per_second",
        }


class RawMeasurementObserver(NoisyObserver):
    def __init__(self, config: MeasurementConfig | Mapping | None = None):
        super().__init__(config, filtered=False)


class KalmanObserver(NoisyObserver):
    def __init__(self, config: MeasurementConfig | Mapping | None = None):
        super().__init__(config, filtered=True)


def make_observer(
    mode: str, config: MeasurementConfig | Mapping | None = None
) -> NoisyObserver | None:
    """Construct the three lesson conditions: oracle, raw measurement, filter."""
    if mode == "oracle":
        return None
    if mode == "raw":
        return RawMeasurementObserver(config)
    if mode == "filter":
        return KalmanObserver(config)
    raise ValueError("mode must be oracle, raw, or filter")


__all__ = [
    "KalmanObserver",
    "MeasurementConfig",
    "NoisyObserver",
    "RawMeasurementObserver",
    "ScalarKalman",
    "SE2",
    "ego_to_world",
    "make_observer",
    "world_to_ego",
]


@dataclass(frozen=True)
class SE2:
    """Pose helper exposing the same world/ego convention as the functions."""

    position: tuple[float, float]
    heading_rad: float

    def to_ego(self, point_world: Sequence[float]) -> tuple[float, float]:
        return world_to_ego(point_world, self.position, self.heading_rad)

    def to_world(self, point_ego: Sequence[float]) -> tuple[float, float]:
        return ego_to_world(point_ego, self.position, self.heading_rad)
