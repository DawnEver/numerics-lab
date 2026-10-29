"""Portable accepted event-locator clocks; no event callback or physical law is serialized."""

import math
from dataclasses import asdict, dataclass, fields

__all__ = ['EventCheckpointState']


@dataclass(frozen=True)
class EventCheckpointState:
    step_dt: float
    bracket_tolerance: float
    adaptive: bool
    minimum_step: float
    refine: float | None
    nominal_index: int
    nominal_anchor: float
    pending_target: float | None
    pending_events: int
    last_fired: tuple[tuple[int, float], ...]
    surface_directions: tuple[int, ...]
    line_search: str = 'backtrack'

    def __post_init__(self) -> None:
        if self.nominal_anchor is None:
            msg = 'event checkpoint requires its actual nominal anchor'
            raise ValueError(msg)
        for value in (self.step_dt, self.bracket_tolerance, self.minimum_step):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                msg = 'event checkpoint requires finite positive step policies'
                raise ValueError(msg)
        for value in (self.refine, self.nominal_anchor, self.pending_target):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                msg = 'event checkpoint clocks must be finite numbers'
                raise ValueError(msg)
        if self.refine is not None and self.refine <= 0:
            msg = 'event checkpoint refinement must be positive'
            raise ValueError(msg)
        if type(self.adaptive) is not bool:
            msg = 'event checkpoint adaptive policy must be a boolean'
            raise ValueError(msg)
        if self.line_search not in ('backtrack', 'directional'):
            msg = 'event checkpoint requires its actual supported line-search policy'
            raise ValueError(msg)
        if any(type(value) is not int or value < 0 for value in (self.nominal_index, self.pending_events)):
            msg = 'event checkpoint cell counters must be nonnegative integers'
            raise ValueError(msg)
        if self.pending_target is None and self.pending_events:
            msg = 'event checkpoint event budget requires an unfinished cell'
            raise ValueError(msg)
        if type(self.surface_directions) is not tuple or any(
            type(value) is not int or value not in (-1, 0, 1) for value in self.surface_directions
        ):
            msg = 'event checkpoint requires explicit surface directions'
            raise ValueError(msg)
        previous = -1
        if type(self.last_fired) is not tuple:
            msg = 'event checkpoint firing history must be immutable'
            raise ValueError(msg)
        for record in self.last_fired:
            if type(record) is not tuple or len(record) != 2:
                msg = 'event checkpoint firing records require index and time'
                raise ValueError(msg)
            index, time = record
            if type(index) is not int or not previous < index < len(self.surface_directions):
                msg = 'event checkpoint firing indices must be unique, ordered and bound'
                raise ValueError(msg)
            if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time):
                msg = 'event checkpoint firing times must be finite'
                raise ValueError(msg)
            previous = index

    def validate_clock(self, time, origin):
        if origin is None:
            msg = 'event checkpoint requires its original nominal grid origin'
            raise ValueError(msg)
        if self.pending_target is not None and self.pending_target <= time:
            msg = 'event checkpoint pending clocks must follow the accepted time'
            raise ValueError(msg)
        planned = self.nominal_anchor == self.pending_target and self.nominal_index == 0
        if not origin <= self.nominal_anchor or (self.nominal_anchor > time and not planned):
            msg = 'event checkpoint nominal anchor requires accepted history or its explicit pending landing'
            raise ValueError(msg)
        if any(not origin <= fired <= time for _index, fired in self.last_fired):
            msg = 'event checkpoint firing times must lie in its accepted history'
            raise ValueError(msg)

    def to_mapping(self):
        data = asdict(self)
        data['last_fired'] = [list(record) for record in self.last_fired]
        data['surface_directions'] = list(self.surface_directions)
        return data

    @classmethod
    def from_mapping(cls, data):
        if not isinstance(data, dict) or set(data) != {item.name for item in fields(cls)}:
            msg = 'event checkpoint requires exactly its declared clock fields'
            raise ValueError(msg)
        if (
            not isinstance(data['last_fired'], list)
            or any(not isinstance(record, list) for record in data['last_fired'])
            or not isinstance(data['surface_directions'], list)
        ):
            msg = 'event checkpoint requires JSON firing and direction arrays'
            raise ValueError(msg)
        return cls(**{
            **data,
            'last_fired': tuple(tuple(record) for record in data['last_fired']),
            'surface_directions': tuple(data['surface_directions']),
        })
