from typing import Protocol

import numpy.typing as npt


class Observation(Protocol):
    board: npt.ArrayLike


class Configuration(Protocol):
    rows: int
    columns: int
