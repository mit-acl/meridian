from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MatchResult:
    """Result of segment matching, supporting multiple hypotheses."""

    association_arrays: List[np.ndarray]
    scores: List[float]
    counts: List[int]

    @property
    def association_array(self) -> np.ndarray:
        return self.association_arrays[0]

    @property
    def score(self) -> float:
        return self.scores[0]

    @property
    def count(self) -> int:
        return self.counts[0]
