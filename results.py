from pydantic import BaseModel
from typing import List, Optional, Tuple


class Rating(BaseModel):
    Score: Optional[float]
    Legal: Optional[float]

    def get_adjusted_score(self) -> float:
        if self.Score is None:
            raise ValueError("Score is None")
        if self.Legal is None:
            raise ValueError("Legal is None")
        if self.Legal >= 2:
            return self.Score
        elif self.Legal >= 1:
            # Allow questionably legal clues to score at most 2.0
            return min(self.Score, 2.0)
        else:
            # Penalize illegal clues
            return -1.0


class Clue(BaseModel):
    Word0: str
    Word1: str
    ClueWord: str
    Rating: Rating = Rating(Score=None, Legal=None)

    def as_tuple(self) -> Tuple[str, str, str]:
        if self.Word0 < self.Word1:
            return (self.Word0, self.Word1, self.ClueWord)
        else:
            return (self.Word1, self.Word0, self.ClueWord)

    @classmethod
    def from_tuple(cls, tuple: Tuple[str, str, str]) -> "Clue":
        return cls(Word0=tuple[0], Word1=tuple[1], ClueWord=tuple[2])


class Configuration(BaseModel):
    method: str
    temperature: float
    trials: List[Clue]


class Results(BaseModel):
    configurations: List[Configuration]


class Evaluations(BaseModel):
    clues: List[Clue]
