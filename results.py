from pydantic import BaseModel
from typing import List, Optional


class Rating(BaseModel):
    Score: Optional[float]
    Legal: Optional[float]


class Clue(BaseModel):
    Word0: str
    Word1: str
    Clue: str
    Rating: Rating = Rating(Score=None, Legal=None)

    def as_tuple(self):
        if self.Word0 < self.Word1:
            return (self.Word0, self.Word1, self.Clue)
        else:
            return (self.Word1, self.Word0, self.Clue)

    @classmethod
    def from_tuple(cls, tuple):
        return cls(Word0=tuple[0], Word1=tuple[1], Clue=tuple[2])


class Configuration(BaseModel):
    method: str
    temperature: float
    trials: List[Clue]


class Results(BaseModel):
    configurations: List[Configuration]


class Evaluations(BaseModel):
    clues: List[Clue]
