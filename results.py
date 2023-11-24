from pydantic import BaseModel
from typing import List, Optional


class Clue(BaseModel):
    Word0: str
    Word1: str
    Clue: str
    Score: Optional[float] = None

    def as_tuple(self):
        return (self.Word0, self.Word1, self.Clue)

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
