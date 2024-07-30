from dataclasses import dataclass

@dataclass
class Record:
    id: str
    metric: float
    kps: list[list[int]]