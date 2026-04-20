from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Vertex:
    Y: float
    X: float
    Z: float | None = None

@dataclass
class Point:
    Number: Optional[int] = None
    Name: Optional[str] = None
    Tag: Optional[str] = None
    Identifier: str = ""
    Type: str = "Point"
    Fill: Optional[str] = None
    Verticies: List[Vertex] = field(default_factory=list)

@dataclass
class CompoundShape:
    Name: Optional[str] = None
    Identifier: str = ""
    Type: str = "CompoundShape"
    Fill: Optional[str] = None
    Children: List[Point] = field(default_factory=list)

@dataclass
class StageOverviewRegions:
    Regions: List[CompoundShape] = field(default_factory=list)
    DefinedRegions: List[dict] = field(default_factory=list)
    StackList: List[dict] = field(default_factory=list)