from dataclasses import dataclass, field, asdict
from typing import List, Optional
import uuid

@dataclass
class Vertex:
    Y: float
    X: float
    Z: Optional[float] = None

@dataclass
class Point:
    Number: Optional[int] = None
    Name: Optional[str] = None
    Tag: Optional[str] = None
    Identifier: str = field(default_factory=lambda: str(uuid.uuid4()))
    Type: str = "Point"
    Fill: str = "R:1,G:0,B:0,A:0"
    Verticies: List[Vertex] = field(default_factory=list)

@dataclass
class CompoundShape:
    Name: Optional[str] = None
    Identifier: str = field(default_factory=lambda: str(uuid.uuid4()))
    Type: str = "CompoundShape"
    Fill: str = "R:1,G:0,B:0,A:0"
    Children: List[Point] = field(default_factory=list)

@dataclass
class Entry:
    Identifier: str = field(default_factory=lambda: str(uuid.uuid4()))
    HolderRow: str = "0"
    HolderColumn: str = "0"
    CarrierIndex: str = "0"
    ChamberRow: str = "0"
    ChamberColumn: str = "0"
    ChamberImageIndex: str = "-1"
    MosaicRows: str = "0"
    MosaicColumns: str = "0"

@dataclass
class StageOverviewRegions:
    Regions: List[CompoundShape] = field(default_factory=list)
    DefinedRegions: List[Entry] = field(default_factory=list)
    StackList: List[dict] = field(default_factory=list)

