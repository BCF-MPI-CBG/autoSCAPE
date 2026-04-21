from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np

@dataclass
class Vertex:
    X: float
    Y: float

@dataclass
class ExtendedProperty:
    pass

@dataclass
class Point:
	Name: str
	Identifier: str
	Type: str
	Fill: np.ndarray | list[int] | str
	Tag: Optional[str] = None
	Number: Optional[int] = None
	Verticies: List[Vertex] = field(default_factory=list)
	ExtendedProperties: List[ExtendedProperty] = field(default_factory=list)

	def __post_init__(self):
		# Convert Fill to a string representation if it's a numpy array
		if isinstance(self.Fill, np.ndarray):
			self.Fill = f"R:{self.Fill[0]} ,G:{self.Fill[1]} ,B:{self.Fill[2]} ,A:{self.Fill[3]}"
		elif isinstance(self.Fill, list):
			self.Fill = f"R:{self.Fill[0]} ,G:{self.Fill[1]} ,B:{self.Fill[2]} ,A:{self.Fill[3]}"

@dataclass
class CompoundShape:
	Name: str
	Identifier: str
	Type: str
	Fill: np.ndarray | list[int] | str
	Verticies: List[Vertex] = field(default_factory=list)
	ExtendedProperties: List[ExtendedProperty] = field(default_factory=list)
	Children: List[object] = field(default_factory=list)  # Can be Point or CompoundShape

	def __post_init__(self):
		# Convert Fill to a string representation if it's a numpy array
		if isinstance(self.Fill, np.ndarray):
			self.Fill = f"R:{self.Fill[0]} ,G:{self.Fill[1]} ,B:{self.Fill[2]} ,A:{self.Fill[3]}"
		elif isinstance(self.Fill, list):
			self.Fill = f"R:{self.Fill[0]} ,G:{self.Fill[1]} ,B:{self.Fill[2]} ,A:{self.Fill[3]}"

@dataclass
class ShapeList:
    Items: List[CompoundShape] = field(default_factory=list)
    FillMaskMode: str = "None"
    VertexUnitMode: str = "Pixels"

@dataclass
class DefinedRegionEntry:
	Identifier: str
	HolderRow: int | str
	HolderColumn: int | str
	CarrierIndex: int | str
	ChamberRow: int | str
	ChamberColumn: int | str
	ChamberImageIndex: int | str
	MosaicRows: int | str
	MosaicColumns: int | str

	def __post_init__(self):
		# Convert string fields to integers if they are numeric
		for field_name in self.__dataclass_fields__:
			value = getattr(self, field_name)
			if isinstance(value, int):
				setattr(self, field_name, str(value))

@dataclass
class DefinedRegions:
    Entries: List[DefinedRegionEntry] = field(default_factory=list)

@dataclass
class StackEntry:
	Identifier: str
	Begin: float | str
	End: float | str
	SectionCount: int | str
	ReferenceX: float | str
	ReferenceY: float | str
	FocusStabilizerOffset: float | str
	FocusStabilizerOffsetFixed: bool | str
	StackValid: bool | str
	Marked: bool | str

	def __post_init__(self):
		# Convert string fields to appropriate types if they are numeric or boolean
		for field_name in self.__dataclass_fields__:
			value = getattr(self, field_name)
			if isinstance(value, (int, float, bool)):
				setattr(self, field_name, str(value))

@dataclass
class StackList:
    Entries: List[StackEntry] = field(default_factory=list)

@dataclass
class Regions:
    ShapeList: ShapeList

@dataclass
class StageOverviewRegions:
    Regions: Regions
    DefinedRegions: DefinedRegions
    StackList: StackList
		

import xml.etree.ElementTree as ET
from typing import List, Any

def dataclass_to_xml(element: ET.Element, obj: Any, item_index: int = None):
    if item_index is not None:
        element.tag = f"Item{item_index}"

    for field_name, field_value in obj.__dict__.items():
        # Skip private fields (e.g., _formatted)
        if field_name.startswith('_'):
            continue

        # Handle the Children field specifically for CompoundShape
        if field_name == "Children" and isinstance(obj, CompoundShape):
            children_element = ET.SubElement(element, "Children")
            items_element = ET.SubElement(children_element, "Items")
            for i, item in enumerate(field_value):
                item_element = ET.SubElement(items_element, f"Item{i}")
                if hasattr(item, '__dataclass_fields__'):
                    dataclass_to_xml(item_element, item)
                else:
                    item_element.text = str(item)

        # Handle other lists (e.g., Verticies, ExtendedProperties)
        elif isinstance(field_value, list):
            items_element = ET.SubElement(element, f"{field_name}")
            inner_items_element = ET.SubElement(items_element, "Items")
            for i, item in enumerate(field_value):
                item_element = ET.SubElement(inner_items_element, f"Item{i}")
                if hasattr(item, '__dataclass_fields__'):
                    dataclass_to_xml(item_element, item)
                else:
                    item_element.text = str(item)

        # Handle nested dataclasses
        elif hasattr(field_value, '__dataclass_fields__'):
            child_element = ET.SubElement(element, field_name)
            dataclass_to_xml(child_element, field_value)
        # Handle simple fields (e.g., Name, Identifier, Type)
        else:
            child_element = ET.SubElement(element, field_name)
            child_element.text = str(field_value)

def serialize_to_rgn(root_obj: Any) -> str:
    root = ET.Element("StageOverviewRegions")

    # Serialize Regions
    regions_element = ET.SubElement(root, "Regions")
    shape_list_element = ET.SubElement(regions_element, "ShapeList")
    items_element = ET.SubElement(shape_list_element, "Items")
    for i, shape in enumerate(root_obj.Regions.ShapeList.Items):
        item_element = ET.SubElement(items_element, f"Item{i}")
        dataclass_to_xml(item_element, shape)
    ET.SubElement(shape_list_element, "FillMaskMode").text = root_obj.Regions.ShapeList.FillMaskMode
    ET.SubElement(shape_list_element, "VertexUnitMode").text = root_obj.Regions.ShapeList.VertexUnitMode

    # Serialize DefinedRegions
    defined_regions_element = ET.SubElement(root, "DefinedRegions")
    for entry in root_obj.DefinedRegions.Entries:
        ET.SubElement(
            defined_regions_element,
            "Entry",
            attrib=asdict(entry)
		)

    # Serialize StackList
    stack_list_element = ET.SubElement(root, "StackList")
    for entry in root_obj.StackList.Entries:
        ET.SubElement(
            stack_list_element,
            "Entry",
            attrib=asdict(entry)
		)

    return ET.tostring(root, encoding='unicode', method='xml')