"""
Coordinate Map Manager for CAE System.

Manages EHR field coordinate mappings for RPA injection.
Supports multiple EHR types with saved configurations.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

from config import DATA_DIR
from utils.logging import get_logger
from utils.exceptions import CoordinateMapError

logger = get_logger(__name__)

COORDINATE_MAPS_DIR = DATA_DIR / "coordinate_maps"


@dataclass
class FieldCoordinate:
    """Coordinate mapping for a single EHR field."""
    field_name: str
    x: int
    y: int
    width: int
    height: int
    field_type: str = "input"  # input, button, dropdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "field_type": self.field_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldCoordinate":
        return cls(
            field_name=data["field_name"],
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            field_type=data.get("field_type", "input"),
        )

    @property
    def center(self) -> tuple:
        """Get center point of field."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> List[int]:
        """Get bounding box [x, y, width, height]."""
        return [self.x, self.y, self.width, self.height]


@dataclass
class CoordinateMap:
    """Complete coordinate mapping for an EHR type."""
    id: str
    ehr_type: str
    name: str
    fields: Dict[str, FieldCoordinate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ehr_type": self.ehr_type,
            "name": self.name,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoordinateMap":
        fields = {
            k: FieldCoordinate.from_dict(v)
            for k, v in data.get("fields", {}).items()
        }
        return cls(
            id=data["id"],
            ehr_type=data["ehr_type"],
            name=data["name"],
            fields=fields,
        )

    def get_field(self, field_name: str) -> Optional[FieldCoordinate]:
        """Get field coordinate by name."""
        return self.fields.get(field_name)

    def get_click_point(self, field_name: str) -> Optional[tuple]:
        """Get click point (center) for a field."""
        field = self.get_field(field_name)
        return field.center if field else None


class CoordinateMapManager:
    """
    Manages EHR coordinate mappings.

    Loads, saves, and provides coordinate mappings for RPA.
    """

    def __init__(self, maps_dir: Path = COORDINATE_MAPS_DIR):
        self.maps_dir = maps_dir
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, CoordinateMap] = {}

    def _get_map_path(self, ehr_type: str, name: str) -> Path:
        """Get file path for a coordinate map."""
        safe_name = name.replace(" ", "_").lower()
        return self.maps_dir / f"{ehr_type}_{safe_name}.json"

    async def save_map(self, coord_map: CoordinateMap) -> str:
        """
        Save coordinate map to file.

        Args:
            coord_map: CoordinateMap to save

        Returns:
            Map ID
        """
        path = self._get_map_path(coord_map.ehr_type, coord_map.name)

        try:
            with open(path, "w") as f:
                json.dump(coord_map.to_dict(), f, indent=2)

            # Update cache
            cache_key = f"{coord_map.ehr_type}:{coord_map.name}"
            self._cache[cache_key] = coord_map

            logger.info("Coordinate map saved", id=coord_map.id, path=str(path))
            return coord_map.id

        except Exception as e:
            raise CoordinateMapError(coord_map.name, f"Failed to save: {str(e)}")

    async def load_map(self, ehr_type: str, name: str) -> Optional[CoordinateMap]:
        """
        Load coordinate map from file.

        Args:
            ehr_type: EHR type identifier
            name: Map name

        Returns:
            CoordinateMap or None
        """
        cache_key = f"{ehr_type}:{name}"

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_map_path(ehr_type, name)

        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            coord_map = CoordinateMap.from_dict(data)
            self._cache[cache_key] = coord_map

            return coord_map

        except Exception as e:
            logger.warning("Failed to load coordinate map", path=str(path), error=str(e))
            return None

    async def list_maps(self, ehr_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available coordinate maps.

        Args:
            ehr_type: Optional filter by EHR type

        Returns:
            List of map metadata
        """
        maps = []

        for path in self.maps_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)

                if ehr_type and data.get("ehr_type") != ehr_type:
                    continue

                maps.append({
                    "id": data["id"],
                    "ehr_type": data["ehr_type"],
                    "name": data["name"],
                    "field_count": len(data.get("fields", {})),
                })

            except Exception:
                continue

        return maps

    async def delete_map(self, ehr_type: str, name: str) -> bool:
        """Delete a coordinate map."""
        path = self._get_map_path(ehr_type, name)

        if path.exists():
            path.unlink()
            cache_key = f"{ehr_type}:{name}"
            self._cache.pop(cache_key, None)
            logger.info("Coordinate map deleted", ehr_type=ehr_type, name=name)
            return True

        return False

    async def create_map_from_detection(
        self,
        ehr_type: str,
        name: str,
        detected_elements: List[Dict[str, Any]],
    ) -> CoordinateMap:
        """
        Create coordinate map from detected UI elements.

        Args:
            ehr_type: EHR type identifier
            name: Map name
            detected_elements: List of detected UI elements with bbox

        Returns:
            Created CoordinateMap
        """
        import uuid

        fields = {}
        for elem in detected_elements:
            label = elem.get("label", "").lower().replace(" ", "_")
            if not label:
                continue

            bbox = elem.get("bbox", [0, 0, 100, 30])
            fields[label] = FieldCoordinate(
                field_name=label,
                x=bbox[0],
                y=bbox[1],
                width=bbox[2],
                height=bbox[3],
                field_type=elem.get("type", "input"),
            )

        coord_map = CoordinateMap(
            id=str(uuid.uuid4()),
            ehr_type=ehr_type,
            name=name,
            fields=fields,
        )

        await self.save_map(coord_map)
        return coord_map


# =============================================================================
# SINGLETON
# =============================================================================

_coordinate_manager: Optional[CoordinateMapManager] = None


def get_coordinate_manager() -> CoordinateMapManager:
    """Get singleton coordinate map manager."""
    global _coordinate_manager
    if _coordinate_manager is None:
        _coordinate_manager = CoordinateMapManager()
    return _coordinate_manager
