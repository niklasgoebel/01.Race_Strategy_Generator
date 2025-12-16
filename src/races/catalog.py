from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


CATALOG_PATH = Path("data/races/catalog.json")


@dataclass(frozen=True)
class AidStation:
    name: str
    km: float


@dataclass(frozen=True)
class Race:
    id: str
    name: str
    gpx_path: str
    provider: str | None
    distance_km_label: float | None
    gain_m_label: float | None
    aid_stations: List[AidStation]


def _load_catalog() -> Dict[str, dict]:
    if not CATALOG_PATH.exists():
        return {}
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        return data

    # Backwards compatibility: if someone kept the old list format, convert it.
    if isinstance(data, list):
        out: Dict[str, dict] = {}
        for r in data:
            if isinstance(r, dict) and "id" in r:
                out[str(r["id"])] = r
        return out

    return {}


def list_races() -> List[Race]:
    catalog = _load_catalog()
    out: List[Race] = []

    for race_id, r in catalog.items():
        if not isinstance(r, dict):
            continue

        aid_raw = r.get("aid_stations") or []
        aids: List[AidStation] = []
        for a in aid_raw:
            if not isinstance(a, dict):
                continue
            try:
                aids.append(AidStation(name=str(a["name"]), km=float(a["km"])))
            except Exception:
                continue

        out.append(
            Race(
                id=str(r.get("id", race_id)),
                name=str(r.get("name", race_id)),
                gpx_path=str(r["gpx_path"]),
                provider=(str(r["provider"]) if "provider" in r else None),
                distance_km_label=(float(r["distance_km_label"]) if "distance_km_label" in r else None),
                gain_m_label=(float(r["gain_m_label"]) if "gain_m_label" in r else None),
                aid_stations=aids,
            )
        )

    # Nice UX: stable order in dropdown
    out.sort(key=lambda x: x.name.lower())
    return out


def load_race_gpx_bytes(race_id: str) -> bytes:
    catalog = _load_catalog()
    r = catalog.get(race_id)

    if not isinstance(r, dict):
        raise FileNotFoundError(f"Race id not found in catalog: {race_id}")

    gpx_path = Path(str(r["gpx_path"]))
    if not gpx_path.exists():
        raise FileNotFoundError(f"GPX file missing: {gpx_path}")

    return gpx_path.read_bytes()