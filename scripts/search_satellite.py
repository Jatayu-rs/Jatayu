"""
Satellite scene discovery for Jatayu.

Uses Microsoft Planetary Computer STAC to discover Sentinel-2 L2A scenes.

This module ONLY discovers scenes.
It does not download imagery or run analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pystac_client


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class SatelliteScene:
    """Small, API-friendly description of a satellite scene."""

    id: str
    datetime: datetime | None
    cloud_cover: float
    collection: str
    bbox: list[float]
    assets: list[str]

    # Keep the original STAC item available internally.
    item: Any


def search_sentinel2(
    bbox: list[float],
    start_date: str,
    end_date: str,
    *,
    max_cloud: float = 20.0,
    limit: int = 20,
) -> list[SatelliteScene]:
    """
    Search Sentinel-2 L2A scenes intersecting an AOI.

    Parameters
    ----------
    bbox:
        [min_lon, min_lat, max_lon, max_lat]

    start_date:
        Start date, e.g. "2026-07-01"

    end_date:
        End date, e.g. "2026-08-27"

    max_cloud:
        Maximum scene cloud percentage.

    limit:
        Maximum number of scenes returned.

    Returns
    -------
    list[SatelliteScene]
        Scenes sorted by cloud cover.
    """

    if len(bbox) != 4:
        raise ValueError(
            "bbox must contain "
            "[min_lon, min_lat, max_lon, max_lat]"
        )

    min_lon, min_lat, max_lon, max_lat = bbox

    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bbox coordinates.")

    if not 0 <= max_cloud <= 100:
        raise ValueError("max_cloud must be between 0 and 100.")

    if limit < 1:
        raise ValueError("limit must be >= 1.")

    catalog = pystac_client.Client.open(CATALOG_URL)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lt": max_cloud,
            }
        },
    )

    items = list(search.items())

    # Lowest cloud cover first.
    items.sort(
        key=lambda item: float(
            item.properties.get(
                "eo:cloud_cover",
                100.0,
            )
        )
    )

    scenes: list[SatelliteScene] = []

    for item in items[:limit]:
        cloud = float(
            item.properties.get(
                "eo:cloud_cover",
                100.0,
            )
        )

        scenes.append(
            SatelliteScene(
                id=item.id,
                datetime=item.datetime,
                cloud_cover=cloud,
                collection="sentinel-2-l2a",
                bbox=list(item.bbox or bbox),
                assets=list(item.assets.keys()),
                item=item,
            )
        )

    return scenes


def best_sentinel2_scene(
    bbox: list[float],
    start_date: str,
    end_date: str,
    *,
    max_cloud: float = 20.0,
) -> SatelliteScene | None:
    """
    Return the best available Sentinel-2 scene.

    Currently the ranking criterion is lowest cloud cover.

    We will improve this later to consider:
        - AOI coverage
        - cloud-free AOI percentage
        - acquisition date
        - spatial resolution
        - seasonal relevance
    """

    scenes = search_sentinel2(
        bbox,
        start_date,
        end_date,
        max_cloud=max_cloud,
        limit=1,
    )

    if not scenes:
        return None

    return scenes[0]


def scene_summary(
    scene: SatelliteScene,
) -> dict[str, Any]:
    """Convert a scene into a JSON-friendly dictionary."""

    return {
        "id": scene.id,
        "datetime": (
            scene.datetime.isoformat()
            if scene.datetime is not None
            else None
        ),
        "cloud_cover": scene.cloud_cover,
        "collection": scene.collection,
        "bbox": scene.bbox,
        "assets": scene.assets,
    }


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Kolkata demo AOI
    # ------------------------------------------------------------

    bbox = [
        88.30,
        22.50,
        88.45,
        22.65,
    ]

    print("Searching Sentinel-2...")
    print("BBox:", bbox)

    scenes = search_sentinel2(
        bbox,
        "2026-07-01",
        "2026-08-27",
        max_cloud=20,
        limit=10,
    )

    print()
    print("Found:", len(scenes))

    print()
    print("=== BEST SCENES ===")

    for scene in scenes:

        print()
        print("ID:", scene.id)
        print("DATE:", scene.datetime)
        print("CLOUD:", scene.cloud_cover)
        print("ASSETS:", scene.assets)

    print()
    print("=== BEST SCENE ===")

    best = best_sentinel2_scene(
        bbox,
        "2026-07-01",
        "2026-08-27",
        max_cloud=20,
    )

    if best is None:
        print("No suitable Sentinel-2 scene found.")
    else:
        print(scene_summary(best))
