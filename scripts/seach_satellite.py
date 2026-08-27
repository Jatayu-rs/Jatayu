from datetime import datetime
import pystac_client


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search_sentinel2(
    bbox,
    start_date,
    end_date,
    max_cloud=20,
):
    catalog = pystac_client.Client.open(CATALOG_URL)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lt": max_cloud
            }
        },
    )

    items = list(search.items())

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100
        )
    )

    return items


if __name__ == "__main__":

    # Kolkata example
    bbox = [
        88.30,
        22.50,
        88.45,
        22.65,
    ]

    items = search_sentinel2(
        bbox,
        "2026-07-01",
        "2026-08-27",
        max_cloud=20,
    )

    print("Found:", len(items))

    for item in items[:5]:
        print()
        print("ID:", item.id)
        print("Date:", item.datetime)
        print(
            "Cloud:",
            item.properties.get("eo:cloud_cover")
        )

        print("Assets:")

        for name in item.assets:
            print(" ", name)
