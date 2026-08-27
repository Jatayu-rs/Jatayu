import pystac_client


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


catalog = pystac_client.Client.open(CATALOG_URL)

# Kolkata
bbox = [
    88.30,
    22.50,
    88.45,
    22.65,
]

print("Searching Sentinel-2...")
print("BBox:", bbox)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-01-01/2026-08-27",
)

items = list(search.items())

print()
print("Found:", len(items))

if not items:
    print("No Sentinel-2 scenes found.")
    raise SystemExit(1)

# Sort by cloud cover
items.sort(
    key=lambda item: item.properties.get(
        "eo:cloud_cover",
        999
    )
)

print()
print("=== BEST SCENES ===")

for item in items[:10]:

    print()
    print("ID:", item.id)
    print("DATE:", item.datetime)
    print(
        "CLOUD:",
        item.properties.get(
            "eo:cloud_cover"
        )
    )
    print("ASSETS:", list(item.assets.keys()))
