"""
Extended GEE export: multi-region chips, bi-temporal pairs, optional SAR co-export.
Builds on the single-AOI single-window version — same grid + cloud-mask core.
"""

import ee
import time
import csv
from pathlib import Path

ee.Initialize(project="gen-lang-client-0864516812")



# ---------------------------------------------------------------------------
# CONFIG — multiple AOIs matching your actual deployment domains
# ---------------------------------------------------------------------------
AOIS = {
    "kolkata_urban": ee.Geometry.Rectangle([88.20, 22.40, 88.50, 22.65]),
    "gbm_delta_mangrove": ee.Geometry.Rectangle([88.80, 21.60, 89.20, 22.00]),  # Sundarbans-ish, swap for your real zone
    "agri_belt": ee.Geometry.Rectangle([87.80, 23.00, 88.20, 23.30]),          # farmland contrast class
}

CHIP_SIZE_M = 2560          # 256px chips at 10m/px — matches common ViT input tiling better than 5km
SCALE = 10
OPTICAL_BANDS = ["B2", "B3", "B4", "B8", "B11"]
CLOUD_PROB_THRESH = 20
MAX_CLOUD_COVER = 30

# For change-VQA pairs: (before, after) windows over the SAME AOI
CHANGE_WINDOW_PAIRS = [
    (("2019-11-01", "2020-02-28"), ("2024-11-01", "2025-02-28")),  # ~5yr urban growth
]

# For domain-adaptation chips: just need coverage + season diversity, no pairing
ADAPTATION_WINDOWS = [
    ("2024-11-01", "2025-02-28"),
    ("2024-06-01", "2024-09-30"),
]

EXPORT_SAR = True  # co-export Sentinel-1 for fusion testing

DRIVE_FOLDER = "jatayu_training_chips"
MAX_CONCURRENT_TASKS = 20
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CLOUD MASKING + COLLECTIONS  (unchanged from before)
# ---------------------------------------------------------------------------
def mask_clouds(img):
    cloud_prob = ee.Image(img.get("s2cloudless")).select("probability")
    return img.updateMask(cloud_prob.lt(CLOUD_PROB_THRESH))


def get_s2_collection(aoi, start, end):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi).filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
    )
    s2cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi).filterDate(start, end)
    )
    joined = ee.Join.saveFirst("s2cloudless").apply(
        primary=s2, secondary=s2cloudless,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined).map(mask_clouds).select(OPTICAL_BANDS)


def get_s1_collection(aoi, start, end):
    """VV+VH, IW mode — matches the sigma0 dB range your fusion ADR relies on."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi).filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )


# ---------------------------------------------------------------------------
# GRID TILING  (unchanged logic, now per-AOI)
# ---------------------------------------------------------------------------
def make_grid(aoi, chip_size_m):
    bounds = aoi.bounds()
    coords = bounds.coordinates().get(0).getInfo()
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)

    mid_lat = (min_lat + max_lat) / 2
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * abs(ee.Number(mid_lat).multiply(3.14159 / 180).cos().getInfo())
    step_lat = chip_size_m / m_per_deg_lat
    step_lon = chip_size_m / m_per_deg_lon

    chips = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            chips.append(ee.Geometry.Rectangle([lon, lat, lon + step_lon, lat + step_lat]))
            lon += step_lon
        lat += step_lat
    return chips


# ---------------------------------------------------------------------------
# EXPORT HELPERS
# ---------------------------------------------------------------------------
def submit_export(image, chip_geom, task_name, tasks, metadata_rows, extra_meta=None):
    task = ee.batch.Export.image.toDrive(
        image=image.clip(chip_geom),
        description=task_name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=task_name,
        region=chip_geom,
        scale=SCALE,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )
    task.start()
    tasks.append(task)

    centroid = chip_geom.centroid().coordinates().getInfo()
    row = {"task_name": task_name, "centroid_lon": centroid[0], "centroid_lat": centroid[1]}
    if extra_meta:
        row.update(extra_meta)
    metadata_rows.append(row)
    print(f"  started {task_name}")


def wait_for_slot(tasks):
    while len([t for t in tasks if t.status()["state"] in ("READY", "RUNNING")]) >= MAX_CONCURRENT_TASKS:
        print("  export queue full, waiting...")
        time.sleep(30)


# ---------------------------------------------------------------------------
# MODE 1 — domain adaptation chips (diverse coverage, optical [+ SAR])
# ---------------------------------------------------------------------------
def export_adaptation_chips(region_name, aoi, tasks, metadata_rows):
    grid = make_grid(aoi, CHIP_SIZE_M)
    print(f"\n=== {region_name}: {len(grid)} chips × {len(ADAPTATION_WINDOWS)} windows ===")

    for start, end in ADAPTATION_WINDOWS:
        date_label = f"{start}_{end}".replace("-", "")
        s2 = get_s2_collection(aoi, start, end)
        s1 = get_s1_collection(aoi, start, end) if EXPORT_SAR else None

        for i, chip in enumerate(grid):
            img = s2.filterBounds(chip).median()
            if img.bandNames().size().getInfo() == 0:
                continue

            wait_for_slot(tasks)
            name = f"adapt_{region_name}_{date_label}_chip{i:04d}_opt"
            submit_export(img, chip, name, tasks, metadata_rows,
                          {"region": region_name, "window": date_label, "modality": "optical", "chip_idx": i})

            if EXPORT_SAR:
                sar_img = s1.filterBounds(chip).median()
                if sar_img.bandNames().size().getInfo() > 0:
                    wait_for_slot(tasks)
                    sar_name = f"adapt_{region_name}_{date_label}_chip{i:04d}_sar"
                    submit_export(sar_img, chip, sar_name, tasks, metadata_rows,
                                  {"region": region_name, "window": date_label, "modality": "sar", "chip_idx": i})


# ---------------------------------------------------------------------------
# MODE 2 — bi-temporal change pairs (feeds change-VQA training)
# ---------------------------------------------------------------------------
def export_change_pairs(region_name, aoi, tasks, metadata_rows):
    grid = make_grid(aoi, CHIP_SIZE_M)

    for pair_idx, (before_window, after_window) in enumerate(CHANGE_WINDOW_PAIRS):
        print(f"\n=== {region_name}: change pair {pair_idx} ({before_window} -> {after_window}) ===")
        s2_before = get_s2_collection(aoi, *before_window)
        s2_after = get_s2_collection(aoi, *after_window)

        for i, chip in enumerate(grid):
            img_before = s2_before.filterBounds(chip).median()
            img_after = s2_after.filterBounds(chip).median()
            if img_before.bandNames().size().getInfo() == 0 or img_after.bandNames().size().getInfo() == 0:
                continue

            pair_id = f"{region_name}_pair{pair_idx}_chip{i:04d}"

            wait_for_slot(tasks)
            submit_export(img_before, chip, f"change_{pair_id}_before", tasks, metadata_rows,
                          {"region": region_name, "pair_id": pair_id, "role": "before", "chip_idx": i})

            wait_for_slot(tasks)
            submit_export(img_after, chip, f"change_{pair_id}_after", tasks, metadata_rows,
                          {"region": region_name, "pair_id": pair_id, "role": "after", "chip_idx": i})


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    all_tasks = []
    metadata_rows = []

    for region_name, aoi in AOIS.items():
        export_adaptation_chips(region_name, aoi, all_tasks, metadata_rows)
        export_change_pairs(region_name, aoi, all_tasks, metadata_rows)

    csv_path = OUT_DIR / "export_metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\n{len(all_tasks)} tasks submitted across {len(AOIS)} regions. Metadata -> {csv_path}")
    print("Monitor at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
