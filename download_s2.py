"""
cd ~/jatayu
python3 download_s2_fixed.py
"""
import ee
import requests
import rasterio
from pathlib import Path

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

out = Path('data/demo')
out.mkdir(parents=True, exist_ok=True)

BANDS = ['blue', 'green', 'red', 'nir', 'swir1']
MAX_BYTES = 50 * 1024 * 1024  # GEE sync download cap


def download_s2(name, bbox, date_start, date_end, scale=10, max_retries=4):
    """
    Download a cloud-filtered Sentinel-2 composite from GEE.
    Stays under the 50MB sync download cap by:
      - keeping data as int16 (DN values, NOT divided to reflectance) -> half
        the bytes of float32
      - auto-doubling `scale` (halving resolution) on a 400 "Total request
        size" error, up to `max_retries` times
    Reflectance scaling (divide by 10000) should happen when you LOAD the
    tif later (e.g. in rasterio/numpy), not before download.
    """
    print(f'\n=== {name} ===')

    aoi = ee.Geometry.Rectangle(bbox)

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(aoi)
           .filterDate(date_start, date_end)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
           .sort('CLOUDY_PIXEL_PERCENTAGE')
           .first())

    info = col.getInfo()
    if info is None:
        print('No scenes found!')
        return None
    print(f"Scene: {info['id']}")

    # Keep as int16 DN values -- do NOT divide/toFloat before download.
    # This halves bytes-per-pixel vs float32 and buys real headroom.
    image = (col
             .select(['B2', 'B3', 'B4', 'B8', 'B11'], BANDS)
             .clip(aoi)
             .toInt16())

    current_scale = scale
    for attempt in range(max_retries):
        try:
            url = image.getDownloadURL({
                'scale': current_scale,
                'region': aoi,
                'format': 'GEO_TIFF',
                'bands': BANDS,
            })

            print(f'Downloading @ scale={current_scale}m...', end=' ', flush=True)
            resp = requests.get(url)

            if resp.status_code != 200:
                print(f'FAILED: HTTP {resp.status_code}: {resp.text[:200]}')
                return None

            fname = out / f'{name}.tif'
            fname.write_bytes(resp.content)

            with rasterio.open(fname) as src:
                print(f'{src.width}x{src.height}, {src.count} bands')

            with rasterio.open(fname, 'r+') as src:
                for i, band_name in enumerate(BANDS, 1):
                    src.set_band_description(i, band_name)

            print(f'Saved: {fname}  (int16 DN -- divide by 10000 for reflectance on load)')
            return fname

        except ee.ee_exception.EEException as e:
            if 'Total request size' in str(e) and attempt < max_retries - 1:
                current_scale *= 2
                print(f'too big, retrying at scale={current_scale}m...')
                continue
            print(f'FAILED: {e}')
            return None

    print('FAILED: exceeded max_retries, region still too large. '
          'Consider ee.batch.Export.image.toDrive for this AOI instead.')
    return None


# === NEPAL — pre-flood (dry season) ===
download_s2('nepal_pre_flood',
            [85.28, 27.58, 85.42, 27.72],
            '2026-03-01', '2026-05-31', scale=10)

# === NEPAL — post-flood (monsoon/recent) ===
download_s2('nepal_post_flood',
            [85.28, 27.58, 85.42, 27.72],
            '2026-07-15', '2026-08-29', scale=10)

# === PUNJAB — kharif crop stress ===
download_s2('punjab_crop',
            [75.08, 30.55, 75.32, 30.82],
            '2026-08-01', '2026-08-29', scale=10)

# === ODISHA — coastal paddy (bonus if time) ===
download_s2('odisha_paddy',
            [86.20, 20.30, 86.50, 20.55],
            '2026-08-01', '2026-08-29', scale=10)

print('\n=== ALL FILES ===')
for f in sorted(out.glob('*.tif')):
    with rasterio.open(f) as src:
        print(f'  {f.name}: {src.count} bands, {src.width}x{src.height}')
