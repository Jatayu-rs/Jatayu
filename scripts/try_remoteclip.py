from pathlib import Path 

import numpy as np 
import open_clip 
import rasterio 
import torch 
from huggingface_hub import hf_hub_download 
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device ", device)
import logging
logging.getLogger("open_clip").setLevel(logging.ERROR)

# pretrained=None silences the "no pretrained weights loaded" warning —
# we're about to load RemoteCLIP's weights manually anyway
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
tokenizer = open_clip.get_tokenizer("ViT-B-32")


ckpt = hf_hub_download(
    "chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", cache_dir="checkpoints"
)

model.load_state_dict(torch.load(ckpt, map_location="cpu"))

model = model.to(device).eval()


def strech(band, lo=2, hi=98):
    finite = band[np.isfinite(band)]
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    scale = np.clip((band - p_lo) / (p_hi - p_lo), 0, 1)
    return (np.nan_to_num(scale) * 255).astype(np.uint8)


def load_rgb(path, window=None):
    with rasterio.open(path) as src:
        red = src.read(3, window=window).astype("float32")
        green = src.read(2, window=window).astype("float32")
        blue = src.read(1, window=window).astype("float32")
    rgb = np.stack([strech(b) for b in (red, green, blue)], axis=-1)
    return Image.fromarray(rgb)


def answer(image: Image.Image, candidates: list[str]) -> list[tuple[str, float]]:
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = tokenizer(candidates).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    return sorted(zip(candidates, probs), key=lambda x: -x[1])


tile = rasterio.windows.Window(600, 400, 900, 900)  # a chunk with river + city
img = load_rgb(ROOT / "data/samples/kolkata_optical.tif", window=tile)
img.save(ROOT / "outputs/clip_input.png")  # ALWAYS look at the input


def water_fraction(path, window, threshold=0.10):
    """Fraction of pixels that MNDWI calls water. Finds water-dominated tiles."""
    with rasterio.open(path) as src:
        green = src.read(2, window=window).astype("float32")
        swir = src.read(5, window=window).astype("float32")
    denom = green + swir
    mndwi = np.zeros_like(denom)
    np.divide(green - swir, denom, out=mndwi, where=np.abs(denom) > 1e-6)
    return float((mndwi > threshold).mean())


SRC = ROOT / "data/samples/kolkata_optical.tif"
SIZE = 600

# Slide a grid of windows across the image and score each one.
candidates = []
with rasterio.open(SRC) as src:
    for row in range(0, src.height - SIZE, 300):
        for col in range(0, src.width - SIZE, 300):
            w = rasterio.windows.Window(col, row, SIZE, SIZE)
            candidates.append((mndwi_mean(SRC, w), col, row, w))

candidates.sort()
driest = candidates[0]
wettest = candidates[-1]

print(f"driest  window at ({driest[1]},{driest[2]})  MNDWI mean {driest[0]:+.3f}")
print(f"wettest window at ({wettest[1]},{wettest[2]})  MNDWI mean {wettest[0]:+.3f}")

water_question = [
    "a satellite image containing a river or water body",
    "a satellite image with no visible water",
]

for label, (score, col, row, win) in [("WET", wettest), ("DRY", driest)]:
    # renamed from `img` -> `tile_img` so it doesn't clobber the original city tile above
    tile_img = load_rgb(SRC, window=win)
    tile_img.save(ROOT / f"outputs/clip_{label.lower()}.png")
    print(f"\n--- {label} tile ({col},{row}) · MNDWI {score:+.3f} ---")
    for text, p in answer(tile_img, water_question):
        print(f"  {p:6.1%}  {text}")

tests = {
    "water present": [
        "a satellite image containing a river or water body",
        "a satellite image with no visible water",
    ],
    "urban density": [
        "a satellite image of a dense urban area with many buildings",
        "a satellite image of farmland and open fields",
        "a satellite image of dense forest",
    ],
}

# `img` here is still the original Kolkata city crop saved as clip_input.png —
# no longer overwritten by the WET/DRY loop
for name, candidates in tests.items():
    print(f"\n--- {name} ---")
    for text, p in answer(img, candidates):
        print(f"  {p:6.1%}  {text}")

print(f"\nVRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
