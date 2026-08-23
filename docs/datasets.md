# Dataset Reference — Licensing, Download & Storage
### Person D — SatQuery AI

⚠️ **Read the flags in Section 6 before anyone downloads or redistributes anything.** Two of the four datasets carry restrictions that limit what license we can put on our own repo.

---

## 1. BigEarthNet (v2.0 / "reBEN")

| Field | Detail |
|---|---|
| **Official site** | https://bigearth.net/ |
| **Download URL — Sentinel-2 (optical)** | https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1 |
| **Download URL — Sentinel-1 (SAR)** | https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1 |
| **Reference maps** | https://zenodo.org/records/10891137/files/Reference_Maps.tar.zst?download=1 |
| **Metadata** | https://zenodo.org/records/10891137/files/metadata.parquet?download=1 |
| **Exact size on disk** | S2: **~59 GiB** (compressed .tar.zst as listed on the official site). S1: **~51 GiB**. Combined S1+S2 ≈ **110 GiB** compressed; expect meaningfully more once decompressed — budget storage accordingly. |
| **License** | **Community Data License Agreement – Permissive, Version 1.0 (CDLA-Permissive-1.0)**. Full text: https://cdla.dev/permissive-1-0/ |
| **Permits our use?** | **Yes.** CDLA-Permissive-1.0 allows use, modification, and redistribution, including commercial use, with no share-alike or non-commercial restriction. No conflict with putting our own license on the repo. |
| **Account/registration required?** | **No.** Direct download from Zenodo, no login or request form. |
| **Folder structure after extraction** | Per-patch folders named `<Sentinel-2_tile_ID>_<patch_number>/`, each containing individual band GeoTIFFs (e.g. `B01.tif`...`B12.tif` for S2) plus a `_labels_metadata.json` per patch. The **official "Dataset Description Document"** (https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf) documents the exact directory layout and naming convention — read this before writing any dataloader, since v2.0's structure differs from v1.0 used in older tutorials. |

---

## 2. RSVQA (Low Resolution + High Resolution)

| Field | Detail |
|---|---|
| **Official site** | https://rsvqa.sylvainlobry.com/ |
| **Download URL — LR** | https://zenodo.org/record/6344334 |
| **Download URL — HR** | https://zenodo.org/record/6344367 |
| **Exact size on disk** | **LR: 150.5 MB total** (images 95.0 MB + question/answer JSONs ~55 MB). **HR: 14.4 GB total** (dominated by `Images.tar` at 13.5 GB). |
| **License** | **CC BY 4.0** (Creative Commons Attribution 4.0 International) for both LR and HR, per Zenodo listing. Full text: https://creativecommons.org/licenses/by/4.0/legalcode |
| **Permits our use?** | **Yes**, including commercial/redistribution, as long as we credit the creators (Lobry, Marcos, Murray, Tuia) per the TGRS 2020 paper. No share-alike or non-commercial clause. |
| **Account/registration required?** | **No.** Direct download, no login. |
| **Folder structure after extraction** | **LR:** `Images_LR.zip` → flat folder of `.tif` images numbered by ID; JSONs (`LR_split_{train,val,test}_{images,questions,answers}.json`) map image IDs to Q/A pairs. **HR:** `Images.tar` → flat folder of `.tif` tiles from USGS High-Resolution Orthoimagery; paired JSONs follow the same `USGS_split_{train,val,test}_{images,questions,answers}.json` naming, plus a `USGS_split_test_phili_*` variant for the "Test Set 2" unseen-area split mentioned in the paper. |

---

## 3. VRSBench

| Field | Detail |
|---|---|
| **Official site** | https://vrsbench.github.io/ · GitHub: https://github.com/lx709/VRSBench |
| **Download URL** | https://huggingface.co/datasets/xiang709/VRSBench |
| **Exact size on disk** | Not published as a single figure on the dataset card; scale is **29,614 images** with captions, object references, and QA pairs. **Action item:** download and run `du -sh` once pulled — flag back to the team with the real number since HF doesn't list it directly. |
| **License** | ⚠️ **CONFLICTING SOURCES — see flag below.** |
| **Permits our use?** | **Needs resolution before we build on this dataset.** See Section 6. |
| **Account/registration required?** | **No** login required to browse, but Hugging Face streaming/download works via `datasets` library (`load_dataset("xiang709/VRSBench", streaming=True)`) — no gated-access request needed either way. |
| **Folder structure after extraction** | Delivered via Hugging Face `datasets` library rather than a manually-extracted archive — structure is exposed as dataset splits/columns (image, caption, object references, QA pairs) rather than a folder tree. If we need raw files, check the GitHub repo's `extract_patch_json.py` for the underlying file layout, since VRSBench is built by pulling attributes from existing object-detection datasets (DOTA-v2, DIOR). |

---

## 4. CDVQA (Change Detection VQA)

| Field | Detail |
|---|---|
| **Official site / QA annotations** | https://github.com/YZHJessica/CDVQA |
| **Underlying imagery** | SECOND dataset — https://captain-whu.github.io/SCD/ (Google Drive link on that page) |
| **Exact size on disk** | Not stated by either source. The QA JSON files (`Train/Val/Test/Test2_{images,questions,answers}.json`) are small (low tens of MB based on typical JSON annotation sizes for ~122K QA pairs). The bulk of the size is the underlying **SECOND imagery**: 2,968 pairs of 512×512 images — **budget several GB**, but confirm once downloaded since WHU doesn't publish an exact figure either. |
| **License — QA annotations (CDVQA repo)** | **Apache-2.0**, per the repo's listed license. |
| **License — underlying images (SECOND dataset)** | ⚠️ **No license is published on the official SECOND page.** Access is via a Google Drive link with no attached license file — only a request to contact the authors (Kunping Yang, Gui-Song Xia) for questions on use. |
| **Permits our use?** | **Partially resolved, partially not** — see flag below. The QA text/annotations are clearly Apache-2.0 (permissive). The **images themselves have no stated license**, which is the actual gating factor since the images are the dataset. |
| **Account/registration required?** | **No account**, but the SECOND imagery is hosted on Google Drive (not a direct/scripted download) rather than a standard dataset host — factor this into any automated download pipeline. |
| **Folder structure after extraction** | CDVQA repo ships flat JSON files at the repo root (`Train_images.json`, `Train_questions.json`, `Train_answers.json`, and equivalents for `Val`, `Test`, `Test2`) mapping to image IDs. The actual bi-temporal image pairs come from SECOND separately — expect `im1/`, `im2/` (pre-/post-event images) and `label1/`, `label2/` (per-date semantic maps) based on how SECOND is structured in other papers reusing it; confirm exact folder names once the Google Drive archive is pulled. |

---

## 5. Quick-reference table (as requested in the task)

| Dataset | Download URL | Size | License | Commercial/redistribution OK? | Account needed? |
|---|---|---|---|---|---|
| BigEarthNet v2.0 | zenodo.org/records/10891137 | ~59 GiB (S2) + ~51 GiB (S1) | CDLA-Permissive-1.0 | ✅ Yes | No |
| RSVQA-LR | zenodo.org/record/6344334 | 150.5 MB | CC BY 4.0 | ✅ Yes (attribution required) | No |
| RSVQA-HR | zenodo.org/record/6344367 | 14.4 GB | CC BY 4.0 | ✅ Yes (attribution required) | No |
| VRSBench | huggingface.co/datasets/xiang709/VRSBench | Not published (~29.6K images) | ⚠️ Conflicting — see below | ⚠️ **Unresolved** | No |
| CDVQA (QA text) | github.com/YZHJessica/CDVQA | Small (JSON only) | Apache-2.0 | ✅ Yes | No |
| CDVQA (images, via SECOND) | captain-whu.github.io/SCD (Google Drive) | Not published | ⚠️ **No license stated** | ⚠️ **Unresolved** | No account, but Drive-hosted |

---

## 6. 🚩 Flags — read before choosing our repo license

### VRSBench — license conflict between sources
- The **arXiv paper appendix** (Section D, "Data License Confirmation") states VRSBench is released under **CC-BY-4.0**, which permits unrestricted use including commercial.
- The **Hugging Face dataset card** (the actual download page) states it is released under **CC BY-**_**NC**_**-4.0 (Non-Commercial)**.
- These directly contradict each other. **This needs to be resolved before we rely on VRSBench for anything beyond internal research/hackathon use** — a non-commercial clause would forbid any commercial use or sublicensing downstream, and would force our own repo/dataset mix into a non-commercial license if it's the binding version.
- **Recommended next step:** email the VRSBench authors (contact listed on https://vrsbench.github.io) to confirm which license is authoritative, and keep a copy of their reply. Until confirmed, treat VRSBench as **non-commercial-only** (the more restrictive reading) for any decision-making — e.g. don't use it to justify a permissive license on our own repo, and don't include VRSBench-derived weights/data in anything we'd call "open" without this caveat stated.

### CDVQA — underlying SECOND imagery has no published license
- The QA annotations on the CDVQA GitHub repo are Apache-2.0 — genuinely permissive.
- But those annotations are meaningless without the **SECOND dataset images**, and the official SECOND page (captain-whu.github.io/SCD) does not publish a license for the imagery at all — just a Google Drive link and an email contact for questions.
- **This is a no-redistribution-clarity situation, not a confirmed restriction** — but the absence of a license is not the same as permission. Academic remote-sensing datasets distributed this way are conventionally "research use, contact authors for anything beyond that," even when unstated.
- **Recommended next step:** email Kunping Yang / Gui-Song Xia (contacts on the SECOND page) before assuming we can redistribute SECOND-derived imagery in our repo or demo materials. Safe to use internally for training/eval during the hackathon; **do not bundle the raw images into our public repo** until this is confirmed in writing.

### What this means for our repo's license
Because of the two flags above, **our repo cannot currently claim a fully permissive license if it bundles or redistributes VRSBench or SECOND/CDVQA data directly.** Recommended approach until resolved:
1. License our own code (model adapters, controller, GUI) under a standard permissive license (MIT/Apache-2.0) — this part is unaffected.
2. **Do not commit or redistribute the raw VRSBench or SECOND images in the repo.** Document download instructions instead (pointing to the official sources above) so users pull the data themselves under whatever terms those sources actually apply — this sidesteps us needing to resolve the conflict ourselves before the hackathon deadline.
3. Note this limitation explicitly in the repo's README/LICENSE section so judges and future contributors aren't misled about what's redistributable.

---

## 7. Sources consulted
- bigearth.net (official site, downloads, license statement)
- zenodo.org records 10891137 (BigEarthNet v2.0), 6344334 (RSVQA-LR), 6344367 (RSVQA-HR)
- rsvqa.sylvainlobry.com (official RSVQA project page)
- huggingface.co/datasets/xiang709/VRSBench (dataset card)
- vrsbench.github.io and the VRSBench arXiv paper (arxiv.org/abs/2406.12384), Appendix D
- github.com/YZHJessica/CDVQA (CDVQA repo, license file)
- captain-whu.github.io/SCD (official SECOND dataset page)
