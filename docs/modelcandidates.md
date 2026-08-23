# Model Candidates

**Owner:** Person F · **Last updated:** 23 Aug 2026
**Scope:** Five candidate models, each answering only the five questions we agreed on.
**Reading note for the tech team:** every "stated" figure below is taken from the paper or repo and is cited. Every "estimate" is mine, marked as such — please sanity-check before relying on it.

---

## TL;DR — the verdict table

| Model | Public checkpoint? | Licence | Runs on a free 16GB T4? | Reported on *our* benchmarks? |
|---|---|---|---|---|
| **TEOChat** | ✅ Yes | Apache-2.0 (+ LLaMA/OpenAI terms) | Yes, quantised (est.) | **Yes — CDVQA 47.2 F1 zero-shot** |
| **RemoteCLIP** | ✅ Yes | Apache-2.0 | Yes, easily — even CPU | No (retrieval/classification only) |
| **Change-Agent** | ✅ Yes | MIT, *academic use only* | Yes, trivially — tiny model | No (LEVIR-MCI only) |
| **GeoGround** | ✅ Yes | ⚠️ Not stated in repo | Yes, quantised (est.) | No (grounding only) |
| **RSGPT** | ❌ **No weights released** | Academic use only | N/A | Yes on paper — RSVQA-LR 92.29% |

**The two lines to remember when someone asks:**

1. **TEOChat is the only candidate that reports a number on a benchmark we are actually scored on (CDVQA), and it was trained on a 16GB GPU.** That makes it the closest match to both our task list and our compute budget.
2. **RSGPT has the best published numbers of the five and we cannot use it**, because the authors never released weights. Anyone who proposes RSGPT has not checked this.

---

## 1. TEOChat

*TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data* — ICLR 2025, Stanford (Ermon group)

**Is there a public checkpoint?**
Yes. Weights and the training dataset (TEOChatlas) are both published.

**Hugging Face link**
- Model: `https://huggingface.co/jirvin16/TEOChat`
- Dataset: `https://huggingface.co/datasets/jirvin16/TEOChatlas`
- Code: `https://github.com/ermongroup/TEOChat`

**Licence**
Apache-2.0 on the model card, **with additional terms inherited from LLaMA 2 and from OpenAI-generated training data**. Practically: fine for a hackathon and for research, but the LLaMA/OpenAI terms would need a lawyer's read before any commercial deployment. Worth one honest line on a slide rather than pretending it's clean Apache.

**Hardware needed for inference**
- *Stated in paper:* trained on a **single NVIDIA A4000, 16GB VRAM**, using LoRA with 8-bit quantisation, on sequences of up to 8 images. Requires Python ≥3.9, PyTorch 2.2.1, CUDA ≥12.1.
- *My estimate:* it is a LLaVA-style 7B (CLIP ViT-L/14 vision encoder + LLaMA 2 + 2-layer MLP projector). Full fp16 inference would be ~14–15GB and too tight on a T4 once you add image tokens; **4-bit quantised inference should land around 5–6GB and fit comfortably.** The fact that the authors *trained* it on 16GB is the strong signal that inference on our Kaggle T4s is realistic.

**What it scored, in its own paper**

| Task / dataset | Result |
|---|---|
| **CDVQA — change QA (zero-shot)** | **47.2 F1** |
| Change QA — xBD | 89.9% accuracy |
| Change QA — S2Looking | 73.4% accuracy |
| Change detection — xBD building damage | 50.0 F1 (specialist model: 26.5) |
| Change detection — S2Looking | 33.6 F1 (specialist: 26.5) |
| Change detection — ABCD (zero-shot) | 85.6 F1 |
| QFabric region QA (2 img / 5 img) | 66.7 / 74.3 F1 |
| Scene classification — AID (zero-shot) | 80.9% accuracy |

Also reported to beat GPT-4o and Gemini 1.5 Pro on multiple temporal tasks.

> ⚠️ **Read the CDVQA number correctly.** 47.2 F1 is *zero-shot* — TEOChat was never trained on CDVQA. It is not a ceiling, it is a floor, and it is the honest baseline for us to beat. Do not put it on a slide as "state of the art on CDVQA," because it isn't.

---

## 2. RemoteCLIP

*RemoteCLIP: A Vision Language Foundation Model for Remote Sensing* — IEEE TGRS

**Is there a public checkpoint?**
Yes — three of them, in standard OpenCLIP format.

**Hugging Face link**
- Model: `https://huggingface.co/chendelong/RemoteCLIP`
- Code: `https://github.com/ChenDelong1999/RemoteCLIP`

**Licence**
**Apache-2.0.** The cleanest licence of the five — no inherited LLaMA or OpenAI terms. If licensing ever comes up in judging, this is the model to point at.

**Hardware needed for inference**
- *Stated in repo:* checkpoints load with `map_location="cpu"`, and inference uses `torch.cuda.amp.autocast()` when a GPU is present. No minimum stated.
- *My estimate:* three sizes are released — **RN50, ViT-B/32 (~150M params), ViT-L/14 (~430M params)**. Even ViT-L/14 is well under 2GB in fp16. **This runs on CPU.** It is effectively free to include and cannot break our demo.

**What it scored, in its own paper**
- Image–text retrieval: **+9.14% mean recall over previous SOTA on RSICD**; also evaluated on RSITMD and UCM (R@1/R@5/R@10).
- Zero-shot classification: **up to +6.39% average accuracy over the CLIP baseline across 12 downstream datasets.**
- A `retrieval.py` evaluation script ships with the repo for reproducing these.

> **Note for the team:** RemoteCLIP is *not* a VQA model — it is a retrieval/embedding model, so it cannot answer questions on its own. Its value to us is as a cheap, always-available component: zero-shot land-cover tagging, and patch-similarity heatmaps for grounding. It cannot be our main answering engine and shouldn't be pitched as one.

---

## 3. Change-Agent

*Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis* — IEEE TGRS 2024

**Is there a public checkpoint?**
Yes — the multi-level change interpretation (MCI) model weights are released as `MCI_model.pth`.

**Hugging Face link**
- Model: `https://huggingface.co/lcybuaa/Change-Agent/tree/main`
- Code: `https://github.com/Chen-Yang-Liu/Change-Agent`

**Licence**
**MIT**, but the repo adds the restriction that **"the code can be used for academic purposes only."** That's a contradiction in the repo's own terms (MIT permits commercial use; the note forbids it). For a hackathon it is fine. If a judge asks about productisation, the honest answer is "we would need to clarify licensing with the authors."

**Hardware needed for inference**
- *Stated in paper:* trained on a **single NVIDIA RTX 4090**, Adam, lr 1e-4, up to 200 epochs.
- *My estimate:* the backbone is a **Siamese weight-shared SegFormer-B1 — roughly 14M parameters.** This is by far the smallest model on this list. Inference is well under 2GB and would run on a free Kaggle T4 with room to spare; even CPU inference is plausible. **We could realistically retrain this ourselves inside our compute budget**, which is not true of anything else here.

**What it scored, in its own paper (LEVIR-MCI dataset)**

| Task | Metric | Result |
|---|---|---|
| Change detection | MIoU | **86.43%** |
| Change captioning | BLEU-4 | 65.95 |
| Change captioning | METEOR | 40.80 |
| Change captioning | CIDEr-D | 140.29 |

The agent layer is built on the `lagent` framework; **the paper does not name which LLM drives the agent**, which is a real gap if we want to cite it as an architectural precedent.

> **Why this matters to us:** these numbers are on LEVIR-MCI, not CDVQA, so they don't transfer directly to our score. But it is the one candidate small enough that our own compute constraints are genuinely a non-issue.

---

## 4. GeoGround

*GeoGround: A Unified Large Vision-Language Model for Remote Sensing Visual Grounding*

**Is there a public checkpoint?**
Yes. The repo states the weights are released and "can be run directly with LLaVA."

**Hugging Face link**
- Model: `https://huggingface.co/erenzhou/GeoGround`
- Dataset (refGeo): `https://huggingface.co/datasets/erenzhou/refGeo` — 161k image–text pairs over 80k RS images
- Code: `https://github.com/VisionXLab/GeoGround`

**Licence**
⚠️ **No licence section in the repo README, and none found on the model page.** This is an open item — I could not verify it. **Someone needs to either find a LICENSE file in the repo or email the authors before we build anything on it.** Treat as "unknown, assume restrictive" until confirmed.

**Hardware needed for inference**
- *Stated:* nothing. Neither the paper excerpt nor the repo documents training or inference hardware.
- *My estimate:* the architecture is CLIP-ViT + 2-layer MLP connector + **Vicuna 1.5** LLM — i.e. a LLaVA clone, almost certainly 7B. Same profile as TEOChat: **~14–15GB fp16, ~5–6GB at 4-bit**, so quantised inference on a T4 should work. Unverified.

**What it scored, in its own paper**
The paper reports **best performance across all REC (referring expression comprehension) benchmarks, surpassing the specialist model on the DIOR-RSVG test set**, plus strong results on oriented-bounding-box grounding and referring segmentation, and it introduces a generalised multi-target REC benchmark built on AVVG.

⚠️ **I could not extract the actual numbers.** The results are in tables rendered as images in the HTML version, which I could not read. **The numbers exist in the PDF — someone needs to open `arxiv.org/abs/2411.11904` and fill in the DIOR-RSVG figures.** I have left this deliberately blank rather than guess.

**No RSVQA or VQA results reported** — this is a grounding-only model.

---

## 5. RSGPT

*RSGPT: A Remote Sensing Vision Language Model and Benchmark* — ISPRS J. Photogrammetry & Remote Sensing

**Is there a public checkpoint?**
❌ **No.** Code was released around May 2025, but **the repo contains no weights, no download links, and no Hugging Face repo.** The README describes it as "an ongoing project."

**Hugging Face link**
**None exists.** Code only: `https://github.com/Lavender105/RSGPT`

**Licence**
Academic use only — the repo states images and DOTA annotations "can be used for academic purposes only, but any commercial use is prohibited."

**Hardware needed for inference**
- *Stated in paper:* trained on **8× NVIDIA A100** for 5 epochs at batch size 64. The training script uses `torchrun --nproc_per_node=8`.
- *My estimate:* moot, since there are no weights. For reference, the architecture is InstructBLIP with a frozen EVA-G image encoder and a **Vicuna 7B/13B** LLM — the 13B variant would be far outside our budget even quantised.

**What it scored, in its own paper**

| Benchmark | Result |
|---|---|
| **RSVQA-LR** | **92.29% average accuracy** |
| RSVQA-HR test set 1 | 92.00% |
| RSVQA-HR test set 2 | 89.78% |
| RSIEval (own benchmark) VQA | 65.24% avg over 10 question categories |
| UCM-Captions | BLEU-1 86.12 (+4.55% over prior SOTA) |
| Sydney-Captions | BLEU-1 82.26 (+3.89%) |
| RSICD captions | BLEU-1 70.32 (+3.42%) |

> ⚠️ **Do not build any plan around RSGPT.** The RSVQA-LR number is the best on this page and it is unreachable — no weights, 8×A100 to reproduce. It belongs in our slides as a *literature reference point* ("published SOTA on RSVQA-LR is 92.29%"), never as a component. If it stays in our architecture diagram, a judge who knows the field will catch it.

---

## Open items I could not close

| # | Item | Who should close it |
|---|---|---|
| 1 | **GeoGround licence** — not stated anywhere I could find | Tech team: check for a LICENSE file in the repo, else email authors |
| 2 | **GeoGround DIOR-RSVG numbers** — in image-rendered tables | Anyone: open the arXiv PDF and read the results table |
| 3 | **Which LLM drives the Change-Agent agent layer** — unnamed in paper | Tech team, if we cite it as precedent |
| 4 | **TEOChat exact parameter count** — not stated on the model card | Tech team: check `config.json` on the HF repo |
| 5 | All VRAM figures marked *estimate* | Tech team: confirm by actually loading one on Kaggle |

---

## Sources

- [TEOChat — Hugging Face](https://huggingface.co/jirvin16/TEOChat) · [GitHub](https://github.com/ermongroup/TEOChat) · [arXiv 2410.06234](https://arxiv.org/abs/2410.06234) · [full text](https://arxiv.org/html/2410.06234v1)
- [RemoteCLIP — GitHub](https://github.com/ChenDelong1999/RemoteCLIP) · [arXiv 2306.11029](https://arxiv.org/abs/2306.11029)
- [Change-Agent — GitHub](https://github.com/Chen-Yang-Liu/Change-Agent) · [arXiv 2403.19646](https://arxiv.org/html/2403.19646v2)
- [GeoGround — GitHub](https://github.com/VisionXLab/GeoGround) · [arXiv 2411.11904](https://arxiv.org/abs/2411.11904)
- [RSGPT — GitHub](https://github.com/Lavender105/RSGPT) · [arXiv 2307.15266](https://arxiv.org/pdf/2307.15266)
