# Compute Infrastructure Research

**Task:** Compute options for the geosatellite project  
**Date:** 2026-08-23  
**Status:** Partially verified. Account creation/phone verification and college confirmation require human action or institution-specific access.

## 1. Kaggle

### Account checklist

Create **three separate Kaggle accounts**, one for each technical team member, and phone-verify each account.

- [ ] Technical member 1: Kaggle account created
- [ ] Technical member 1: phone verified
- [ ] Technical member 1: GPU quota recorded
- [ ] Technical member 1: session limit recorded
- [ ] Technical member 1: trivial GPU notebook completed
- [ ] Technical member 2: Kaggle account created
- [ ] Technical member 2: phone verified
- [ ] Technical member 2: GPU quota recorded
- [ ] Technical member 2: session limit recorded
- [ ] Technical member 2: trivial GPU notebook completed
- [ ] Technical member 3: Kaggle account created
- [ ] Technical member 3: phone verified
- [ ] Technical member 3: GPU quota recorded
- [ ] Technical member 3: session limit recorded
- [ ] Technical member 3: trivial GPU notebook completed

**Important:** Account creation and phone/OTP verification cannot be completed by this document/research process because they require the individual members' identities, phone numbers, OTPs, and direct interaction with Kaggle.

### Current Kaggle GPU facts

Kaggle's current notebook documentation states that GPU notebook sessions can run for **up to 12 hours**, while Kaggle's GPU tips page states an **individual GPU session can run up to 9 hours**. This discrepancy should be treated as a documentation inconsistency and verified from the actual account/notebook UI before using the value as a project commitment.

Kaggle's GPU tips page currently states **up to 30 GPU hours per week**. The actual accelerator quota is account-dependent and should be recorded from each member's Kaggle quota display.

Kaggle's documented GPU configurations include:

| Kaggle option | GPU | VRAM | Other documented resources |
|---|---|---:|---|
| P100 | 1 × NVIDIA Tesla P100 | **16 GB** | 4 CPU cores, 29 GB RAM |
| T4 ×2 | 2 × NVIDIA Tesla T4 | **16 GB each / 32 GB aggregate** | 4 CPU cores, 29 GB RAM |

Kaggle's notebook documentation lists P100 and T4×2 configurations. NVIDIA's specifications confirm **16 GB** memory for the P100 configuration used by Kaggle and **16 GB GDDR6** per T4.

### Trivial GPU verification notebook

Use this minimal test:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    print("GPU computation successful:", y.shape)
else:
    print("GPU NOT AVAILABLE")
```

For each account, save a screenshot showing:
1. GPU enabled in notebook settings.
2. `CUDA available: True`.
3. GPU name.
4. Reported VRAM.
5. Successful GPU computation.

## 2. SIH 2026 cloud/GPU credits

The official Smart India Hackathon website confirms that **SIH 2026 is the current edition** and identifies Space Technology as one of its themes. The official site provides SIH process information and a contact route (`sih@aicte-india.org`), but the publicly accessible SIH 2026 material checked here does **not establish a general cloud-credit/GPU-credit allocation for participating teams**.

**Current conclusion: NOT CONFIRMED.**

Do not claim that SIH provides cloud credits unless the team receives written confirmation from SIH/MIC/AICTE or an official 2026 participant document explicitly stating the credit amount, provider, eligibility, and expiry.

### Direct question to SIH

> Does SIH 2026 provide participating teams with cloud computing credits, GPU credits, or sponsored cloud infrastructure? If yes, please provide the provider, credit amount, eligible services, eligibility conditions, activation process, and expiry date.

Official SIH contact listed on the SIH website:
- `sih@aicte-india.org`
- 011 29581222
- 011 29581223
- 011 29581239
- 011 29581240
- 011 29581241
- 011 29581319

## 3. College GPU access

**Status: PENDING.**

A straight answer requires the actual college/institution and its current lab policy. No college-specific GPU inventory or access policy was available in the task information.

Ask the college's lab administrator / department / SPOC these exact questions:

1. Does the college currently provide students access to GPU-equipped systems for project work?
2. Which lab/system has the GPU?
3. Exact GPU model?
4. Exact VRAM?
5. How many GPUs are available?
6. Can SIH teams use them for project development?
7. Is access available outside class hours?
8. Is remote/SSH access available?
9. Is installation of CUDA/PyTorch/other frameworks permitted?
10. Is there a booking or quota system?
11. What is the maximum continuous usage time?
12. Is internet access available from the GPU machine?
13. Can large satellite datasets be downloaded/processed?
14. Who grants access and what approval is required?

**Required evidence:** Get the response in writing (email/message) and record the exact GPU model and VRAM.

## 4. Compute comparison

| Option | Availability | GPU | VRAM | Time/quota | Verification |
|---|---|---|---:|---|---|
| Kaggle P100 | Public cloud | Tesla P100 | 16 GB | Up to 30 GPU h/week stated by Kaggle; session limit needs account/UI verification because Kaggle docs currently conflict | 3 member accounts + GPU test |
| Kaggle T4×2 | Public cloud | 2 × Tesla T4 | 32 GB aggregate | Same GPU quota pool; session limit needs account/UI verification | 3 member accounts + GPU test |
| SIH cloud credits | Unknown | Unknown | Unknown | Unknown | Official written confirmation required |
| College GPU lab | Unknown | Unknown | Unknown | Unknown | College written confirmation required |

## 5. Current findings

### Confirmed
- Kaggle provides remote GPU notebook environments.
- Kaggle documents P100 and T4×2 GPU notebook configurations.
- P100: 16 GB VRAM in the Kaggle configuration.
- T4: 16 GB VRAM per GPU; T4×2 therefore provides 32 GB aggregate GPU memory.
- Kaggle currently advertises up to 30 GPU hours/week on its GPU tips page.

### Not yet confirmed
- The exact current GPU quota for each of the three new accounts.
- Phone verification for the three members.
- The exact session limit shown for each account.
- Whether SIH 2026 provides cloud/GPU credits.
- Which GPU hardware the college currently provides.
- Whether the college permits SIH-team access and under what schedule.

## 6. Evidence to collect

For the final project documentation, collect:

- Kaggle account/quota screenshots for all 3 technical members.
- GPU notebook output for all 3 members.
- Screenshot of GPU model and VRAM.
- Official SIH response/document concerning cloud credits.
- College email/message confirming GPU access.
- GPU model + VRAM + number of machines at the college.
- Access restrictions and usage limits.

## Sources

- Kaggle Notebooks documentation: https://www.kaggle.com/docs/notebooks
- Kaggle GPU tips: https://www.kaggle.com/page/GPU-tips-and-tricks
- NVIDIA Tesla P100 specifications: https://www.nvidia.com/en-au/data-center/tesla-p100/
- NVIDIA T4 specifications: https://www.nvidia.com/en-in/data-center/tesla-t4/
- Smart India Hackathon official website: https://www.sih.gov.in/
