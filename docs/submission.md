# Submission Details

**Owner:** Person F · **Last updated:** 23 Aug 2026
**Confidence key:** ✅ verified against a primary or near-primary source · ⚠️ from a secondary source, must be confirmed with our SPOC

---

## 1. The problem statement

| Field | Value | Confidence |
|---|---|---|
| **PS ID** | **SIH26167** (PS Number `26167`) | ✅ |
| **Title** | SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries | ✅ |
| **Organisation** | Indian Space Research Organisation (ISRO) — listed via **Space Applications Centre (SAC)** | ✅ |
| **Department** | Department of Space / ISRO | ✅ |
| **Category** | **Software** | ✅ |
| **Theme** | **Space Technology** | ✅ |
| **Reference link given in the PS** | `https://arxiv.org/abs/2603.29630` | ✅ |
| **Contact / YouTube** | Not provided | ✅ |

**Official PS description (as published):** an agentic vision-language system for analysing satellite imagery through natural-language queries, handling single optical/SAR images, cross-modal paired imagery, and multi-temporal data. Core requirements: remote-sensing adaptation of at least one visual component using **BigEarthNet or open-source training data**; mandatory VQA on single images; change detection from bi-temporal pairs; optical–SAR joint analysis; automated model orchestration; interactive interface.

> **📌 The single most useful thing I found.** The arXiv link embedded in the problem statement is **not** a generic reference — it is **BigEarthNet.txt**, and it looks like it was put there on purpose.
>
> It is **464,044 co-registered Sentinel-1 SAR + Sentinel-2 multispectral image pairs with 9.6M text annotations**, covering three annotation types: LULC captions, **VQA pairs**, and **referring expressions with bounding boxes**. The paper reports that fine-tuning on it gives consistent gains across all tasks, and that VLMs otherwise struggle with complex LULC classification.
>
> That means one dataset covers our optical–SAR requirement, our VQA requirement, our grounding option, *and* the "adapt using BigEarthNet" instruction — all in one, already co-registered. **This should go to the tech team today**, ahead of anything else in this doc. If we are not using it, we need a reason.

**Competitive note:** at the time of the data snapshot this PS showed **0 of 500 submission slots used**. That is likely stale by now and I would not repeat it in front of judges, but it suggests this PS is not heavily contested — probably because it is hard.

---

## 2. Deadline

| Milestone | Date | Confidence |
|---|---|---|
| **PS submission deadline** | **20 September 2026** | ⚠️ from the portal mirror, not read off sih.gov.in directly |
| Screening / evaluation of idea PPTs | Oct – Nov 2026 | ⚠️ |
| National shortlist announced | Oct – Nov 2026 | ⚠️ |
| Grand Finale | December 2026 | ⚠️ |

**That is 28 days from today (23 Aug 2026).**

Surrounding timeline, for context (⚠️ all secondary):
- SPOC registration ran June – Aug 2026; the deadline was extended to **14 August 2026** — so our SPOC must already be registered. **If they are not, nothing else in this document matters. Check this first.**
- Problem statements released July – Aug 2026.
- Internal college hackathon: Aug – Sept 2026 — the college's internal round comes *before* national submission, so our real deadline is whatever our college sets, not the 20th.

> **Action:** the 20 September date and our internal-round date both need confirming with our SPOC **this week**. Every published mirror of the SIH timeline disagrees slightly with the others, and only the SPOC has the authoritative version. I would not plan against these dates until that conversation has happened.

---

## 3. Required format

**Verified rules:**

- ✅ Submission is an **idea presentation, uploaded by the SPOC** — individual teams do not submit to the national portal themselves. The SPOC uploads the idea PPT (and, per one source, a video demonstration) for national screening after the internal hackathon.
- ✅ SIH publishes an **official IDEA Presentation Format template** at `sih.gov.in` (the 2025 file lived at `sih.gov.in/letters/SIH2025-IDEA-Presentation-Format.pptx`; the 2026 equivalent should be in the same place). **Teams are expected to use it as-is.**
- ✅ The template's standard section structure, consistent across recent years:

  1. **Title page** — team name, PS number, PS title, theme, category
  2. **Idea / proposed solution** — description, how it addresses the problem, innovation and uniqueness
  3. **Technical approach** — technologies used, methodology, flow diagrams
  4. **Feasibility and viability** — feasibility analysis, potential challenges, strategies to overcome them
  5. **Impact and benefits** — impact on the target audience; social/economic/environmental benefits
  6. **Research and references** — links and citations

⚠️ **Not yet verified — I could not find authoritative answers and do not want to guess:**

| Question | Status |
|---|---|
| Exact slide count for 2026 (historically ~6) | Unconfirmed |
| Upload file type — PPT or PDF | Unconfirmed (historically **PDF**) |
| File size limit | Unconfirmed |
| File naming convention | Unconfirmed (historically `SIH2026_<PSnumber>_<TeamName>`) |
| Whether a demo video is mandatory or optional | Sources conflict |
| Whether a team may submit to more than one PS | Unconfirmed |

The official sih.gov.in FAQ page is currently serving **SIH 2020 content**, which is why these are open. **The reliable route is our SPOC, not the public site.**

---

## 4. Team eligibility rules

⚠️ Secondary sources, but consistent across all of them and unchanged for several years:

- **Exactly 6 members**, including the team leader — matches our team.
- **Minimum 1 female member — mandatory.** We need to confirm our roster satisfies this.
- **All 6 members from the same institution.**
- Software-category teams are expected to be well versed in programming.

---

## 5. What I need from the SPOC — one message, this week

Suggested text to send:

> Hi, we're a 6-member team preparing for SIH 2026 on PS **SIH26167** (SatQuery AI, ISRO/Space Applications Centre, Software / Space Technology). Could you confirm:
> 1. Our college's internal hackathon date and what we need to submit for it
> 2. The national idea-submission deadline — we have 20 September 2026, is that right?
> 3. The official SIH 2026 IDEA presentation template file, and the required slide count, file format, size limit, and file naming convention
> 4. Whether a demo video is required alongside the PPT
> 5. Whether a team may submit against more than one problem statement
>
> Thanks!

---

## 6. Checklist

- [ ] **Send the BigEarthNet.txt link to the tech team** — highest priority item in this doc
- [ ] Confirm SPOC is registered
- [ ] Send the message in §5 and log the answers back here
- [ ] Download the official SIH 2026 IDEA template once confirmed
- [ ] Confirm our team has ≥1 female member and all 6 are from the same college
- [ ] Get our internal hackathon date into the team calendar
- [ ] Re-check the PS page on sih.gov.in for any amendment to the statement before we submit

---

## Sources

- [SIH 2026 problem statements dataset — SIH26167](https://raw.githubusercontent.com/vedantchalke36/sih-2026-problem-statements/main/ps_2026/SIH26167.md) · [repo](https://github.com/vedantchalke36/sih-2026-problem-statements)
- [SIH 2026 all problem statements (PDF mirror)](https://sih-2026-problem-statements.shaikrohit187.workers.dev/public/pdfs/SIH_2026_All_PS.pdf)
- [BigEarthNet.txt — arXiv 2603.29630](https://arxiv.org/abs/2603.29630) *(the link given inside the PS)*
- [SIH 2026 official timeline](https://techpathdaily.com/sih-2026-official-timeline-is-here/) · [SIH 2026 schedule](https://thenewviews.com/smart-india-hackathon/)
- [SIH 2026 registration and team rules](https://blogs.reskilll.com/smart-india-hackathon-2026-complete-guide-registration-themes-winning/)
- [SIH official FAQs](https://www.sih.gov.in/faqs) *(currently serving 2020 content)*
- [SIH 2025 IDEA presentation format template](https://www.sih.gov.in/letters/SIH2025-IDEA-Presentation-Format.pptx)
