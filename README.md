# Pet Parenting Style (PPS) — Streamlit app

A research-based Streamlit application for administering the Pet Parenting Style questionnaire and generating profile-based results for pet caregivers.

The underlying questionnaire is based on the 36-item Pet Parenting Style scale developed in research led by Lauren Brubaker under the supervision of Dr. Monique Udell within the OSU Human–Animal Interaction context. This repository contains Alisa Tananaeva's Streamlit prototype for practical administration, score calculation, and profile visualization.

`pps_app.py` is the current main app: one question per screen, progress bar, Back/Continue, a clean result screen (including mixed-style output when classification is borderline), and an optional research consent flow (consent → contact → optional demographics → completion). The legacy one-page prototype is kept in `archive/PPS_1.py`.

> **Research prototype** · Python 3.10+ & Streamlit · Functional research app with optional Supabase storage

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)

---

## What this app does

- Presents **36 five-point Likert** items (frequency: *Never* … *Always*) to pet caregivers.
- Computes **three subscale means** (12 items each): Permissive, Authoritative, Authoritarian.
- Assigns a **most probable parenting style** by Euclidean distance from the respondent’s \([P, A_{auth}, A_{authn}]\) profile to three fixed **centroids**, after per-style **bias multipliers** (\(\beta_{perm}\), \(\beta_{authv}\), \(\beta_{authn}\)).
- Computes normative **z-scores and approximate percentiles** internally using embedded reference values (N = 953; normal-CDF approximation). The user-facing result screen is focused on the **style result**, a readable **interpretation**, and a donut chart of model-based similarity (\(\exp(-d_{eff})\)).
- Optional post-questionnaire flow (mirrors `DSLQ_App`): **research consent**, optional **future-contact**, optional **demographics**, and **completion** screen.

---

## Features

| Area | Detail |
|------|--------|
| Items | 36 caregiver self-report statements, English |
| Scales | 12 + 12 + 12 items → three means (1–5) |
| Classification | Nearest centroid after bias-weighted distance |
| Norms | Embedded means/SDs (953-pet reference sample) |
| Privacy | Answers live in the current Streamlit session unless you explicitly opt in to share data for research |
| Copy | UI text is loaded from `PPS_App_Copy_Extended.csv` |
| Optional modules | Optional demographics are defined in `PPS_App_OptionalModules.csv` |
| Storage | Supabase (optional) or local JSON fallback (see `pps_research.py`) |

---

## Repository layout (high level)

```
PPS/
├── pps_app.py                     # Main Streamlit app (one question per screen)
├── pps_scoring_baseline.py        # Single source of truth for scoring constants + math
├── pps_mixed_ambiguous_logic.py   # Borderline / mixed-style thresholds + labels
├── pps_research.py                # Export payload + optional Supabase/local persistence
├── PPS_App_Copy_Extended.csv      # UI copy (key,text)
├── PPS_App_OptionalModules.csv    # Optional demographics module definitions
├── .streamlit/
│   └── secrets.toml.example       # Supabase secrets template (optional)
├── archive/
│   ├── PPS_1.py                   # Legacy one-page prototype (baseline)
│   ├── PPS_Scoring.py             # Archived batch scoring helper (not used by app)
│   └── PPS_App_Copy_ru.csv        # Archived RU copy draft (not used by app)
├── docs/
│   └── PPS_Stage1_Baseline.md     # Baseline freeze note (documentation)
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
cd PPS
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 -m streamlit run pps_app.py
```

Open the local URL (typically `http://localhost:8501`). Answer all items, then proceed through consent/contact (optional) and view your result.

To run the legacy baseline one-page app:

```bash
python3 -m streamlit run archive/PPS_1.py
```

---

## Optional: Supabase setup

If you want to persist consented research data:

1. Copy `./.streamlit/secrets.toml.example` to `./.streamlit/secrets.toml`
2. Fill in your Supabase project URL and publishable key
3. Create Supabase tables:
   - `pps_sessions`
   - `pps_contacts`

The payload shape is defined in `pps_research.py` (see `_supabase_pps_insert` and `_supabase_pps_contact_insert`).

---

## Scoring logic (summary)

1. **Likert → numbers:** *Never* = 1 … *Always* = 5.
2. **Subscale means:** unweighted mean of the 12 items in each subscale.
3. **Profile vector:** \([M_{perm}, M_{authv}, M_{authn}]\) in that order (matching centroid coordinates in code).
4. **Distance:** Euclidean distance from the profile to each style’s **centroid** (fixed constants aligned with the original model).
5. **Bias:** multiply each style’s distance by its \(\beta\) (`BETA_PERM`, `BETA_AUTHV`, `BETA_AUTHN`); smallest **effective** distance wins.
6. **Normative scores:** z = \((M - \mu) / \sigma\) using embedded \(\mu,\sigma\) per subscale; percentile ≈ \(\Phi(z) \times 100\) (normal approximation).
7. **Donut “similarity”:** \(\exp(-d_{eff})\) per style (larger = closer to that centroid after bias).
8. **Mixed / ambiguous display:** when the margin between the two closest effective distances is < \(1/24\), the app presents a mixed-style interpretation.

Other files in this repository may hold supporting tables or drafts; they are **not** needed to run the Streamlit prototype.

---

## Tech stack

| Layer | Tool |
|--------|------|
| UI | [Streamlit](https://streamlit.io) |
| Numeric / tables | NumPy, pandas |
| Charts | [Altair](https://altair-viz.github.io/) |

---

## Author

**Alisa Tananaeva**  
Animal behavior & welfare scientist — [alicetananaeva.com](https://alicetananaeva.com)

The **Pet Parenting Style questionnaire** was developed in the research program described above; this Streamlit implementation is maintained here as a separate, practical layer on top of that work.

---

## Project status

- Functional research app (DSLQ-style UI flow)
- Main entry point: `pps_app.py`
- Legacy baseline kept: `archive/PPS_1.py`
- Further interface refinement and interpretation layers may be added later

---

## License

Shared for **portfolio and research transparency**. The questionnaire content and scoring logic are tied to ongoing academic work. **Contact the repository maintainer** before reusing items, centroids, or norms in derivative instruments or commercial products.
